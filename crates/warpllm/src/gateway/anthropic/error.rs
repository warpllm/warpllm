//! What an Anthropic failure MEANS: an error envelope in, warpllm's canonical
//! failure form out.
//!
//! One direction only. Its [`openai_compat`](crate::gateway::openai_compat::error)
//! counterpart has an egress half because warpllm ANSWERS in that protocol;
//! nothing answers in this one, so there is nothing here to render.
//!
//! Sibling of [`api`](super::api) rather than a child of one of its surfaces:
//! the envelope is protocol-wide, shared by every endpoint Anthropic serves.
//!
//! # Two signals, not three
//!
//! [`ErrorMapper`] offers `from_code`, `from_status` and `from_type`, and this
//! protocol implements only the last two. Anthropic sends **no `code`** — the
//! envelope is `{type, message}` and nothing else — so a distinction OpenAI
//! draws with a slug has to be drawn from the status here or not at all.
//!
//! That is not a gap to work around. `classify` asks the strongest signal
//! first, and with the strongest one absent the ordering does more work than it
//! does next door: `invalid_request_error` is Anthropic's documented catch-all
//! for "other 4XX status codes not listed", so reading the family before the
//! status would report a 402 billing failure as a malformed request. The status
//! table below is deliberately the wider of the two.

use std::time::Duration;

use serde_json::Value;

use crate::error::Error;
use crate::gateway::error::{Classified, ErrorMapper, MatchesProtocol, classify};
use crate::gateway::types::ProviderError;

/// Anthropic error bodies look like
/// `{"type": "error", "error": {"type": …, "message": …}, "request_id": …}`.
/// Unparseable bodies fall back to the raw text.
///
/// `retry_after` comes from the response HEADERS, which is the only place it
/// exists. `request_id` is read from the header too, but Anthropic also repeats
/// it in the BODY — so unlike its openai_compat counterpart this one falls back
/// to the body's copy, which is what a body read from a log still has.
pub(crate) fn error_from_body(
    provider: &'static str,
    status: u16,
    body: &str,
    retry_after: Option<Duration>,
    request_id: Option<String>,
) -> Error {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    let error = parsed.as_ref().map(|v| &v["error"]);
    let field = |name: &str| error.and_then(|e| e[name].as_str());
    // No provider on this protocol has a vocabulary of its own to consult; see
    // this module's docs on why there is no `provider_overrides` directory.
    let classified = classify(&MatchesProtocol, &Anthropic, status, field("type"), None);
    classified(Box::new(ProviderError {
        provider,
        status,
        message: field("message")
            .map(str::to_string)
            .unwrap_or_else(|| body.to_string()),
        error_type: field("type").map(str::to_string),
        // The field this protocol does not have. Left `None` rather than
        // filled from `type`, which is the FAMILY and is already reported as
        // one — copying it across would invent a per-failure slug Anthropic
        // never sent and make the two fields look like corroboration.
        provider_code: None,
        retry_after,
        request_id: request_id.or_else(|| {
            parsed
                .as_ref()
                .and_then(|v| v["request_id"].as_str())
                .map(str::to_string)
        }),
        raw_body: ProviderError::bounded(body),
    }))
}

/// The error vocabulary every backend speaking this protocol shares.
///
/// Deliberately does not sniff `message`, for the reason its openai_compat
/// counterpart states: free text is what providers reword without notice, and a
/// substring match on it misclassifies silently.
pub(super) struct Anthropic;

impl ErrorMapper for Anthropic {
    /// Anthropic publishes one status per failure and a matching `type` for
    /// each, so the status table is complete rather than the residue it is on a
    /// protocol with per-failure codes.
    ///
    /// **529** is the one that has to be here. It is Anthropic's own overload
    /// status, outside any range `classify`'s residual reads — 5xx there stops
    /// at 599 but means [`Error::ServerError`], so a 529 left unmapped would
    /// report a retriable overload as an unattributed server fault and lose the
    /// one thing a caller does about it.
    ///
    /// **402** is [`Error::QuotaExceeded`], not [`Error::RateLimited`]: the
    /// account has a billing problem, and backing off does not fix one. This is
    /// the distinction OpenAI draws with an `insufficient_quota` code, and the
    /// reason the split matters even more here — the status is the ONLY place
    /// it can be drawn.
    ///
    /// **413** is [`Error::ContextLengthExceeded`]. Anthropic's is a byte limit
    /// on the request rather than a token limit on the context, and they are
    /// not the same measure — but they are the same PROBLEM to the caller, who
    /// fixes both by sending less, and there is no other variant that says so.
    ///
    /// **400, 500 and 504 are absent**, and adding them would be wrong:
    /// `classify`'s residual already reads them as `InvalidRequest` and
    /// `ServerError`, and naming them here would only outrank the `type` family
    /// for the statuses whose family is most likely to say something more
    /// specific.
    ///
    /// **404 is [`Error::ModelNotFound`] because a model is the only resource
    /// any reachable path asks Anthropic to find** — NOT because 404 means
    /// that. Anthropic answers an inaccessible file id with the same status and
    /// the same `not_found_error` family, and the two are separable only by the
    /// free-text `message`, which this mapper refuses to read for the reason
    /// stated above it. What makes the mapping true is that a file id cannot be
    /// sent: a foreign one is refused in `request::render_source`, and a native
    /// one needs an Anthropic-shaped ingress that does not exist. Both halves
    /// are pinned by `a_file_reference_cannot_reach_anthropic` below, so this
    /// comment cannot quietly become false.
    ///
    /// Files API support is what breaks it, and that work owes this mapping a
    /// second look. Delineating them then means using what warpllm SENT — it
    /// knows whether the request carried a file reference — never what
    /// Anthropic wrote back.
    fn from_status(&self, status: u16) -> Option<Classified> {
        Some(match status {
            401 => Error::Authentication,
            402 => Error::QuotaExceeded,
            403 => Error::PermissionDenied,
            404 => Error::ModelNotFound,
            413 => Error::ContextLengthExceeded,
            429 => Error::RateLimited,
            529 => Error::Overloaded,
            _ => return None,
        })
    }

    /// The `type` family — the weaker of the two signals this protocol has, and
    /// the one that only decides a failure whose status said nothing.
    ///
    /// In practice that is a mid-stream `error` event, which arrives with no
    /// status at all because the HTTP exchange already answered 200. Every
    /// family below is reachable that way, which is why the table mirrors the
    /// status one rather than stopping at the families a 4xx can carry.
    fn from_type(&self, error_type: &str) -> Option<Classified> {
        Some(match error_type {
            "authentication_error" => Error::Authentication,
            "billing_error" => Error::QuotaExceeded,
            "permission_error" => Error::PermissionDenied,
            "not_found_error" => Error::ModelNotFound,
            "request_too_large" => Error::ContextLengthExceeded,
            "rate_limit_error" => Error::RateLimited,
            "overloaded_error" => Error::Overloaded,
            // A 504 from Anthropic's own edge. NOT `Error::StreamStalled`,
            // however closely the words match: that variant is warpllm
            // reporting that IT stopped waiting, carries the timeout it
            // enforced, and holds no provider evidence — so it cannot even be
            // constructed here. An upstream that timed out and said so is an
            // unattributed 5xx, which is what `ServerError` means.
            "api_error" | "timeout_error" => Error::ServerError,
            // Anthropic's catch-all, and the reason it is read LAST: it is
            // documented as covering "other 4XX status codes not listed",
            // which makes it the family that means least.
            "invalid_request_error" => Error::InvalidRequest,
            // `conflict_error` (409) is deliberately absent, and it is the one
            // family here that falls all the way through to `Error::Unknown` —
            // `classify`'s residual reads 400 and 422, not the whole 4xx range.
            // That is the honest answer rather than a gap: a conflict means a
            // resource changed underneath the request, which nothing warpllm
            // sends to `/messages` can do, and the nearest variant
            // (`InvalidRequest`) would tell a caller to go fix a payload that
            // is fine. `Unknown` says nothing and keeps the status and the
            // family intact for whoever reads it.
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(error_type: &str, message: &str) -> String {
        serde_json::json!({
            "type": "error",
            "error": {"type": error_type, "message": message},
            "request_id": "req_body"
        })
        .to_string()
    }

    fn coded(status: u16, body: &str) -> &'static str {
        error_from_body("anthropic", status, body, None, None).code()
    }

    /// What makes `404 => ModelNotFound` true, tested where the claim is made.
    ///
    /// Anthropic answers an inaccessible file id with the identical status and
    /// family, so the mapping is honest only while a file id cannot be sent.
    /// This pins the half that is code — the request renderer refuses a file
    /// reference that did not arrive with this protocol's own retained content
    /// — and it is the test that must fail when Files API support arrives, so
    /// that work cannot inherit a mapping that has quietly become a guess.
    ///
    /// The other half is an absence (no Anthropic-shaped ingress exists to
    /// produce a native reference) and is pinned next door, by
    /// `request::tests::an_anthropic_native_file_reference_still_round_trips`.
    #[test]
    fn a_file_reference_cannot_reach_anthropic() {
        use crate::gateway::anthropic::api::messages::render_request;
        use crate::gateway::types::{self, ContentBlock, MediaSource};

        let request = types::ChatRequest {
            model: "claude-opus-5".into(),
            messages: vec![types::Message {
                role: crate::gateway::types::Role::User,
                content: vec![ContentBlock::Document {
                    source: MediaSource::ProviderFile {
                        id: "file_011CNha8iCJcU1wXNR6q4V8w".into(),
                    },
                    title: None,
                    cache: None,
                }],
                ext: types::ProviderExt::new(),
            }],
            ..Default::default()
        };
        let error = render_request(&request, "anthropic", Some(64)).unwrap_err();
        assert!(
            matches!(error, Error::InvalidInput(_)),
            "a file reference reached the wire, so a 404 can no longer be read \
             as a missing MODEL: {error}"
        );
    }

    /// Every documented status, in one place, so a change to the table is a
    /// change to this list.
    #[test]
    fn the_documented_status_table() {
        let cases = [
            (400, "invalid_request_error", "provider_invalid_request"),
            (401, "authentication_error", "authentication"),
            (402, "billing_error", "quota_exceeded"),
            (403, "permission_error", "permission_denied"),
            (404, "not_found_error", "model_not_found"),
            // Deliberately unclassified — see `from_type`. Pinned so that
            // giving it a meaning later is a visible decision.
            (409, "conflict_error", "provider_unknown"),
            (413, "request_too_large", "context_length_exceeded"),
            (429, "rate_limit_error", "rate_limited"),
            (500, "api_error", "provider_server_error"),
            (504, "timeout_error", "provider_server_error"),
            (529, "overloaded_error", "overloaded"),
        ];
        for (status, error_type, expected) in cases {
            assert_eq!(
                coded(status, &envelope(error_type, "boom")),
                expected,
                "{status} {error_type}"
            );
        }
    }

    /// 529 is outside `classify`'s 5xx residual, so an unmapped one would
    /// report a retriable overload as an unattributed server fault. Pinned on
    /// its own because it is the single reason `from_status` exists at all.
    #[test]
    fn a_529_is_an_overload_rather_than_a_server_fault() {
        assert_eq!(coded(529, "not json at all"), "overloaded");
    }

    /// A mid-stream `error` event has no status to read — the HTTP exchange
    /// already answered 200 — so the family is the only signal left.
    #[test]
    fn a_family_decides_a_failure_that_arrived_after_a_200() {
        assert_eq!(
            coded(200, &envelope("overloaded_error", "busy")),
            "overloaded"
        );
        assert_eq!(
            coded(200, &envelope("billing_error", "unpaid")),
            "quota_exceeded"
        );
    }

    /// The catch-all family must not outrank a status that says something
    /// specific. Anthropic documents `invalid_request_error` as covering "other
    /// 4XX status codes not listed", so a 402 carrying it is a BILLING failure
    /// wearing the generic family — and reporting it as a malformed request
    /// would send the caller to fix a payload that is fine.
    #[test]
    fn a_status_outranks_the_catch_all_family() {
        assert_eq!(
            coded(402, &envelope("invalid_request_error", "payment required")),
            "quota_exceeded"
        );
    }

    /// A family Anthropic adds tomorrow must not be guessed at: the status
    /// answers, and the raw family survives on the error for a caller to read.
    #[test]
    fn an_unknown_family_falls_back_to_http_semantics() {
        let error = error_from_body(
            "anthropic",
            418,
            &envelope("quantum_error", "?"),
            None,
            None,
        );
        assert_eq!(error.code(), "provider_unknown");
        assert!(error.to_string().contains('?'), "{error}");
    }

    /// The message is the body's when there is no envelope to read one from —
    /// an HTML error page from a proxy in front of the API, say.
    #[test]
    fn an_unparseable_body_survives_as_the_message() {
        let error = error_from_body("anthropic", 502, "<html>bad gateway</html>", None, None);
        assert_eq!(error.code(), "provider_server_error");
        assert!(error.to_string().contains("bad gateway"), "{error}");
    }

    /// This protocol has no `code`, and the field must stay empty rather than
    /// be filled from the family — which is already reported as the family.
    #[test]
    fn no_code_is_invented_from_the_family() {
        let Error::RateLimited(details) = error_from_body(
            "anthropic",
            429,
            &envelope("rate_limit_error", "slow down"),
            None,
            None,
        ) else {
            panic!("expected a rate limit");
        };
        assert_eq!(details.error_type.as_deref(), Some("rate_limit_error"));
        assert_eq!(details.provider_code, None);
    }

    /// The header is authoritative, and the body's copy is the fallback — which
    /// is the whole reason Anthropic repeating it in the body is worth reading.
    #[test]
    fn the_request_id_falls_back_to_the_body() {
        let body = envelope("not_found_error", "nope");
        let from_body = error_from_body("anthropic", 404, &body, None, None);
        let Error::ModelNotFound(details) = from_body else {
            panic!("expected a model-not-found");
        };
        assert_eq!(details.request_id.as_deref(), Some("req_body"));

        let from_header = error_from_body("anthropic", 404, &body, None, Some("req_header".into()));
        let Error::ModelNotFound(details) = from_header else {
            panic!("expected a model-not-found");
        };
        assert_eq!(details.request_id.as_deref(), Some("req_header"));
    }
}
