//! ERRORS, both directions: an OpenAI-compatible error body in, warpllm's
//! canonical failure form out — and back again, as OpenAI would have
//! reported it.
//!
//! Both halves are VOCABULARY, which is what a protocol module owns. Ingest
//! asks what a provider meant by what it said; egress asks what OpenAI would
//! have said about the same failure. Neither belongs beside the wire shapes,
//! and neither belongs to one API surface.
//!
//! Sibling of [`api`](super::api) rather than a child of one of its
//! surfaces: the error envelope is protocol-wide, shared by every endpoint
//! this protocol serves, so it belongs to the protocol and not to one API
//! surface.
//!
//! This is an ingest conversion like any other — a wire shape in, a canonical
//! shape out — which is why it lives HERE and not beside the wire types.
//! `protocol::openai_compat::chat_completions::transport` deliberately hands
//! back a non-2xx as DATA, saying that deciding which [`Error`] it becomes
//! "would mean `protocol` reaching into the conversion layer". This module is
//! that conversion layer.
//!
//! Two jobs, and only one of them is this protocol's alone:
//!
//! * EXTRACTION — pulling `type`, `code`, and `message` out of
//!   `{"error": {…}}`. Nothing but a protocol can do this, because the
//!   envelope shape is what a protocol IS. A backend that shapes its errors
//!   differently is a different protocol, not a provider override.
//! * VOCABULARY — which spelling means which failure. Stated here as the
//!   baseline every backend on this protocol shares, and overridden per
//!   provider in [`provider_overrides`](super::provider_overrides) only where one genuinely
//!   diverges.

use std::collections::BTreeMap;
use std::time::Duration;

use serde_json::Value;

use crate::error::Error;
use crate::gateway::error::{Classified, ErrorMapper, classify};
use crate::gateway::types::ProviderError;
use crate::protocol::openai_compat::error::{ErrorBody, ErrorClass, OpenAiError};

/// OpenAI-compatible error bodies look like
/// `{"error": {"message": ..., "type": ..., "code": ...}}`. Unparseable
/// bodies fall back to the raw text.
///
/// `retry_after` and `request_id` come from the response HEADERS, which is
/// the only place they exist — no error body carries either.
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
    let classified = classify(
        super::provider_overrides::error_mapper(provider),
        &OpenAiCompat,
        status,
        field("type"),
        field("code"),
    );
    classified(Box::new(ProviderError {
        provider,
        status,
        message: field("message")
            .map(str::to_string)
            .unwrap_or_else(|| body.to_string()),
        error_type: field("type").map(str::to_string),
        provider_code: field("code").map(str::to_string),
        retry_after,
        request_id,
        raw_body: ProviderError::bounded(body),
    }))
}

/// The error mapping every backend speaking this protocol shares.
///
/// PROTOCOL-LEVEL, not per-provider. DeepSeek, vLLM, SGLang and OpenRouter
/// all reuse OpenAI's envelope spellings, so gating these per provider would
/// mean a new roster entry silently lost the mapping until someone wrote Rust
/// for it. A provider states only its DELTA, next door.
///
/// Deliberately does NOT sniff `message`. Free text is the one part of an
/// error body providers reword without notice, and a substring match on it is
/// the exact staleness liability a capability table would be — it would
/// misclassify silently, which is worse than [`Error::Unknown`] saying
/// nothing. Unrecognized envelopes stay `Unknown` with the status and the raw
/// type intact for the caller to read.
pub(super) struct OpenAiCompat;

impl ErrorMapper for OpenAiCompat {
    fn from_code(&self, code: &str) -> Option<Classified> {
        Some(match code {
            "context_length_exceeded" => Error::ContextLengthExceeded,
            "content_filter" | "content_policy_violation" => Error::ContentFilter,
            "model_not_found" => Error::ModelNotFound,
            "rate_limit_exceeded" => Error::RateLimited,
            // NOT a rate limit, however much the 429 beside it looks like
            // one: the account is out of credit, and backing off does not buy
            // any. OpenAI ships both spellings.
            "insufficient_quota" | "billing_hard_limit_reached" => Error::QuotaExceeded,
            "invalid_api_key" => Error::Authentication,
            _ => return None,
        })
    }

    /// The envelope short of a `code`: a status, and the `type` family beside
    /// it. ON THIS PROTOCOL the status is the stronger of the two, and the
    /// arms below say so by being read first.
    ///
    /// `invalid_request_error` is why. It is OpenAI's CATCH-ALL, sent for a
    /// bad API key (401), an unknown model (404), and a genuinely malformed
    /// body alike, so reading it over an unambiguous 401 would report a
    /// credential failure as a bad request and send a caller to fix a payload
    /// that was fine. Note that this is a claim about OPENAI'S FAMILIES, not
    /// about HTTP — a provider whose `type` names one failure each is entitled
    /// to the opposite order, and `opencode` next door takes it.
    ///
    /// Only the statuses that mean ONE thing on this protocol are listed. 404
    /// is deliberately absent: it may name a model, a proxy route, or the
    /// endpoint itself, so it needs the family below before becoming
    /// [`Error::ModelNotFound`]. 402 is absent because it is not this
    /// protocol's — DeepSeek documents it, and states it itself.
    fn from_status_and_type(&self, status: u16, error_type: Option<&str>) -> Option<Classified> {
        Some(match (status, error_type) {
            (401, _) => Error::Authentication,
            (403, _) => Error::PermissionDenied,
            (429, _) => Error::RateLimited,
            (503, _) => Error::Overloaded,
            (_, Some("rate_limit_error")) => Error::RateLimited,
            (_, Some("overloaded_error")) => Error::Overloaded,
            (_, Some("authentication_error")) => Error::Authentication,
            (_, Some("permission_error")) => Error::PermissionDenied,
            (_, Some("not_found_error")) => Error::ModelNotFound,
            (_, Some("invalid_request_error")) => Error::InvalidRequest,
            (_, Some("api_error")) => Error::ServerError,
            _ => return None,
        })
    }
}

// ---------------------------------------------------------------------------
// EGRESS — a warpllm failure, rendered as OpenAI would have reported it.
//
// The mirror of everything above. Ingest asks "what did this provider mean";
// egress asks "what would OpenAI have said". Both live here because both are
// vocabulary, and vocabulary is what a protocol module owns.
// ---------------------------------------------------------------------------

/// What OpenAI answers for a failure warpllm has classified.
///
/// `None` in a field means warpllm has NO OPINION and the upstream's own
/// answer stands. That is the whole design: normalizing a failure warpllm
/// understood is the service it provides, and inventing an answer for one it
/// did not would be a guess wearing the same clothes.
struct Opinion {
    status: OpinionStatus,
    error_type: Option<&'static str>,
    code: Option<&'static str>,
}

enum OpinionStatus {
    /// warpllm knows what OpenAI answers for this failure, and says so even
    /// when the upstream said something else — DeepSeek's 402 for an
    /// exhausted balance is OpenAI's 429 `insufficient_quota`. A caller
    /// handling a billing failure should not have to know which provider
    /// served the request.
    Is(u16),
    /// No HTTP exchange happened at all, which is the case the official SDKs
    /// raise `APIConnectionError` for.
    NoResponse,
    /// Unclassified: the upstream's status stands, untouched.
    Upstream,
}

/// The one place a warpllm failure becomes OpenAI's vocabulary.
///
/// Codes are OpenAI's own spellings, not warpllm's. Where OpenAI has no
/// canonical slug for a failure the provider's own is used instead, and where
/// there is neither the field is simply absent — `code` is nullable, and a
/// wrong slug is worse than none.
fn opinion(error: &Error) -> Opinion {
    use OpinionStatus::{Is, NoResponse, Upstream};
    let (status, error_type, code) = match error {
        // ------------------------------------------- warpllm's own failures
        // No upstream to borrow from, and OpenAI has no vocabulary for "this
        // gateway could not route your call", so warpllm's slug stands in.
        Error::InvalidInput(_) | Error::InvalidModel { .. } => (
            Is(400),
            Some("invalid_request_error"),
            Some("invalid_request"),
        ),
        // Beside InvalidModel, its nearest sibling: warpllm will not route
        // this string, and the fix is in the request or in the deployment's
        // configuration. Deliberately NOT 401 — a caller told to check their
        // credentials for a provider they chose not to serve is the exact
        // confusion this variant exists to prevent — and not 404, which on a
        // gateway reads as "no such endpoint".
        Error::ProviderNotDeclared { .. } => (
            Is(400),
            Some("invalid_request_error"),
            Some("provider_not_declared"),
        ),
        // OpenAI answers a missing or unusable key the same way, so this one
        // does have a spelling to borrow.
        Error::MissingApiKey { .. } => (
            Is(401),
            Some("invalid_request_error"),
            Some("invalid_api_key"),
        ),
        Error::Network { .. } => (
            NoResponse,
            Some("api_connection_error"),
            Some("connection_error"),
        ),
        Error::Decode { .. } => (Is(502), Some("api_error"), Some("decode_error")),
        // A bad answer from upstream, like `Decode` — the difference is that
        // this one arrived incomplete rather than unreadable.
        Error::StreamTruncated { .. } => (Is(502), Some("api_error"), Some("stream_truncated")),
        // 504 rather than 502: the upstream did not answer badly, it stopped
        // answering — which is the one thing a gateway timeout means.
        Error::StreamStalled { .. } => (Is(504), Some("api_error"), Some("stream_stalled")),
        Error::NotImplemented(_) => (Is(501), Some("api_error"), Some("not_implemented")),
        Error::Internal(_) => (Is(500), Some("server_error"), Some("internal_error")),
        // The gateway's own configuration, never the caller's payload — so it
        // sits beside `Internal` at 500 rather than at 400, which would send
        // someone off to fix a request that was fine. Its own code, because
        // the remedy is a file the operator can open.
        Error::InvalidRoster(_) => (Is(500), Some("server_error"), Some("invalid_roster")),

        // ------------------------------------ the provider's, normalized
        Error::RateLimited(_) => (
            Is(429),
            Some("rate_limit_error"),
            Some("rate_limit_exceeded"),
        ),
        // The reason the internal enum earns its keep. Both of these are 429
        // to a caller, and only `code` separates a wait from a bill.
        Error::QuotaExceeded(_) => (
            Is(429),
            Some("insufficient_quota"),
            Some("insufficient_quota"),
        ),
        Error::Overloaded(_) => (Is(503), Some("server_error"), Some("server_error")),
        Error::ContextLengthExceeded(_) => (
            Is(400),
            Some("invalid_request_error"),
            Some("context_length_exceeded"),
        ),
        Error::ContentFilter(_) => (
            Is(400),
            Some("invalid_request_error"),
            Some("content_filter"),
        ),
        Error::Authentication(_) => (
            Is(401),
            Some("invalid_request_error"),
            Some("invalid_api_key"),
        ),
        Error::ModelNotFound(_) => (
            Is(404),
            Some("invalid_request_error"),
            Some("model_not_found"),
        ),
        // Classified, but OpenAI has no one slug for either — the provider's
        // own is more use than a made-up one.
        Error::PermissionDenied(_) => (Is(403), Some("invalid_request_error"), None),
        Error::InvalidRequest(_) => (Is(400), Some("invalid_request_error"), None),
        // An unattributed 5xx is already the status OpenAI would send, and
        // which 5xx it is came from the provider — nothing to improve on.
        Error::ServerError(_) => (Upstream, Some("server_error"), None),
        // All candidates in a failover list failed. A fixed 503 because no
        // single provider produced this — it is service-unavailable at the
        // gateway level, and borrowing one attempt's status would make the
        // caller's HTTP status depend on candidate order, and its retry_after
        // on a single provider's rate limit, for a failure that spans several
        // providers. The per-attempt detail rides in the body instead.
        Error::CandidatesExhausted { .. } => {
            (Is(503), Some("server_error"), Some("candidates_exhausted"))
        }
        // The failover chain exceeded the configured total deadline. Same
        // reasoning as StreamStalled below: an operational timeout is the
        // backend's fault, never the caller's input.
        Error::DeadlineExceeded { .. } => {
            (Is(504), Some("server_error"), Some("deadline_exceeded"))
        }
        // Nothing classified it. Everything the upstream said stands.
        _ => (Upstream, None, None),
    };
    Opinion {
        status,
        error_type,
        code,
    }
}

/// Renders a failure for every OpenAI-compatible surface warpllm serves.
pub(crate) fn to_openai(error: &Error) -> OpenAiError {
    let opinion = opinion(error);
    let upstream = error.provider_error();
    let mut headers = BTreeMap::new();
    if let Some(e) = upstream {
        // Both live only in the response headers, and both are things a
        // caller can act on — so they travel as headers here too rather than
        // being flattened into the body OpenAI has no field for them in.
        if let Some(retry_after) = e.retry_after {
            headers.insert("retry-after".to_string(), retry_after.as_secs().to_string());
        }
        if let Some(request_id) = &e.request_id {
            headers.insert("x-request-id".to_string(), request_id.clone());
        }
    }
    let status = match opinion.status {
        OpinionStatus::Is(status) => Some(status),
        OpinionStatus::NoResponse => None,
        OpinionStatus::Upstream => upstream.map(|e| e.status),
    };
    OpenAiError {
        class: ErrorClass::from_status(status),
        status,
        headers,
        error: ErrorBody {
            message: error.to_string(),
            error_type: opinion
                .error_type
                .map(str::to_string)
                .or_else(|| upstream.and_then(|e| e.error_type.clone()))
                .unwrap_or_else(|| "api_error".to_string()),
            param: None,
            code: opinion
                .code
                .map(str::to_string)
                .or_else(|| upstream.and_then(|e| e.provider_code.clone())),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classified(status: u16, body: &str) -> Error {
        error_from_body("demo", status, body, None, None)
    }

    /// The classified variant, named by its stable wire slug — which makes
    /// the tables below assert on the public contract rather than on an
    /// internal spelling.
    fn kind_of(status: u16, body: &str) -> &'static str {
        classified(status, body).code()
    }

    fn detail_of(status: u16, body: &str) -> ProviderError {
        let error = classified(status, body);
        error
            .provider_error()
            .unwrap_or_else(|| panic!("expected a provider failure, got {error:?}"))
            .clone()
    }

    #[test]
    fn unparseable_error_body_is_preserved() {
        let body = "x".repeat(1_024);
        let err = error_from_body("openai", 503, &body, None, None);
        assert_eq!(err.provider_error().unwrap().message, body);
    }

    /// Every classified failure is provider-driven, whatever the variant —
    /// the grouping the flat enum gives up in its shape.
    #[test]
    fn a_classified_failure_is_always_provider_driven() {
        for (status, body) in [(429, "{}"), (400, "not json"), (402, "{}")] {
            assert_eq!(
                classified(status, body).origin(),
                crate::error::Origin::Provider,
                "{status}"
            );
        }
    }

    /// The seam, end to end: the same status classifies differently by
    /// provider, because DeepSeek documents 402 as an exhausted balance and
    /// this protocol has no reading of it. Proves the override table is
    /// actually reached — the unit tests next door only prove the vocabulary
    /// itself is right.
    #[test]
    fn a_providers_own_vocabulary_reaches_the_classifier() {
        assert!(matches!(
            error_from_body("deepseek", 402, "{}", None, None),
            Error::QuotaExceeded(_)
        ));
        // Anyone else gets the protocol, which says nothing about 402 and
        // falls to the HTTP residual.
        assert!(matches!(
            error_from_body("openai", 402, "{}", None, None),
            Error::Unknown(_)
        ));
    }

    /// ...and a provider's status rule still loses to an explicit `code`,
    /// through the real dispatch rather than a stub vocabulary.
    #[test]
    fn a_providers_status_rule_loses_to_a_code_it_did_not_claim() {
        let body = r#"{"error":{"message":"m","code":"invalid_api_key"}}"#;
        assert!(matches!(
            error_from_body("deepseek", 402, body, None, None),
            Error::Authentication(_)
        ));
    }

    /// `code` is the most specific signal, so it decides even when the status
    /// and family would say something else. A context overflow is the case
    /// that matters: it arrives as a 400 `invalid_request_error`, and calling
    /// it `InvalidRequest` would tell a router to give up on a request that a
    /// bigger-window model would serve.
    #[test]
    fn code_outranks_type_and_status() {
        for (status, error_type, code, expected) in [
            (
                400,
                "invalid_request_error",
                "context_length_exceeded",
                "context_length_exceeded",
            ),
            (
                400,
                "invalid_request_error",
                "content_filter",
                "content_filter",
            ),
            (
                404,
                "invalid_request_error",
                "model_not_found",
                "model_not_found",
            ),
            // The 429 says rate limit; the code says the account is out of
            // credit. The code wins, and the difference is the whole point —
            // one is worth waiting out, the other never is.
            (
                429,
                "invalid_request_error",
                "insufficient_quota",
                "quota_exceeded",
            ),
            (
                429,
                "invalid_request_error",
                "billing_hard_limit_reached",
                "quota_exceeded",
            ),
            (
                401,
                "invalid_request_error",
                "invalid_api_key",
                "authentication",
            ),
        ] {
            let body =
                format!(r#"{{"error":{{"message":"m","type":"{error_type}","code":"{code}"}}}}"#);
            assert_eq!(kind_of(status, &body), expected, "{code}");
        }
    }

    /// The real 401: OpenAI sends its catch-all `invalid_request_error`
    /// family for a bad API key, an unknown model, and a malformed body
    /// alike. An unambiguous status has to win, or a credential failure gets
    /// reported as a bad payload and the caller goes looking in the wrong
    /// place.
    #[test]
    fn an_unambiguous_status_outranks_the_catch_all_family() {
        for (status, expected) in [
            (401, "authentication"),
            (403, "permission_denied"),
            (429, "rate_limited"),
            (503, "overloaded"),
        ] {
            let body = r#"{"error":{"message":"m","type":"invalid_request_error"}}"#;
            assert_eq!(kind_of(status, body), expected, "{status}");
        }
        // ...but where the status says nothing specific, the family still
        // decides.
        let body = r#"{"error":{"message":"m","type":"invalid_request_error"}}"#;
        assert_eq!(kind_of(400, body), "provider_invalid_request");
    }

    /// A 404 is NOT decisive on this protocol: it may name a model, a proxy
    /// route, or the endpoint itself. Without envelope evidence it stays a
    /// generic client error rather than claiming the model does not exist.
    #[test]
    fn a_bare_404_is_not_read_as_a_missing_model() {
        assert_ne!(kind_of(404, "{}"), "model_not_found");
        // ...but a 404 that NAMES the failure is.
        let body = r#"{"error":{"message":"m","type":"not_found_error"}}"#;
        assert_eq!(kind_of(404, body), "model_not_found");
    }

    /// Where the status is not decisive, the family decides over the residual
    /// status ranges.
    #[test]
    fn type_outranks_a_non_decisive_status() {
        for (error_type, expected) in [
            ("rate_limit_error", "rate_limited"),
            ("overloaded_error", "overloaded"),
            ("authentication_error", "authentication"),
            ("permission_error", "permission_denied"),
            ("not_found_error", "model_not_found"),
            ("invalid_request_error", "provider_invalid_request"),
            ("api_error", "provider_server_error"),
        ] {
            // A status that would classify differently on its own.
            let body = format!(r#"{{"error":{{"message":"m","type":"{error_type}"}}}}"#);
            assert_eq!(kind_of(418, &body), expected, "{error_type}");
        }
    }

    /// Plenty of OpenAI-compatible backends send a bare status with no
    /// envelope at all. The status is the last signal, never a guess beyond
    /// it.
    #[test]
    fn status_classifies_an_empty_envelope() {
        for (status, expected) in [
            (400, "provider_invalid_request"),
            (422, "provider_invalid_request"),
            (401, "authentication"),
            (403, "permission_denied"),
            (429, "rate_limited"),
            (503, "overloaded"),
            (500, "provider_server_error"),
            (502, "provider_server_error"),
        ] {
            assert_eq!(kind_of(status, "not json"), expected, "{status}");
        }
    }

    /// An envelope nothing recognizes must say nothing rather than guess.
    /// `Unknown` is a real answer here — the status and the raw type are both
    /// retained for the caller to read.
    #[test]
    fn an_unrecognized_envelope_is_unknown_and_keeps_the_raw_type() {
        let body = r#"{"error":{"message":"m","type":"vendor_specific_error","code":"weird"}}"#;
        assert!(matches!(classified(418, body), Error::Unknown(_)));
        let detail = detail_of(418, body);
        assert_eq!(detail.status, 418);
        assert_eq!(detail.error_type.as_deref(), Some("vendor_specific_error"));
    }

    /// The classification must never consume the provider's own spelling:
    /// the variant is what it means, `error_type` is what it said.
    #[test]
    fn classifying_retains_the_providers_own_spelling() {
        let body = r#"{"error":{"message":"slow down","type":"rate_limit_error"}}"#;
        assert!(matches!(classified(429, body), Error::RateLimited(_)));
        let detail = detail_of(429, body);
        assert_eq!(detail.error_type.as_deref(), Some("rate_limit_error"));
        assert_eq!(detail.message, "slow down");
    }

    /// Free text is the one part of an error body providers reword without
    /// notice, so it must not influence classification — a message that says
    /// "rate limit" under a 400 is still an invalid request.
    #[test]
    fn the_message_never_influences_classification() {
        let body = r#"{"error":{"message":"rate limit exceeded, context length exceeded"}}"#;
        assert_eq!(kind_of(400, body), "provider_invalid_request");
    }

    /// PROMOTE BUT RETAIN: the `code` is what the classification was READ
    /// from, so discarding it would throw away the strongest evidence in the
    /// envelope and leave a caller unable to check warpllm's reading.
    #[test]
    fn the_provider_code_survives_the_classification_it_produced() {
        let body = r#"{"error":{"message":"m","type":"invalid_request_error","code":"insufficient_quota"}}"#;
        let detail = detail_of(429, body);
        assert_eq!(detail.provider_code.as_deref(), Some("insufficient_quota"));
        assert_eq!(detail.error_type.as_deref(), Some("invalid_request_error"));
        // ...and the whole envelope is there as a last resort.
        assert!(detail.raw_body.contains("insufficient_quota"));
    }

    /// A vendor code nothing recognizes is exactly when the raw evidence
    /// matters most: warpllm has no answer, so the caller needs everything
    /// the provider said in order to form one.
    #[test]
    fn an_unrecognized_code_is_still_retained() {
        let body = r#"{"error":{"message":"m","type":"vendor_error","code":"tpm_exhausted_v2"}}"#;
        let detail = detail_of(418, body);
        assert_eq!(detail.provider_code.as_deref(), Some("tpm_exhausted_v2"));
        assert!(detail.raw_body.contains("tpm_exhausted_v2"));
    }

    /// `Retry-After` and the upstream request id live ONLY in headers, so
    /// they have to be threaded in from the transport — a body-only error
    /// path cannot recover them.
    #[test]
    fn header_evidence_reaches_the_error() {
        let err = error_from_body(
            "demo",
            429,
            r#"{"error":{"message":"slow down"}}"#,
            Some(Duration::from_secs(42)),
            Some("req-abc".into()),
        );
        assert!(matches!(err, Error::RateLimited(_)));
        let detail = err.provider_error().unwrap();
        assert_eq!(detail.retry_after, Some(Duration::from_secs(42)));
        assert_eq!(detail.request_id.as_deref(), Some("req-abc"));
    }

    /// An exhausted chain maps to a FIXED 503 — the gateway is unavailable,
    /// not one provider — with warpllm's own slug. It has no retry-after: no
    /// single provider produced this, so no single provider's backoff belongs
    /// in the response.
    #[test]
    fn an_exhausted_chain_is_a_gateway_wide_503() {
        let err = Error::CandidatesExhausted {
            models: vec!["openai/gpt-5.6".into(), "deepseek/deepseek-v4-flash".into()],
            tried: vec![
                ("openai/gpt-5.6".into(), Box::new(classified(429, "{}"))),
                (
                    "deepseek/deepseek-v4-flash".into(),
                    Box::new(classified(503, "{}")),
                ),
            ],
        };
        assert!(err.provider_error().is_none(), "no attempt IS the failure");
        let rendered = to_openai(&err);
        assert_eq!(rendered.status, Some(503));
        assert_eq!(rendered.error.error_type, "server_error");
        assert_eq!(rendered.error.code.as_deref(), Some("candidates_exhausted"));
        assert_eq!(
            rendered.headers.get("retry-after"),
            None,
            "one attempt's rate limit must not leak onto the gateway's verdict"
        );
        assert!(
            !rendered.headers.contains_key("x-request-id"),
            "same for a request id: several attempts, not one"
        );
    }

    /// A chain that ran out of TIME is the backend's fault — never the
    /// caller's payload — so it is a 504, and it carries no retry-after for
    /// the same reason an exhausted chain carries none.
    #[test]
    fn a_chain_deadline_is_a_gateway_504() {
        let err = Error::DeadlineExceeded {
            tried: vec![("openai/gpt-5.6".into(), Box::new(classified(429, "{}")))],
        };
        let rendered = to_openai(&err);
        assert_eq!(rendered.status, Some(504));
        assert_eq!(rendered.error.error_type, "server_error");
        assert_eq!(rendered.error.code.as_deref(), Some("deadline_exceeded"));
        assert!(rendered.headers.is_empty(), "{:?}", rendered.headers);
    }
}
