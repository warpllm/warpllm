//! What a Google failure MEANS: an error envelope in, warpllm's canonical
//! failure form out.
//!
//! One direction only, like [`anthropic`](crate::gateway::anthropic::error):
//! nothing answers in this protocol, so there is nothing to render.
//!
//! # One lookup, not two
//!
//! Google's envelope is `{"error": {"code": <http status>, "message": …,
//! "status": "RESOURCE_EXHAUSTED"}}`. The numeric `code` repeats the HTTP
//! status and the string `status` names the failure's family, so the family
//! plays the part Anthropic's `type` does and there is no per-failure `code` to
//! read. Only [`ErrorMapper::from_status_and_type`] is implemented.
//!
//! The status arms come first and the family arms second, which is Anthropic's
//! order and for its reason: the family only decides a failure whose status
//! said nothing, such as a mid-stream error event that arrives after a 200.
//!
//! The family names are the canonical gRPC status names. They are written from
//! knowledge of Google's error reference, not a fetched copy: check each
//! against <https://ai.google.dev/gemini-api/docs/troubleshooting> before
//! relying on one.

use std::time::Duration;

use serde_json::Value;

use crate::error::Error;
use crate::gateway::error::{Classified, ErrorMapper, MatchesProtocol, classify};
use crate::gateway::types::ProviderError;

/// Parses a Google error body, falling back to the raw text when it is not one.
///
/// `retry_after` comes from the response headers. Google's body carries no
/// request id, so the one passed in is the only one there is.
// Dead until the transport (subtask 6 of #35) calls it.
#[allow(dead_code)]
pub(crate) fn error_from_body(
    provider: &'static str,
    status: u16,
    body: &str,
    retry_after: Option<Duration>,
    request_id: Option<String>,
) -> Error {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    let error = parsed.as_ref().map(|value| &value["error"]);
    let field = |name: &str| error.and_then(|error| error[name].as_str());
    let classified = classify(&MatchesProtocol, &Google, status, field("status"), None);
    classified(Box::new(ProviderError {
        provider,
        status,
        message: field("message")
            .map(str::to_string)
            .unwrap_or_else(|| body.to_string()),
        error_type: field("status").map(str::to_string),
        // The envelope's `code` is the HTTP status again, not a per-failure slug.
        provider_code: None,
        retry_after,
        request_id,
        raw_body: ProviderError::bounded(body),
    }))
}

/// The error vocabulary every Google backend shares. Deliberately does not read
/// `message`, for the reason the Anthropic mapper gives.
struct Google;

impl ErrorMapper for Google {
    /// **429 is [`Error::RateLimited`] for both a rate limit and an exhausted
    /// quota.** Google uses one status and one family for both
    /// (`RESOURCE_EXHAUSTED`), and telling them apart would mean reading the
    /// message. So a quota failure reads as a rate limit here.
    ///
    /// **503 is [`Error::Overloaded`].** Google documents it as the service
    /// being temporarily overloaded or down.
    ///
    /// **404 is [`Error::ModelNotFound`]** on the same footing as Anthropic's:
    /// a model is the only resource a reachable request asks Google to find,
    /// because the request renderer refuses every media source that would name
    /// another (a URL, or another provider's file).
    ///
    /// **400, 500 and 504 mostly fall through** to `classify`'s residual, which
    /// already reads them. A 400 `FAILED_PRECONDITION` (a region or billing
    /// restriction) is the one an `INVALID_ARGUMENT` reading would misdescribe,
    /// but both are the caller's to fix, so both are `InvalidRequest`.
    fn from_status_and_type(&self, status: u16, error_type: Option<&str>) -> Option<Classified> {
        Some(match (status, error_type) {
            (401, _) => Error::Authentication,
            (403, _) => Error::PermissionDenied,
            (404, _) => Error::ModelNotFound,
            (429, _) => Error::RateLimited,
            (503, _) => Error::Overloaded,

            (_, Some("UNAUTHENTICATED")) => Error::Authentication,
            (_, Some("PERMISSION_DENIED")) => Error::PermissionDenied,
            (_, Some("NOT_FOUND")) => Error::ModelNotFound,
            (_, Some("RESOURCE_EXHAUSTED")) => Error::RateLimited,
            (_, Some("UNAVAILABLE")) => Error::Overloaded,
            // A timeout Google reported is an unattributed server fault, not
            // `StreamStalled`, which is warpllm reporting that IT stopped waiting.
            (_, Some("INTERNAL" | "DEADLINE_EXCEEDED")) => Error::ServerError,
            (_, Some("INVALID_ARGUMENT" | "FAILED_PRECONDITION" | "OUT_OF_RANGE")) => {
                Error::InvalidRequest
            }
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(status: &str, message: &str) -> String {
        serde_json::json!({"error": {"code": 0, "message": message, "status": status}}).to_string()
    }

    fn error(status: u16, body: &str) -> Error {
        error_from_body("google", status, body, None, None)
    }

    #[test]
    fn statuses_decide_first() {
        let body = envelope("INVALID_ARGUMENT", "x");
        // A status outranks a family that disagrees with it.
        assert!(matches!(error(429, &body), Error::RateLimited(_)));
        assert!(matches!(
            error(401, &envelope("", "x")),
            Error::Authentication(_)
        ));
        assert!(matches!(
            error(403, &envelope("", "x")),
            Error::PermissionDenied(_)
        ));
        assert!(matches!(
            error(404, &envelope("", "x")),
            Error::ModelNotFound(_)
        ));
        assert!(matches!(
            error(503, &envelope("", "x")),
            Error::Overloaded(_)
        ));
    }

    #[test]
    fn a_family_decides_when_the_status_says_nothing() {
        // A mid-stream error event arrives after a 200.
        for (family, check) in [
            (
                "UNAUTHENTICATED",
                (|e: &Error| matches!(e, Error::Authentication(_))) as fn(&Error) -> bool,
            ),
            ("PERMISSION_DENIED", |e| {
                matches!(e, Error::PermissionDenied(_))
            }),
            ("NOT_FOUND", |e| matches!(e, Error::ModelNotFound(_))),
            ("RESOURCE_EXHAUSTED", |e| matches!(e, Error::RateLimited(_))),
            ("UNAVAILABLE", |e| matches!(e, Error::Overloaded(_))),
            ("INTERNAL", |e| matches!(e, Error::ServerError(_))),
            ("DEADLINE_EXCEEDED", |e| matches!(e, Error::ServerError(_))),
            ("INVALID_ARGUMENT", |e| {
                matches!(e, Error::InvalidRequest(_))
            }),
            ("FAILED_PRECONDITION", |e| {
                matches!(e, Error::InvalidRequest(_))
            }),
        ] {
            assert!(check(&error(200, &envelope(family, "x"))), "{family}");
        }
    }

    #[test]
    fn plain_statuses_fall_through_to_the_residual() {
        assert!(matches!(
            error(400, &envelope("", "x")),
            Error::InvalidRequest(_)
        ));
        assert!(matches!(
            error(500, &envelope("", "x")),
            Error::ServerError(_)
        ));
    }

    #[test]
    fn a_body_that_is_not_an_envelope_still_classifies_by_status() {
        assert!(matches!(
            error(429, "<html>slow down</html>"),
            Error::RateLimited(_)
        ));
    }
}
