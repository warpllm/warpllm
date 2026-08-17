//! Anthropic's error envelope, as it arrives.
//!
//! Sibling in spirit to [`messages::types`](super::messages::types): shapes
//! only. Deciding which [`crate::Error`] one of these becomes is an ingest
//! conversion and lives at `crate::gateway::anthropic::error`.
//!
//! One direction only, which is the difference from
//! [`openai_compat::error`](crate::protocol::openai_compat::error). That
//! module models the envelope warpllm SENDS, because an OpenAI-compatible
//! surface is what callers program against. Anthropic is an egress protocol
//! here — warpllm speaks it to a provider and answers its own callers in their
//! protocol — so nothing renders this shape, and the exception-class
//! discriminant that surface needs has no counterpart here.
//!
//! <https://platform.claude.com/docs/en/api/errors>

use serde::{Deserialize, Serialize};

use crate::protocol::UnknownFields;

/// The whole error body: `{"type": "error", "error": {...}, "request_id": ...}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Conventionally `"error"`.
    #[serde(rename = "type")]
    pub r#type: String,
    pub error: ErrorBody,
    /// The upstream's own request identifier, repeated from the `request-id`
    /// header into the body — which is the only reason to model it: an error
    /// read from a body alone still carries what a provider's support desk
    /// asks for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

/// The `error` object.
///
/// Two fields, and NO `code`. That absence is the load-bearing difference from
/// OpenAI's envelope: with no per-failure slug, `type` and the HTTP status are
/// the entire taxonomy, so a distinction OpenAI draws in `code` — a quota
/// exhaustion against an ordinary rate limit, both 429 — has nowhere to live
/// here and must be drawn from the status alone. `ErrorMapper::from_code`
/// stays defaulted for this protocol for that reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorBody {
    /// The Anthropic error family, e.g. `invalid_request_error`,
    /// `overloaded_error`. Documented to grow, so it stays a `String`.
    #[serde(rename = "type")]
    pub r#type: String,
    pub message: String,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// The documented body, verbatim, including the `request_id` a caller
    /// would otherwise have to read off a header it no longer has.
    #[test]
    fn the_documented_envelope_round_trips() {
        let body = json!({
            "type": "error",
            "error": {
                "type": "not_found_error",
                "message": "The requested resource could not be found."
            },
            "request_id": "req_011CSHoEeqs5C35K2UUqR7Fy"
        });
        let parsed: ErrorResponse = serde_json::from_value(body.clone()).unwrap();
        assert_eq!(parsed.error.r#type, "not_found_error");
        assert_eq!(
            parsed.request_id.as_deref(),
            Some("req_011CSHoEeqs5C35K2UUqR7Fy")
        );
        assert_eq!(serde_json::to_value(&parsed).unwrap(), body);
    }

    /// A body with no `request_id` — the shape the streaming `error` event
    /// carries — must not grow a null one on the way back out.
    #[test]
    fn an_envelope_without_a_request_id_stays_without_one() {
        let body = json!({
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Overloaded"}
        });
        let parsed: ErrorResponse = serde_json::from_value(body.clone()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), body);
    }

    /// A `type` Anthropic adds tomorrow must reach the classifier rather than
    /// failing the parse — the whole taxonomy is documented to grow.
    #[test]
    fn an_unknown_error_type_parses_and_keeps_its_extra_fields() {
        let body = json!({
            "type": "error",
            "error": {"type": "quantum_error", "message": "?", "hint": "retry"}
        });
        let parsed: ErrorResponse = serde_json::from_value(body.clone()).unwrap();
        assert_eq!(parsed.error.r#type, "quantum_error");
        assert_eq!(serde_json::to_value(&parsed).unwrap(), body);
    }
}
