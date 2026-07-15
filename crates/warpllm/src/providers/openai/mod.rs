//! OpenAI provider: the wire format already matches `crate::types::openai`,
//! so endpoint impls only strip the `openai/` model prefix and map errors.
//! The error envelope below is provider-wide and shared by every endpoint.

pub(crate) mod chat;

use serde_json::Value;

use crate::error::Error;

pub(crate) const PROVIDER: &str = "openai";

/// OpenAI error bodies look like `{"error": {"message": ..., "type": ...}}`.
/// Unparseable bodies fall back to the raw text.
pub(crate) fn error_from_body(status: u16, body: &str) -> Error {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    let error = parsed.as_ref().map(|v| &v["error"]);
    Error::Provider {
        provider: PROVIDER,
        status,
        error_type: error.and_then(|e| e["type"].as_str()).map(str::to_string),
        message: error
            .and_then(|e| e["message"].as_str())
            .map(str::to_string)
            .unwrap_or_else(|| body.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unparseable_error_body_is_preserved() {
        let body = "x".repeat(1_024);
        let err = error_from_body(503, &body);

        match err {
            Error::Provider { message, .. } => assert_eq!(message, body),
            other => panic!("expected Provider error, got {other:?}"),
        }
    }
}
