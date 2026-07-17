//! Maps [`warpllm::Error`] onto HTTP statuses and the OpenAI error envelope
//! (`{"error": {"message", "type", "code"}}`) so official SDKs raise their
//! proper typed exceptions. `code` values reuse the stable
//! [`warpllm::Error::to_wire_json`] contract.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde_json::{Value, json};

pub fn openai_error_body(error: &warpllm::Error) -> (StatusCode, Value) {
    use warpllm::Error;
    let (status, error_type, code) = match error {
        Error::InvalidInput(_) | Error::InvalidModel { .. } => (
            StatusCode::BAD_REQUEST,
            "invalid_request_error".into(),
            "invalid_request",
        ),
        Error::MissingApiKey { .. } => (
            StatusCode::UNAUTHORIZED,
            "invalid_request_error".into(),
            "missing_api_key",
        ),
        // The upstream status passes through; a genuinely unknown model
        // surfaces here as the provider's own 404.
        Error::Provider {
            status, error_type, ..
        } => (
            StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY),
            error_type.clone().unwrap_or_else(|| "api_error".into()),
            "provider_error",
        ),
        Error::Network { .. } => (
            StatusCode::BAD_GATEWAY,
            "api_error".into(),
            "connection_error",
        ),
        Error::Decode { .. } => (StatusCode::BAD_GATEWAY, "api_error".into(), "decode_error"),
        Error::NotImplemented(_) => (
            StatusCode::NOT_IMPLEMENTED,
            "api_error".into(),
            "not_implemented",
        ),
    };
    let body = json!({
        "error": {
            "message": error.to_string(),
            "type": error_type,
            "code": code,
        }
    });
    (status, body)
}

pub fn error_response(error: &warpllm::Error) -> Response {
    let (status, body) = openai_error_body(error);
    (status, Json(body)).into_response()
}

pub fn invalid_request_response(message: String) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "invalid_request",
            }
        })),
    )
        .into_response()
}
