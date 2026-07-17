//! Request handlers: dispatch to the shared client.

use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse, Json, Response};
use warpllm::CreateChatCompletionRequest;

use crate::AppState;
use crate::error::{error_response, invalid_request_response};

pub(crate) async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok", "version": warpllm::version()}))
}

/// The gateway authenticates upstream with its own provider keys
/// (`OPENAI_API_KEY` for now; a configuration surface later). The caller's
/// Authorization header is ignored, never forwarded — providers each have
/// their own auth methods, so failover rules out per-caller passthrough.
pub(crate) async fn chat_completions(State(state): State<AppState>, body: Bytes) -> Response {
    // Manual deserialization: axum's Json rejections aren't OpenAI-shaped.
    let request: CreateChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(e) => return invalid_request_response(format!("could not parse request body: {e}")),
    };
    // Never log headers or bodies: they carry credentials and prompts.
    tracing::info!(
        model = %request.model,
        stream = request.stream.unwrap_or(false),
        "chat completion request"
    );
    // `stream: true` is rejected inside the client as NotImplemented → 501.
    match state.client.chat_completion(request).await {
        Ok(completion) => Json(completion).into_response(),
        Err(e) => error_response(&e),
    }
}
