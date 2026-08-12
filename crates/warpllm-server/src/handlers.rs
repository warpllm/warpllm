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

/// The gateway authenticates upstream with its own provider keys (each
/// provider's env var; a configuration surface later). The caller's
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
    // The library streams; this surface does not yet, and the refusal is now
    // its own to state. `Client::chat_completions` refuses `stream: true` by
    // naming the Rust entrypoint that serves it — advice with nothing behind
    // it over HTTP, where there is no second endpoint to be sent to. So an
    // unimplemented SURFACE is what a caller is told, which is the truth here
    // and answers 501 rather than 400.
    if request.stream == Some(true) {
        return error_response(&warpllm::Error::NotImplemented(
            "streaming over the HTTP gateway",
        ));
    }
    match state.client.chat_completions(request).await {
        Ok(completion) => Json(completion).into_response(),
        Err(e) => error_response(&e),
    }
}
