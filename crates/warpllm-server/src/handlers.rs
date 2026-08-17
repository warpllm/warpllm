//! Request handlers: dispatch to the shared client.

use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json, Response};
use futures_util::stream::{self, Stream};
use warpllm::{ChatCompletionStream, CreateChatCompletionRequest};

use crate::AppState;
use crate::error::{error_response, invalid_request_response, openai_error_body};

/// The payload that ends an OpenAI-compatible stream — the same sentinel the
/// client's transport consumes on the way in, re-emitted here on the way out.
const DONE: &str = "[DONE]";

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
    // `stream` picks the reply's SHAPE, and the two shapes share nothing past
    // this line — one status and one body, or a status and then events. So
    // this route parses, logs, and routes, and neither shape is written in
    // here as the default with the other as an aside.
    if request.stream == Some(true) {
        streamed_reply(state, request).await
    } else {
        whole_reply(state, request).await
    }
}

/// One status and one JSON body, which is what a request without `stream`
/// asks for.
async fn whole_reply(state: AppState, request: CreateChatCompletionRequest) -> Response {
    match state.client.chat_completions(request).await {
        Ok(completion) => Json(completion).into_response(),
        Err(e) => error_response(&e),
    }
}

/// Server-Sent Events, the shape an OpenAI SDK expects from `stream: true`.
///
/// The status is decided ONCE, here, before a byte of the body exists — which
/// is the only moment it can be. A refusal at this point is an ordinary error
/// response and keeps everything that makes one useful: its real status, its
/// `Retry-After`, the provider's own message. Past this point the 200 is
/// committed and a failure has to travel in the body; see [`error_event`].
async fn streamed_reply(state: AppState, request: CreateChatCompletionRequest) -> Response {
    match state.client.chat_completions_stream(request).await {
        Ok(chunks) => Sse::new(events(chunks)).into_response(),
        Err(e) => error_response(&e),
    }
}

/// The chunk stream as SSE events, terminated by [`DONE`].
///
/// The state is `Option`, not the stream alone: the upstream running out is not
/// the end of this stream, because one event — the sentinel, or a failure — is
/// still owed. `None` is what says that last event has been sent.
fn events(
    chunks: ChatCompletionStream,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    stream::unfold(Some(chunks), |state| async move {
        let mut chunks = state?;
        Some(match chunks.next().await {
            Some(Ok(chunk)) => match serde_json::to_string(&chunk) {
                Ok(json) => (Ok(Event::default().data(json)), Some(chunks)),
                // Serializing a chunk warpllm just built cannot fail, but
                // reporting it beats panicking midway through a response.
                Err(e) => (
                    Ok(error_event(&warpllm::Error::Internal(e.to_string()))),
                    None,
                ),
            },
            // Terminal, matching the client: whatever produced this also ended
            // the stream, so no sentinel follows it.
            Some(Err(e)) => (Ok(error_event(&e)), None),
            None => (Ok(Event::default().data(DONE)), None),
        })
    })
}

/// A failure that arrived after the response committed to 200.
///
/// There is no status left to send, so it travels as the last event, carrying
/// the same OpenAI envelope [`openai_error_body`] builds for a failed request —
/// an SDK reading this finds the fields it already knows.
///
/// No [`DONE`] follows it. The sentinel means the model finished; sending one
/// after a failure would tell a caller the answer is complete when it is
/// truncated, and OpenAI does not send one either.
fn error_event(error: &warpllm::Error) -> Event {
    let (_, body) = openai_error_body(error);
    Event::default().data(body.to_string())
}
