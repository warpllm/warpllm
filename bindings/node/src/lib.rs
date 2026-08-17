use std::sync::Arc;

use napi_derive::napi;
use tokio::sync::Mutex;

#[napi]
pub fn version() -> &'static str {
    warpllm::version()
}

/// Errors cross to JS as `Error` whose message is the wire-format JSON;
/// the TypeScript wrapper parses it into `WarpLLMError`.
fn wire_err(e: warpllm::Error) -> napi::Error {
    napi::Error::from_reason(e.to_openai_json())
}

#[napi]
pub struct Client {
    inner: Arc<warpllm::JsonClient>,
}

#[napi]
impl Client {
    #[napi(constructor)]
    pub fn new(config_json: String) -> napi::Result<Self> {
        let inner = warpllm::JsonClient::new(&config_json).map_err(wire_err)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    #[napi]
    pub async fn chat_completions(&self, request_json: String) -> napi::Result<String> {
        let client = self.inner.clone();
        client
            .chat_completions(&request_json)
            .await
            .map_err(wire_err)
    }

    #[napi]
    pub async fn chat_completions_stream(&self, request_json: String) -> napi::Result<ChatStream> {
        let client = self.inner.clone();
        let stream = client
            .chat_completions_stream(&request_json)
            .await
            .map_err(wire_err)?;
        Ok(ChatStream {
            inner: Arc::new(Mutex::new(stream)),
        })
    }
}

/// A handle over one streamed reply. The iteration protocol is TypeScript's —
/// `client.ts` wraps this in `Symbol.asyncIterator` — because napi's own
/// `Generator` trait is synchronous and cannot express `for await`.
#[napi]
pub struct ChatStream {
    /// `Arc<Mutex<_>>` rather than `&mut self`: a napi async method has to hand
    /// the runtime a `'static` future, which a borrow of `self` is not. The
    /// lock also makes concurrent `next()` calls queue rather than race, which
    /// is the only sane reading of two callers pulling one stream.
    inner: Arc<Mutex<warpllm::JsonChatStream>>,
}

#[napi]
impl ChatStream {
    /// The next chunk as JSON, or `null` once the stream ends.
    #[napi]
    pub async fn next(&self) -> napi::Result<Option<String>> {
        let stream = self.inner.clone();
        let mut stream = stream.lock().await;
        stream.next().await.transpose().map_err(wire_err)
    }
}
