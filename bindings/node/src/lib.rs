use std::sync::Arc;

use napi_derive::napi;

#[napi]
pub fn version() -> &'static str {
    warpllm::version()
}

#[napi]
pub async fn echo(msg: String) -> napi::Result<String> {
    warpllm::echo(&msg)
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Errors cross to JS as `Error` whose message is the wire-format JSON;
/// the TypeScript wrapper parses it into typed error classes.
fn wire_err(e: warpllm::Error) -> napi::Error {
    napi::Error::from_reason(e.to_wire_json())
}

#[napi]
pub struct Client {
    inner: Arc<warpllm::Client>,
}

#[napi]
impl Client {
    #[napi(constructor)]
    pub fn new(config_json: String) -> napi::Result<Self> {
        let config: warpllm::ClientConfig = serde_json::from_str(&config_json)
            .map_err(|e| wire_err(warpllm::Error::InvalidInput(e.to_string())))?;
        let inner = warpllm::Client::new(config).map_err(wire_err)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    #[napi]
    pub async fn chat_completion(&self, request_json: String) -> napi::Result<String> {
        let client = self.inner.clone();
        let request: warpllm::CreateChatCompletionRequest = serde_json::from_str(&request_json)
            .map_err(|e| wire_err(warpllm::Error::InvalidInput(e.to_string())))?;
        let completion = client.chat_completion(request).await.map_err(wire_err)?;
        serde_json::to_string(&completion)
            .map_err(|e| wire_err(warpllm::Error::InvalidInput(e.to_string())))
    }
}
