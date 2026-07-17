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

/// Runs the OpenAI-compatible gateway. `args` are CLI flags passed verbatim
/// to the shared Rust parser (`--host`, `--port`, `--timeout-secs`; see
/// `--help`), so every language wrapper exposes identical flags without its
/// own parsing. `--help` prints usage and resolves; otherwise the promise
/// never resolves on success — the server runs until the Node process exits
/// (Ctrl+C included; signal handling stays with Node, not tokio).
#[napi]
pub async fn serve(args: Vec<String>) -> napi::Result<()> {
    use warpllm_server::config::{Cli, parse_cli};
    match parse_cli(args.into_iter()) {
        Ok(Cli::Print(text)) => {
            print!("{text}");
            Ok(())
        }
        Ok(Cli::Run(config)) => {
            warpllm_server::init_tracing();
            warpllm_server::serve(config, std::future::pending())
                .await
                .map_err(|e| napi::Error::from_reason(e.to_string()))
        }
        Err(e) => Err(napi::Error::from_reason(e)),
    }
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
