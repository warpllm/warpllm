// Shared across test binaries; not every binary uses every helper.
#![allow(dead_code)]

use std::future::Future;

use serde_json::{Value, json};
use warpllm::{ChatCompletionRequestMessage, Client, ClientConfig, CreateChatCompletionRequest};
use wiremock::MockServer;

/// The keys the helpers below put in the environment. Tests assert on these as
/// the expected bearer token.
pub const OPENAI_KEY: &str = "sk-test-openai";
pub const DEEPSEEK_KEY: &str = "sk-test-deepseek";
pub const OPENROUTER_KEY: &str = "sk-test-openrouter";

/// Client with the base URL pointed at the mock server. It carries no key —
/// the client reads the environment as it is BUILT, resolving every provider
/// it can authenticate once — so every test using this must run inside
/// [`with_openai_key`] or [`with_deepseek_key`], and must construct the client
/// inside that closure rather than before it.
pub fn client_for(server: &MockServer) -> Client {
    Client::new(ClientConfig {
        base_url: Some(server.uri()),
        timeout_secs: Some(5),
        stream_read_timeout_secs: None,
    })
    .unwrap()
}

/// Runs `body` with `var` set to `key`.
///
/// Env mutation is process-global and `unsafe` since edition 2024, so
/// temp-env's lock serializes it — which is why these tests are `#[test]` with
/// their own runtime rather than `#[tokio::test]`, and why they do not run in
/// parallel with each other.
///
/// EVERY env-touching test in this binary must go through a LOCKING temp-env
/// helper like this one. `temp_env::async_with_vars` cannot hold the lock
/// across an await, so mixing it in would let a test that SETS a key run
/// concurrently with the one asserting that key is MISSING, and that flakes.
///
/// The future is built before the variable is set, which is safe because
/// futures are lazy: nothing in `body` runs until `block_on` polls it.
pub fn with_env_key<F: Future<Output = ()>>(var: &str, key: &str, body: F) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    temp_env::with_var(var, Some(key), || runtime.block_on(body));
}

pub fn with_openai_key<F: Future<Output = ()>>(body: F) {
    with_env_key("OPENAI_API_KEY", OPENAI_KEY, body);
}

pub fn with_deepseek_key<F: Future<Output = ()>>(body: F) {
    with_env_key("DEEPSEEK_API_KEY", DEEPSEEK_KEY, body);
}

pub fn with_openrouter_key<F: Future<Output = ()>>(body: F) {
    with_env_key("OPENROUTER_API_KEY", OPENROUTER_KEY, body);
}

pub fn request(model: &str) -> CreateChatCompletionRequest {
    CreateChatCompletionRequest {
        model: model.into(),
        messages: vec![ChatCompletionRequestMessage::new("user", "hi")],
        ..Default::default()
    }
}

pub fn openai_completion_body() -> Value {
    json!({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "gpt-5.6-2024-08-06",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there!", "refusal": null},
            "finish_reason": "stop",
            "logprobs": null
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21,
            "prompt_tokens_details": {
                "cached_tokens": 3,
                "cache_write_tokens": 2,
                "audio_tokens": 0
            },
            "completion_tokens_details": {
                "reasoning_tokens": 5,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0
            }
        },
        "service_tier": "default",
        "system_fingerprint": "fp_44709d6fcb"
    })
}
