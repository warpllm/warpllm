// Shared across test binaries; not every binary uses every helper.
#![allow(dead_code)]

use serde_json::{Value, json};
use warpllm::{ChatCompletionRequestMessage, Client, ClientConfig, CreateChatCompletionRequest};
use wiremock::MockServer;

/// Client with the base URL pointed at the mock server and a dummy key,
/// so no test depends on real env vars.
pub fn client_for(server: &MockServer) -> Client {
    Client::new(ClientConfig {
        openai_api_key: Some("sk-test-openai".into()),
        base_url: Some(server.uri()),
        timeout_secs: Some(5),
    })
    .unwrap()
}

pub fn request(model: &str) -> CreateChatCompletionRequest {
    CreateChatCompletionRequest {
        model: model.into(),
        messages: vec![ChatCompletionRequestMessage {
            role: "user".into(),
            content: "hi".into(),
            ..Default::default()
        }],
        ..Default::default()
    }
}

pub fn openai_completion_body() -> Value {
    json!({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "gpt-4o-2024-08-06",
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
