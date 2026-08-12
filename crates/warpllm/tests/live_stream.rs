//! The one check no fixture can stand in for: what providers actually put on
//! the wire today. Recorded transcripts age, and a provider that adds a field
//! or starts omitting one says nothing until something reads its real output.
//!
//! Opt-in and ignored by default, because CI needs no API keys and this needs
//! several:
//!
//! ```text
//! OPENAI_API_KEY=... DEEPSEEK_API_KEY=... \
//!   cargo test -p warpllm --test live_stream -- --ignored --nocapture
//! ```
//!
//! Every provider whose key is absent is skipped and named. A chunk that fails
//! to round-trip is printed in full: that output is a fixture, and belongs
//! under `fixtures/transcript/` so the keyless suite catches it from then on.
//!
//! This goes through [`Client::chat_completions_stream`], so it exercises the
//! whole path a caller uses — the request renderer, the SSE framing, the
//! gateway IR, and the render back. A chunk that reaches the assertions below
//! has already survived normalization, which is what makes the round trip it
//! then checks a statement about warpllm and not merely about serde.

use warpllm::{
    ChatCompletionRequestMessage, ClientConfig, CreateChatCompletionRequest,
    CreateChatCompletionStreamResponse, fetch_model,
};

/// One model per provider on the roster. Cheap and short-answered on purpose:
/// this is a shape check, not a capability survey.
const MODELS: [&str; 3] = [
    "openai/gpt-5-nano",
    "deepseek/deepseek-v4-flash",
    "openrouter/~deepseek/deepseek-v4-flash-latest",
];

#[tokio::test]
#[ignore = "needs live provider API keys; run with --ignored"]
async fn every_configured_provider_streams_chunks_warpllm_can_hold() {
    let mut checked = Vec::new();
    let mut skipped = Vec::new();

    for model_str in MODELS {
        let (provider, _) = fetch_model(model_str).expect("roster entry");
        if provider
            .env_api_key()
            .and_then(|name| std::env::var(name).ok())
            .is_none()
        {
            skipped.push(model_str);
            continue;
        }

        let mut request = CreateChatCompletionRequest {
            model: model_str.to_owned(),
            messages: vec![ChatCompletionRequestMessage {
                role: "user".into(),
                content: "Reply with exactly: hello".into(),
                unknown_fields: Default::default(),
            }],
            max_tokens: Some(16),
            ..Default::default()
        };
        // Unmodeled, and reaching the provider anyway — the catch-all doing
        // the job it exists for, on a request rather than a reply.
        request.unknown_fields.insert(
            "stream_options".into(),
            serde_json::json!({"include_usage": true}),
        );

        let client = warpllm::Client::new(ClientConfig {
            timeout_secs: Some(60),
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("{model_str}: {e}"));

        let mut stream = client
            .chat_completions_stream(request)
            .await
            .unwrap_or_else(|e| panic!("{model_str}: {e}"));

        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap_or_else(|e| panic!("{model_str}: {e}")));
        }
        assert_chunks_are_usable(model_str, &chunks);
        checked.push(model_str);
    }

    println!("streamed: {checked:?}");
    println!("skipped for want of a key: {skipped:?}");
    assert!(
        !checked.is_empty(),
        "no provider key was set, so nothing was checked: {skipped:?}"
    );
}

/// What a caller needs out of a live stream: text that adds up, a finish
/// reason, and the caller's own model string echoed back on every chunk.
fn assert_chunks_are_usable(model_str: &str, chunks: &[CreateChatCompletionStreamResponse]) {
    assert!(
        chunks.len() > 1,
        "{model_str} answered with no stream: {chunks:#?}"
    );

    let mut text = String::new();
    let mut finished = false;
    for chunk in chunks {
        assert_eq!(
            chunk.model, model_str,
            "{model_str} chunk echoes the upstream name, not the caller's"
        );
        for choice in &chunk.choices {
            if let Some(Some(fragment)) = &choice.delta.content {
                text.push_str(fragment);
            }
            finished |= choice.finish_reason.is_some();
        }
    }

    assert!(finished, "{model_str} never sent a finish_reason");
    assert!(!text.trim().is_empty(), "{model_str} streamed no content");
}
