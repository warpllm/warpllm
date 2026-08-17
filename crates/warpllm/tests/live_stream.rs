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

/// One model per provider on the roster, each with the parameter that caps its
/// reply. Cheap and short-answered on purpose: this is a shape check, not a
/// capability survey.
///
/// The cap is NAMED PER MODEL because it genuinely differs: OpenAI's newer
/// families reject `max_tokens` outright — *"Unsupported parameter:
/// 'max_tokens' is not supported with this model"* — and require
/// `max_completion_tokens`, while the others still take `max_tokens`. warpllm
/// passes either through untouched rather than translating between them, so
/// the caller is what has to know, and here that is this table.
const MODELS: [(&str, &str); 3] = [
    ("openai/gpt-5-nano", "max_completion_tokens"),
    ("deepseek/deepseek-v4-flash", "max_tokens"),
    (
        "openrouter/~deepseek/deepseek-v4-flash-latest",
        "max_tokens",
    ),
];

#[tokio::test]
#[ignore = "needs live provider API keys; run with --ignored"]
async fn every_configured_provider_streams_chunks_warpllm_can_hold() {
    let mut checked = Vec::new();
    let mut skipped = Vec::new();

    for (model_str, cap) in MODELS {
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
            messages: vec![ChatCompletionRequestMessage::new(
                "user",
                "Reply with exactly: hello",
            )],
            ..Default::default()
        };
        // Both go through the catch-all — the job it exists for, on a request
        // rather than a reply. `max_completion_tokens` is genuinely unmodeled;
        // `max_tokens` is a modeled field, but naming it here keeps the two
        // spellings side by side instead of branching on which one this model
        // wants.
        //
        // The cap is generous rather than tight, because it exists to bound a
        // runaway and not to shape the answer. 16 does not survive a reasoning
        // model: the budget covers reasoning tokens too, so gpt-5-nano spent
        // all of it thinking and finished on `length` with nothing to show.
        // The prompt is what keeps the reply short.
        request
            .unknown_fields
            .insert(cap.into(), serde_json::json!(512));
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
    let mut reasons = Vec::new();
    for chunk in chunks {
        assert_eq!(
            chunk.model, model_str,
            "{model_str} chunk echoes the upstream name, not the caller's"
        );
        for choice in &chunk.choices {
            if let Some(Some(fragment)) = &choice.delta.content {
                text.push_str(fragment);
            }
            if let Some(reason) = &choice.finish_reason {
                reasons.push(reason.clone());
            }
        }
    }
    // The usage chunk is why the request asks for one: it is the only place
    // that says where the budget went, and a reply that is empty because the
    // model spent it all on reasoning looks exactly like one that is empty
    // because warpllm dropped the content.
    let usage = chunks.iter().rev().find_map(|chunk| chunk.usage.as_ref());

    // Printed rather than asserted. `include_usage` was asked for explicitly, so
    // a provider ignoring it is worth SEEING — but one provider honouring it is
    // not evidence the other two do, and an invariant drawn from one sample is
    // how a live check starts failing for reasons that are nobody's bug.
    println!(
        "{model_str}: {} chunks, finish_reason {reasons:?}, usage {}",
        chunks.len(),
        if usage.is_some() { "sent" } else { "ABSENT" }
    );

    assert!(
        !reasons.is_empty(),
        "{model_str} never sent a finish_reason"
    );
    assert!(
        !text.trim().is_empty(),
        "{model_str} streamed no content (finish_reason {reasons:?}, usage {usage:#?}) — \
         a reasoning model whose cap is too low finishes on `length` having emitted \
         only reasoning tokens"
    );
}
