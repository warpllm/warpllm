//! One chat completion against a model you run yourself.
//!
//! Start a server that speaks the OpenAI API — vLLM, TGI, Ollama, llama.cpp —
//! then point warpllm at a roster describing it. No key, no fork:
//!
//! ```sh
//! python -m vllm.entrypoints.openai.api_server \
//!   --model meta-llama/Llama-3.3-70B-Instruct
//! cargo run -p warpllm --example self_hosted
//! ```
//!
//! `warpllm.yaml` next door is the roster, and its comments are the tour. It is
//! MERGED over warpllm's built-in one, so `openai/gpt-5-nano` still routes from
//! this same client — the second call proves it, and needs `OPENAI_API_KEY`.
//!
//! The subscriber is worth keeping here. Loading a roster logs what it found,
//! and warns when your file replaced a provider warpllm ships — which is the
//! one thing about this feature you want to see rather than discover.

use warpllm::{ChatCompletionRequestMessage, Client, ClientConfig, CreateChatCompletionRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter("warpllm=info")
        .init();

    // Read once, here. A roster that cannot be used fails at THIS line, with
    // the path in the message — not on a request somewhere later.
    let client = Client::new(ClientConfig {
        specs_path: Some(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/warpllm.yaml").into(),
        ),
        ..Default::default()
    })?;

    let hello = |model: &str| CreateChatCompletionRequest {
        model: model.to_string(),
        messages: vec![ChatCompletionRequestMessage::new("user", "Hello!")],
        ..Default::default()
    };

    let local = client.chat_completions(hello("vllm/llama-3.3-70b")).await?;
    println!(
        "{}",
        local.choices[0]
            .message
            .content
            .as_deref()
            .unwrap_or_default()
    );

    // The same client still reaches everything warpllm ships. A roster of your
    // own adds to the list; it does not replace it.
    let shipped = client.chat_completions(hello("openai/gpt-5-nano")).await?;
    println!(
        "{}",
        shipped.choices[0]
            .message
            .content
            .as_deref()
            .unwrap_or_default()
    );
    Ok(())
}
