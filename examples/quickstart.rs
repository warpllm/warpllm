//! One chat completion.
//!
//! ```sh
//! OPENAI_API_KEY=sk-... cargo run -p warpllm --example quickstart
//! ```
//!
//! Model strings are `provider/model`. The prefix is required: the roster
//! matches the whole string, so a bare `gpt-5-nano` is rejected rather than
//! guessed at.
//!
//! The subscriber is what makes the client's provider discovery visible.
//! warpllm emits it through `tracing` and installs no subscriber of its own, so
//! a library caller sees nothing until it opts in.

use warpllm::{ChatCompletionRequestMessage, Client, ClientConfig, CreateChatCompletionRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter("warpllm=info")
        .init();

    let client = Client::new(ClientConfig::default())?;

    let completion = client
        .chat_completions(CreateChatCompletionRequest {
            model: "openai/gpt-5-nano".to_string(),
            messages: vec![
                ChatCompletionRequestMessage::new("system", "You are a helpful assistant."),
                ChatCompletionRequestMessage::new("user", "Hello!"),
            ],
            ..Default::default()
        })
        .await?;

    let content = completion.choices[0].message.content.as_deref();
    println!("{}", content.unwrap_or_default());
    Ok(())
}
