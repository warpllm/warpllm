//! The one check a mock cannot stand in for: whether a REAL self-hosted
//! OpenAI-compatible server round-trips through warpllm.
//!
//! `tests/self_hosted.rs` proves everything warpllm itself does — which model
//! string routes where, which URL is built, which headers go out and which
//! deliberately do not. What it cannot prove is the half that belongs to the
//! backend: that vLLM's reply decodes into warpllm's shapes, that Ollama frames
//! its SSE the way the spec says, that llama.cpp's `usage` block is where a
//! reader expects. Only a real server answers that.
//!
//! Opt-in and ignored by default, because CI has no such server:
//!
//! ```text
//! WARPLLM_LOCAL_BASE_URL=http://localhost:8000/v1 \
//! WARPLLM_LOCAL_MODEL=meta-llama/Llama-3.3-70B-Instruct \
//!   cargo test -p warpllm --test live_self_hosted -- --ignored --nocapture
//! ```
//!
//! `WARPLLM_LOCAL_API_KEY` is optional and usually absent — that is the case
//! this whole feature exists for. Setting it exercises the other arm, for a box
//! that does want a token.
//!
//! Deliberately NOT a CI job. A real vLLM wants a GPU and its CPU build is
//! multi-gigabyte; `llama.cpp`'s server with a tiny GGUF is the lightest
//! genuinely-third-party option and still adds an external artifact, and
//! somebody else's failures, to a run that currently finishes in minutes. The
//! honest claim is "warpllm's half is covered by CI, and here is the command
//! that checks yours" — not a green badge over a backend nobody ran.

use std::io::Write;

use warpllm::{ChatCompletionRequestMessage, Client, ClientConfig, CreateChatCompletionRequest};

const BASE_URL: &str = "WARPLLM_LOCAL_BASE_URL";
const MODEL: &str = "WARPLLM_LOCAL_MODEL";
const API_KEY: &str = "WARPLLM_LOCAL_API_KEY";

/// The roster a reader would write for their own box, generated so the test
/// needs no file checked in — and so the file it does write is exactly the one
/// `examples/warpllm.yaml` documents.
fn roster(base_url: &str, model: &str, api_key_var: Option<&str>) -> String {
    let auth = match api_key_var {
        Some(var) => format!("    env_api_key: {var}\n"),
        None => "    auth: none\n".to_string(),
    };
    format!(
        "providers:\n  local:\n    base_url: \"{base_url}\"\n{auth}    models:\n      \
         local/live:\n        model: \"{model}\"\n        supported_apis:\n          \
         - {{api: openai_compat_chat_completions}}\n          \
         - {{api: openai_compat_chat_completions_stream}}\n"
    )
}

/// Both surfaces against one real backend: a whole reply, then a streamed one.
///
/// One test rather than two because the setup is a running server somebody
/// started by hand, and being told about half of it is worse than being told
/// about all of it at once.
#[test]
#[ignore = "needs a running OpenAI-compatible server; run with --ignored"]
fn a_real_self_hosted_server_round_trips() {
    let (Ok(base_url), Ok(model)) = (std::env::var(BASE_URL), std::env::var(MODEL)) else {
        println!("skipped: set {BASE_URL} and {MODEL} to point at your own server");
        return;
    };
    let api_key_var = std::env::var(API_KEY).ok().map(|_| API_KEY);
    println!(
        "checking {base_url} for `{model}` ({})",
        match api_key_var {
            Some(var) => format!("authenticating with {var}"),
            None => "no credential".to_string(),
        }
    );

    let dir = std::env::temp_dir().join("warpllm-live-self-hosted");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("warpllm.yaml");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(roster(&base_url, &model, api_key_var).as_bytes())
        .unwrap();

    let client = Client::new(ClientConfig {
        specs_path: Some(path),
        timeout_secs: Some(120),
        ..Default::default()
    })
    .expect("the roster this test wrote must load");

    let request = || CreateChatCompletionRequest {
        model: "local/live".to_owned(),
        messages: vec![ChatCompletionRequestMessage::new(
            "user",
            "Reply with exactly: hello",
        )],
        ..Default::default()
    };
    let runtime = tokio::runtime::Runtime::new().unwrap();

    let completion = runtime
        .block_on(client.chat_completions(request()))
        .expect("a whole reply from the local server");
    // The caller's own routing string, not the backend's echo of `model`.
    assert_eq!(completion.model, "local/live");
    assert!(!completion.choices.is_empty(), "{completion:?}");
    println!("whole reply: {:?}", completion.choices[0].message.content);

    let chunks = runtime.block_on(async {
        let mut stream = client
            .chat_completions_stream(request())
            .await
            .expect("a stream from the local server");
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.expect("every chunk must decode into warpllm's shapes"));
        }
        chunks
    });
    assert!(!chunks.is_empty(), "the stream carried no chunks");
    assert!(chunks.iter().all(|chunk| chunk.model == "local/live"));
    println!("stream: {} chunks", chunks.len());
}
