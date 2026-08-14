//! Every fixture in this directory's fixtures/ must survive a deserialize →
//! reserialize round trip byte-for-byte (as JSON values). Any field the
//! upstream API sends that we drop, rename, or mistype fails the diff.
//!
//! One subdirectory per reply shape: `completion/` holds whole
//! `chat.completion` bodies, `stream/` the `chat.completion.chunk` bodies a
//! `stream: true` request answers with.
//!
//! `doc-full-response.json` is hand-built from the API reference.

use warpllm::CreateChatCompletionResponse;
use warpllm::protocol::openai_compat::chat_completions::types::CreateChatCompletionStreamResponse;

use crate::assert_fixtures_round_trip;

mod losslessness;
mod reassembly;

fn fixtures(shape: &str) -> String {
    format!(
        "{}/tests/protocol/openai_compat/chat_completions/fixtures/{shape}",
        env!("CARGO_MANIFEST_DIR")
    )
}

#[test]
fn completion_fixtures_round_trip_losslessly() {
    assert_fixtures_round_trip::<CreateChatCompletionResponse>(&fixtures("completion"));
}

/// A chunk sends `"logprobs": null` on every chunk and `"usage": null` on
/// every chunk but the last, so these fixtures are also what proves an
/// explicit null does not come back out as an absent field.
#[test]
fn stream_fixtures_round_trip_losslessly() {
    assert_fixtures_round_trip::<CreateChatCompletionStreamResponse>(&fixtures("stream"));
}
