//! Every fixture in this directory's `fixtures/` must survive a deserialize →
//! reserialize round trip (as JSON values). Any field Google sends that
//! warpllm drops, renames, or mistypes fails the diff.
//!
//! One subdirectory per shape: `request/` holds the bodies warpllm sends and
//! `response/` whole `GenerateContentResponse` replies. Streaming has no
//! subdirectory yet, because no streaming surface is implemented.
//!
//! `request/` earns a suite here for the same reason it does for Anthropic:
//! warpllm BUILDS these bodies and nobody else ever sees one, so a field
//! silently dropped on the way out would reach no other test.
//!
//! PROVENANCE: every fixture is hand-built from the API reference —
//! <https://ai.google.dev/api/generate-content> and the function-calling and
//! thinking guides under <https://ai.google.dev/gemini-api/docs> — not live
//! captures. They prove warpllm reads what Google DOCUMENTS, not what it
//! sends. The deliberate additions to the documented bodies are fields warpllm
//! does not model — `safetySettings`, `cachedContent`, `seed`,
//! `mediaResolution`, a `fileData` part, a built-in `googleSearch` tool,
//! `safetyRatings`, `groundingMetadata`, `avgLogprobs`,
//! `promptTokensDetails`, `trafficType` — which are exactly the
//! ones a round trip would silently lose.

use warpllm::protocol::google::generate_content::types::{
    GenerateContentRequest, GenerateContentResponse,
};

use crate::assert_fixtures_round_trip;

mod losslessness;

fn fixtures(shape: &str) -> String {
    format!(
        "{}/tests/protocol/google/generate_content/fixtures/{shape}",
        env!("CARGO_MANIFEST_DIR")
    )
}

#[test]
fn request_fixtures_round_trip_losslessly() {
    assert_fixtures_round_trip::<GenerateContentRequest>(&fixtures("request"));
}

/// These also pin the reply with NO candidates: a blocked prompt carries
/// `promptFeedback` instead, and must not be an error.
#[test]
fn response_fixtures_round_trip_losslessly() {
    assert_fixtures_round_trip::<GenerateContentResponse>(&fixtures("response"));
}
