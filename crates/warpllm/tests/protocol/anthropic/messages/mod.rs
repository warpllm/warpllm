//! Every fixture in this directory's `fixtures/` must survive a deserialize →
//! reserialize round trip byte-for-byte (as JSON values). Any field Anthropic
//! sends that warpllm drops, renames, or mistypes fails the diff.
//!
//! One subdirectory per shape: `message/` holds whole `Message` replies,
//! `stream/` the individual `MessageStreamEvent` payloads a `stream: true`
//! request is answered with, and `request/` the bodies warpllm sends.
//!
//! `request/` has no counterpart in the openai_compat suite, and earns one
//! here because of which direction this protocol runs in. There, the request
//! is what CALLERS build, so it is checked from the outside by the TypeScript
//! oracle. Here it is what warpllm builds and nobody else ever sees, so a
//! field silently dropped on the way out would reach no test at all without
//! these.
//!
//! PROVENANCE: every fixture is hand-built from the API reference —
//! <https://platform.claude.com/docs/en/api/messages> and
//! <https://platform.claude.com/docs/en/build-with-claude/streaming> — with
//! ids and token counts kept as the docs print them. They are not live
//! captures, so they prove warpllm reads what Anthropic DOCUMENTS; a live
//! capture from `tests/live_stream.rs` is what would prove it reads what
//! Anthropic sends. The deliberate additions to the documented bodies are
//! fields warpllm does not model — `stop_details`, `container`,
//! `cache_creation`, `output_tokens_details`, `service_tier`, `top_k`,
//! `metadata`, a document's `citations` — which are exactly the ones a round
//! trip would silently lose.

use warpllm::protocol::anthropic::messages::types::{
    CreateMessageRequest, Message, MessageStreamEvent,
};

use crate::assert_fixtures_round_trip;

mod losslessness;

fn fixtures(shape: &str) -> String {
    format!(
        "{}/tests/protocol/anthropic/messages/fixtures/{shape}",
        env!("CARGO_MANIFEST_DIR")
    )
}

#[test]
fn message_fixtures_round_trip_losslessly() {
    assert_fixtures_round_trip::<Message>(&fixtures("message"));
}

/// These also pin the two spots where a `null` is meaningful and an absence is
/// not the same thing: `stop_reason` on a reply still being generated, and the
/// cache counts, which arrive absent, null, or numeric depending on the reply.
#[test]
fn stream_fixtures_round_trip_losslessly() {
    assert_fixtures_round_trip::<MessageStreamEvent>(&fixtures("stream"));
}

/// The outbound direction, which nothing else checks: a parameter warpllm does
/// not model must still reach Anthropic, and the string and block forms of
/// `system` and `content` must each go out as the form they came in.
#[test]
fn request_fixtures_round_trip_losslessly() {
    assert_fixtures_round_trip::<CreateMessageRequest>(&fixtures("request"));
}
