//! What a caller actually does with a stream: iterate the events and add them
//! up. The shapes can be field-perfect and still be unusable, so this walks
//! recorded transcripts the way a consumer would and checks the total.
//!
//! Two claims per transcript, and they fail for different reasons:
//! - every event survives a deserialize → reserialize round trip, which is the
//!   permissiveness claim, one event at a time;
//! - the events add up to the [`Message`] the same request would have returned
//!   unstreamed, which is the claim that the delta shapes carry enough to be
//!   reassembled at all.
//!
//! **This is the contract the streaming half of this protocol actually makes.**
//! Event-level losslessness is NOT claimed and cannot be: `content_block_stop`
//! and `ping` carry nothing a gateway chunk holds, so no renderer can put them
//! back where they were. Reassembly-equivalence is the promise a consumer
//! depends on, and it is the one pinned here.
//!
//! `reassemble` below is a consumer's-eye reference rather than warpllm's own
//! ingest, matching its openai_compat counterpart: it reads only the fields a
//! client needs and makes no decision about how residue should merge. The
//! gateway's own answer to that lives in `gateway/anthropic/.../stream.rs` and
//! is tested there.
//!
//! Three things this protocol does that the openai_compat transcripts do not
//! exercise, which is why these fixtures exist rather than being adapted:
//! - **there is no `[DONE]` sentinel** — a stream ends with a `message_stop`
//!   EVENT, so completion can only be recognized by decoding;
//! - **the index counts ALL blocks**, so a tool call sitting after text is at
//!   index 1 and a chat-completions reader would have called it 0;
//! - **a signature arrives as its own delta**, just before the block closes,
//!   and covers the whole thinking block rather than any fragment of it.
//!
//! PROVENANCE: hand-built from
//! <https://platform.claude.com/docs/en/build-with-claude/streaming>, with ids
//! and token counts kept as the docs print them. They prove warpllm reads what
//! Anthropic DOCUMENTS; a live capture from `tests/live_stream.rs` is what
//! would prove it reads what Anthropic sends, and drops in unchanged.

use std::collections::BTreeMap;

use warpllm::protocol::anthropic::messages::types::{
    ContentBlock, ContentBlockDelta, Message, MessageStreamEvent, TextBlock, ThinkingBlock,
    ToolUseBlock, Usage,
};

fn fixture(name: &str) -> String {
    std::fs::read_to_string(format!(
        "{}/tests/protocol/anthropic/messages/fixtures/transcript/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// The `data:` payloads of an SSE body.
///
/// The `event:` lines are deliberately ignored: Anthropic duplicates the name
/// in the payload's own `type`, so the framing carries nothing the body does
/// not. That is what lets the shared `SseFrames` reader be reused unchanged.
fn payloads(sse: &str) -> Vec<&str> {
    sse.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect()
}

fn events(sse: &str) -> Vec<MessageStreamEvent> {
    payloads(sse)
        .into_iter()
        .map(|payload| {
            serde_json::from_str(payload).unwrap_or_else(|e| panic!("{payload} failed: {e}"))
        })
        .collect()
}

/// A block being built, keyed by the index its events carry.
#[derive(Default)]
struct Accumulated {
    /// The block as it opened, which is what says whether the fragments below
    /// are text, arguments, or thinking.
    opened: Option<ContentBlock>,
    text: String,
    signature: Option<String>,
}

/// Events in, the message they add up to out.
fn reassemble(events: &[MessageStreamEvent]) -> Message {
    let mut message = match events.first() {
        Some(MessageStreamEvent::MessageStart(start)) => start.message.clone(),
        other => panic!("a transcript opens with message_start, not {other:?}"),
    };
    let mut blocks: BTreeMap<u32, Accumulated> = BTreeMap::new();

    for event in events {
        match event {
            MessageStreamEvent::ContentBlockStart(start) => {
                blocks.entry(start.index).or_default().opened = Some(start.content_block.clone());
            }
            MessageStreamEvent::ContentBlockDelta(event) => {
                let entry = blocks.entry(event.index).or_default();
                match &event.delta {
                    ContentBlockDelta::TextDelta(delta) => entry.text.push_str(&delta.text),
                    ContentBlockDelta::ThinkingDelta(delta) => entry.text.push_str(&delta.thinking),
                    ContentBlockDelta::InputJsonDelta(delta) => {
                        entry.text.push_str(&delta.partial_json)
                    }
                    // Covers the whole block, so it replaces rather than
                    // appends — and it is why a thinking block cannot be
                    // rebuilt from its fragments alone.
                    ContentBlockDelta::SignatureDelta(delta) => {
                        entry.signature = Some(delta.signature.clone())
                    }
                    ContentBlockDelta::Unknown(_) => {}
                }
            }
            MessageStreamEvent::MessageDelta(event) => {
                // `Option<Option<_>>`: the outer says the key was there, the
                // inner says whether it was null. Only a stated key overwrites.
                if let Some(reason) = &event.delta.stop_reason {
                    message.stop_reason = reason.clone();
                }
                if let Some(sequence) = &event.delta.stop_sequence {
                    message.stop_sequence = sequence.clone();
                }
                // Cumulative, not per-event: the last one stated is the total.
                message.usage.output_tokens = event.usage.output_tokens;
            }
            _ => {}
        }
    }

    // BTreeMap, so the blocks come back in index order however the events
    // interleaved — which is the whole job the index does.
    message.content = blocks.into_values().map(close).collect();
    message
}

/// One accumulated block as the block it opened as.
fn close(block: Accumulated) -> ContentBlock {
    let Accumulated {
        opened,
        text,
        signature,
    } = block;
    match opened.expect("a block sends its fragments only after it opens") {
        ContentBlock::Text(open) => ContentBlock::Text(TextBlock { text, ..open }),
        ContentBlock::Thinking(open) => ContentBlock::Thinking(ThinkingBlock {
            thinking: text,
            signature,
            ..open
        }),
        ContentBlock::ToolUse(open) => ContentBlock::ToolUse(ToolUseBlock {
            // The fragments concatenate to JSON; no single one is valid on its
            // own, which is exactly why a consumer must accumulate rather than
            // parse per event.
            input: serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("tool arguments did not add up to JSON: {text:?}: {e}")),
            ..open
        }),
        // Nothing else splits into fragments; it arrived whole on its start.
        whole => whole,
    }
}

fn transcripts() -> Vec<String> {
    let dir = format!(
        "{}/tests/protocol/anthropic/messages/fixtures/transcript",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{dir}: {e}"))
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".sse"))
        .collect();
    names.sort();
    assert!(!names.is_empty(), "no transcripts in {dir}");
    names
}

/// The permissiveness claim, event by event: anything Anthropic sends comes
/// back out as itself. A field dropped, renamed, or mistyped fails the diff.
#[test]
fn every_streamed_event_round_trips_losslessly() {
    for name in transcripts() {
        let sse = fixture(&name);
        for payload in payloads(&sse) {
            let original: serde_json::Value = serde_json::from_str(payload).unwrap();
            let parsed: MessageStreamEvent = serde_json::from_str(payload).unwrap();
            assert_eq!(
                serde_json::to_value(&parsed).unwrap(),
                original,
                "{name}: {payload}"
            );
        }
    }
}

/// The claim this protocol's streaming half actually makes: the events add up
/// to the reply the unstreamed call would have returned.
#[test]
fn a_transcript_adds_up_to_its_message() {
    for name in transcripts() {
        let golden = name.replace(".sse", ".message.json");
        let expected: serde_json::Value = serde_json::from_str(&fixture(&golden)).unwrap();
        let reassembled = reassemble(&events(&fixture(&name)));
        assert_eq!(
            serde_json::to_value(&reassembled).unwrap(),
            expected,
            "{name} did not add up to {golden}"
        );
    }
}

/// The index divergence, pinned rather than described. Anthropic counts ALL
/// blocks, so the tool call in this transcript is at index 1 — a
/// chat-completions reader would have called the first call 0, and a gateway
/// that remapped would have to carry per-index state to do it.
#[test]
fn the_index_counts_every_block_not_just_tool_calls() {
    let events = events(&fixture("doc-tool-call.sse"));
    let call = events
        .iter()
        .find_map(|event| match event {
            MessageStreamEvent::ContentBlockStart(start) => match &start.content_block {
                ContentBlock::ToolUse(_) => Some(start.index),
                _ => None,
            },
            _ => None,
        })
        .expect("the transcript opens a tool call");
    assert_eq!(call, 1, "text is at 0, so the first tool call is at 1");
}

/// A stream ends with an EVENT, not a sentinel. Nothing in the body says
/// "done" until it is decoded, which is why the reader cannot key completion on
/// the framing the way an OpenAI-compatible one does.
#[test]
fn a_transcript_ends_with_message_stop_and_carries_no_sentinel() {
    for name in transcripts() {
        let sse = fixture(&name);
        assert!(
            !sse.contains("[DONE]"),
            "{name} carries a sentinel this protocol does not send"
        );
        assert!(
            matches!(
                events(&sse).last(),
                Some(MessageStreamEvent::MessageStop(_))
            ),
            "{name} does not end with message_stop"
        );
    }
}

/// A thinking block's signature is not a fragment of anything: it arrives once,
/// covers the whole block, and is what Anthropic verifies when the block is
/// sent back. Losing it silently is the failure that matters, so the golden
/// carries it and this names it.
#[test]
fn a_thinking_block_reassembles_with_its_signature() {
    let message = reassemble(&events(&fixture("doc-thinking.sse")));
    let thinking = message
        .content
        .iter()
        .find_map(|block| match block {
            ContentBlock::Thinking(block) => Some(block),
            _ => None,
        })
        .expect("the transcript opens a thinking block");
    assert_eq!(thinking.thinking, "Let me work through this.");
    assert!(
        thinking.signature.as_deref().is_some_and(|s| !s.is_empty()),
        "the signature was dropped in reassembly"
    );
}

/// Usage is cumulative and split across two events: `message_start` states the
/// input, `message_delta` restates the output as a running total. A reader that
/// summed the output counts would over-report.
#[test]
fn usage_comes_from_both_ends_of_the_stream() {
    let message = reassemble(&events(&fixture("doc-text.sse")));
    let Usage {
        input_tokens,
        output_tokens,
        ..
    } = message.usage;
    assert_eq!(input_tokens, 25, "the input count came from message_start");
    assert_eq!(
        output_tokens, 15,
        "the output count is message_delta's total, not a sum of the events"
    );
}
