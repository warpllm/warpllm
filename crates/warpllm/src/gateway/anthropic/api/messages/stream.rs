//! Stream conversions: Anthropic's named events → gateway chunks (ingest) and
//! back (render).
//!
//! # The contract is reassembly-equivalence, not event-level losslessness
//!
//! Everywhere else in this crate a round trip is byte-exact. Here it cannot be,
//! and saying so is the honest version of the claim: `content_block_stop` and
//! `ping` carry no gateway content, so they ingest to `None` and no render can
//! put them back where they were. What IS promised is that a transcript
//! ingested and reassembled equals the [`Message`] the unstreamed call would
//! have returned — which is the contract `reassembly.rs` pins, and the one a
//! consumer actually depends on.
//!
//! # Two facts make the mapping tractable
//!
//! **Every delta is self-describing.** `text_delta`, `thinking_delta` and
//! `input_json_delta` each carry their own `type`, so a reader never has to
//! remember what kind of block an index opened. Only `content_block_start`
//! establishes a tool call's `id` and `name`, and those belong on the chunk
//! that start produces. That is why [`StreamState`] holds no per-index map.
//!
//! **`ChatResponseChunk` requires `id` and `model` on every chunk**, and
//! Anthropic sends them once, on `message_start`. That is the whole of the
//! state, and it is unavoidable rather than convenient.
//!
//! # What `render_event` does NOT yet do
//!
//! Rendering a SAME-PROTOCOL chunk is complete: the retained wire event decides
//! the shape and replays it. Rendering a chunk from another protocol is not,
//! and the gap is structural rather than an omission — it is one chunk to one
//! event, where a faithful translation needs to buffer:
//!
//! * chat completions puts a stream's totals on a TRAILING chunk that carries
//!   no choices, so it arrives after the chunk that has the stop reason. This
//!   renders each alone, and the one with no choices renders to nothing;
//! * a fragment carrying a tool call's `id`, `name` AND argument text is two
//!   events here, so only the opener is emitted;
//! * a tool call's index is remapped by neither side — see the note below.
//!
//! None of that is reachable today: nothing renders a foreign chunk to this
//! wire, because doing so needs an Anthropic-shaped INGRESS, which is out of
//! scope. The function is written for the round-trip tests and as the honest
//! starting point for that work, and this list is what that work owes.
//!
//! # A stream's totals are the caller's to ask for
//!
//! Anthropic states them on `message_delta`, unconditionally. Chat completions
//! states them only under `stream_options.include_usage`, and on a TRAILING
//! chunk that carries no choices — and the IR follows chat completions here,
//! because [`ChatResponseChunk::usage`](types::ChatResponseChunk::usage) is
//! documented as present only when the caller asked. So this reads Anthropic's
//! counts into [`StreamState`] and emits them at `message_stop` as the
//! completionless chunk the IR already models, or does not emit them at all.
//!
//! Neither half costs the same-protocol round trip anything. The wire counts
//! are RETAINED on the `message_delta` chunk either way, so a render rebuilds
//! that event from the residue rather than from the promotion; and the
//! trailing chunk carries no residue, so it renders to nothing — which is
//! exactly what `message_stop` is.
//!
//! # The index is Anthropic's, carried verbatim
//!
//! Anthropic's block `index` counts ALL blocks — text at 0, a tool call at 1.
//! Chat completions' `tool_calls[].index` counts only tool calls, so the two are
//! different numbers for the same call. This carries Anthropic's through
//! unchanged: `index` promises to be a stable correlation key and nothing more,
//! which is exactly what consumers use it for, and remapping would need the
//! per-index state the previous section just established is not needed.

use serde_json::Value;

use crate::gateway::anthropic::{merged_ext, namespaced, role_from_wire};
use crate::gateway::types::{self, ContentDelta};
use crate::protocol::UnknownFields;
use crate::protocol::anthropic::messages::types::{
    ContentBlock as WireBlock, ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStartEvent,
    InputJsonDelta, Message, MessageDelta as WireMessageDelta, MessageDeltaEvent,
    MessageDeltaUsage, MessageStartEvent, MessageStreamEvent, TextDelta, ThinkingDelta,
    ToolUseBlock,
};
use crate::types::Protocol;

use super::response::{finish_reason, plain, take_nullable_string, take_typed};

/// What a reader must remember across events, and nothing more.
///
/// `id` and `model` because every gateway chunk requires them and Anthropic
/// sends them once. The input counts because `message_delta` reports only the
/// cumulative OUTPUT tokens, so the usage on the chunk that carries totals has
/// to be completed from what `message_start` said. The totals themselves
/// because they are stated one event before the chunk that carries them.
#[derive(Debug, Default)]
pub(crate) struct StreamState {
    id: String,
    model: String,
    input_tokens: Option<u32>,
    cache_read_tokens: Option<u32>,
    cache_write_tokens: Option<u32>,
    /// Whether the caller asked for the stream's totals.
    ///
    /// Anthropic reports them whether or not anyone did, so without this the
    /// counts reach a caller who never requested them, on a protocol whose
    /// contract is that the field is absent. It gates the PROMOTION only —
    /// the wire value is retained regardless.
    report_usage: bool,
    /// The totals, held from the `message_delta` that states them to the
    /// `message_stop` that carries them out. Always `None` when
    /// [`Self::report_usage`] is false, and again once they have been emitted.
    totals: Option<types::Usage>,
}

impl StreamState {
    /// The state one stream starts from.
    ///
    /// `report_usage` is the caller's
    /// [`stream_include_usage`](types::ChatRequest::stream_include_usage);
    /// everything else is learned from `message_start`.
    pub(crate) fn new(report_usage: bool) -> Self {
        Self {
            report_usage,
            ..Self::default()
        }
    }
}

/// One wire event as a gateway chunk, or `None` where the event carries no
/// gateway content.
///
/// `None` is not a failure and not a dropped value: `content_block_stop` and
/// `ping` say nothing a chunk can hold, and inventing an empty chunk for them
/// would put a value in front of a consumer that the provider never sent.
///
/// An `error` event is deliberately NOT handled here. It ends the stream and
/// becomes an [`Error`](crate::Error) on the stream itself — see
/// `exchange::ChatChunkStream` — because a failure is not a chunk, and giving
/// it one would make a consumer that ignores content miss the failure entirely.
pub(crate) fn ingest_event(
    event: MessageStreamEvent,
    state: &mut StreamState,
) -> Option<types::ChatResponseChunk> {
    match event {
        MessageStreamEvent::MessageStart(start) => Some(ingest_message_start(start, state)),
        MessageStreamEvent::ContentBlockStart(start) => Some(ingest_block_start(start, state)),
        MessageStreamEvent::ContentBlockDelta(delta) => Some(ingest_block_delta(delta, state)),
        MessageStreamEvent::MessageDelta(delta) => Some(ingest_message_delta(delta, state)),
        // No gateway content. See this function's doc.
        MessageStreamEvent::ContentBlockStop(_) => None,
        // Content only when the caller asked for the totals, which is the one
        // thing this protocol says at the end and chat completions says here.
        MessageStreamEvent::MessageStop(_) => ingest_message_stop(state),
        MessageStreamEvent::Ping(_) => None,
        // Ends the stream as an error, never as a chunk.
        MessageStreamEvent::Error(_) => None,
        // A named event this gateway does not model. It has no content to lift
        // and no shape to guess at, so it rides the envelope's own namespace
        // where a same-protocol render replays it verbatim.
        MessageStreamEvent::Unknown(value) => Some(unknown_chunk(value, state)),
    }
}

/// `message_start` — the only event that establishes identity.
fn ingest_message_start(
    start: MessageStartEvent,
    state: &mut StreamState,
) -> types::ChatResponseChunk {
    let MessageStartEvent {
        message,
        unknown_fields,
    } = start;
    // PROMOTE BUT RETAIN, and here the retained copy is the WHOLE wire message
    // rather than the leftovers. `id`, `model` and `role` are promoted onto the
    // chunk and appear in both places, which is the point: this event's payload
    // is a `Message`, every field of it is required to rebuild one, and a
    // render that reassembled it from the chunk's promoted values plus partial
    // residue would have to guess at `type`, at an empty-versus-absent
    // `content`, and at the three-state nullability of `stop_reason`.
    //
    // It also means a field added to `Message` round trips without a change
    // here, which is why this destructure binds what it promotes and lets the
    // retained copy answer for the rest.
    let retained = plain(&message);
    let Message {
        id,
        role,
        model,
        usage,
        // Carried by `retained` above, and read back from it verbatim.
        r#type: _,
        content: _,
        stop_reason: _,
        stop_sequence: _,
        unknown_fields: _,
    } = message;

    state.id = id.clone();
    state.model = model.clone();
    state.input_tokens = Some(usage.input_tokens);
    state.cache_read_tokens = usage.cache_read_input_tokens.flatten();
    state.cache_write_tokens = usage.cache_creation_input_tokens.flatten();

    let (gateway_role, raw_role) = role_from_wire(role);
    let mut delta_ext = UnknownFields::new();
    if let Some(raw) = raw_role {
        delta_ext.insert("role".into(), Value::String(raw));
    }

    let mut chunk_ext = UnknownFields::new();
    chunk_ext.insert("message".into(), retained);
    chunk_ext.extend(unknown_fields);

    types::ChatResponseChunk {
        id,
        model,
        // Anthropic has no creation timestamp, streamed or not.
        created: None,
        completions: vec![types::CompletionDelta {
            delta: types::MessageDelta {
                role: Some(gateway_role),
                content: Vec::new(),
                ext: namespaced(delta_ext),
            },
            finish_reason: None,
            finish_reason_raw: None,
            ext: types::ProviderExt::new(),
        }],
        // The counts are real but PARTIAL — output is still zero and the reply
        // has not happened. Reporting them here would have a consumer that adds
        // up every chunk's usage double-count the input against `message_delta`,
        // which reports the same totals again at the end.
        usage: None,
        ext: namespaced(chunk_ext),
        source: None,
    }
}

/// `content_block_start` — the only event that names a tool call.
fn ingest_block_start(
    start: ContentBlockStartEvent,
    state: &StreamState,
) -> types::ChatResponseChunk {
    let ContentBlockStartEvent {
        index,
        content_block,
        unknown_fields,
    } = start;

    let delta = match &content_block {
        // The empty opener is a REAL value, matching the `"content": ""` an
        // OpenAI stream opens with. A consumer concatenating fragments sees the
        // same sequence either way.
        WireBlock::Text(text) => ContentDelta::Text {
            text: text.text.clone(),
        },
        WireBlock::ToolUse(call) => ContentDelta::ToolCall {
            index,
            id: Some(call.id.clone()),
            kind: Some("tool_use".into()),
            name: Some(call.name.clone()),
            // Arguments arrive as `input_json_delta` fragments. `None` rather
            // than `Some("")`: this fragment carried no argument text at all,
            // which the IR distinguishes from one that carried the empty
            // string.
            arguments: None,
        },
        WireBlock::Thinking(thinking) => ContentDelta::Reasoning {
            text: thinking.thinking.clone(),
            provenance: Some(Protocol::Anthropic.as_str().to_string()),
        },
        // No `ContentDelta` holds encrypted reasoning — the completed form's
        // `ReasoningDetail::Encrypted` has no streaming counterpart, because a
        // signature covers a whole block and cannot ride its fragments. It
        // passes through whole, which is what it is: not a fragment at all.
        //
        // Every other block type reaches here too. A block warpllm does not
        // model is exactly the case `Unknown` exists for.
        other => ContentDelta::Unknown(plain(other)),
    };

    let mut block_ext = UnknownFields::new();
    block_ext.insert("index".into(), Value::from(index));
    // Retained whole rather than field by field: this is what a same-protocol
    // render rebuilds the event from, and the promoted delta above is lossy for
    // every block whose residue has no gateway field.
    block_ext.insert("content_block".into(), plain(&content_block));
    block_ext.extend(unknown_fields);

    chunk(state, delta, block_ext)
}

/// `content_block_delta` — the fragments themselves.
fn ingest_block_delta(
    event: ContentBlockDeltaEvent,
    state: &StreamState,
) -> types::ChatResponseChunk {
    let ContentBlockDeltaEvent {
        index,
        delta,
        unknown_fields,
    } = event;

    let content = match &delta {
        ContentBlockDelta::TextDelta(text) => ContentDelta::Text {
            text: text.text.clone(),
        },
        ContentBlockDelta::ThinkingDelta(thinking) => ContentDelta::Reasoning {
            text: thinking.thinking.clone(),
            provenance: Some(Protocol::Anthropic.as_str().to_string()),
        },
        ContentBlockDelta::InputJsonDelta(json) => ContentDelta::ToolCall {
            index,
            // Established by `content_block_start`; a fragment repeats none of
            // it.
            id: None,
            kind: None,
            name: None,
            arguments: Some(json.partial_json.clone()),
        },
        // No `ContentDelta` holds a signature — the IR says so explicitly, and
        // the reason is the same one that keeps it off `ContentDelta::Reasoning`:
        // it covers a whole block, so it is not a fragment of anything. It
        // survives as residue, which is what makes a run of thinking blocks
        // reassemble with its signature intact.
        ContentBlockDelta::SignatureDelta(_) | ContentBlockDelta::Unknown(_) => {
            ContentDelta::Unknown(plain(&delta))
        }
    };

    let mut block_ext = UnknownFields::new();
    block_ext.insert("index".into(), Value::from(index));
    block_ext.insert("delta".into(), plain(&delta));
    block_ext.extend(unknown_fields);

    chunk(state, content, block_ext)
}

/// `message_delta` — the stop reason and the totals.
fn ingest_message_delta(
    event: MessageDeltaEvent,
    state: &mut StreamState,
) -> types::ChatResponseChunk {
    let MessageDeltaEvent {
        delta,
        usage,
        unknown_fields,
    } = event;
    let WireMessageDelta {
        stop_reason,
        stop_sequence,
        unknown_fields: delta_unknowns,
    } = delta;

    let raw = stop_reason.clone().flatten();
    let mut completion_ext = UnknownFields::new();
    // Both fields are three-state, and a flatten destroys the distinction the
    // third state IS. `stop_reason`'s inner string is promoted to
    // `finish_reason_raw`, so only the present-but-NULL case needs a marker
    // here; an absent key must leave no residue at all, or it renders back as
    // an explicit null. `stop_sequence` has no gateway home, so its outer
    // option is retained exactly when the key was there.
    if matches!(stop_reason, Some(None)) {
        completion_ext.insert("stop_reason".into(), Value::Null);
    }
    if let Some(sequence) = &stop_sequence {
        completion_ext.insert(
            "stop_sequence".into(),
            sequence.clone().map_or(Value::Null, Value::String),
        );
    }
    completion_ext.extend(delta_unknowns);

    let mut chunk_ext = UnknownFields::new();
    chunk_ext.insert("usage".into(), plain(&usage));
    chunk_ext.extend(unknown_fields);

    // Held rather than reported here: the totals belong on a chunk of their
    // own, and this protocol's stream ends one event later. Skipped entirely
    // when nobody asked — the retained copy above is what a same-protocol
    // render reads either way, so this loses nothing.
    if state.report_usage {
        state.totals = Some(delta_usage(&usage, state));
    }

    types::ChatResponseChunk {
        id: state.id.clone(),
        model: state.model.clone(),
        created: None,
        completions: vec![types::CompletionDelta {
            delta: types::MessageDelta::default(),
            finish_reason: raw.as_deref().map(|raw| finish_reason(Some(raw))),
            finish_reason_raw: raw,
            ext: namespaced(completion_ext),
        }],
        usage: None,
        ext: namespaced(chunk_ext),
        source: None,
    }
}

/// `message_stop` — the totals, on a chunk that carries nothing else.
///
/// `None` unless the caller asked, and `None` a second time if it somehow
/// arrives twice: the totals are taken, not copied, so a stream cannot report
/// them more than once.
///
/// The chunk holds no residue, which is deliberate rather than an omission —
/// `message_stop` has no fields to retain, and a render that found none is
/// right to produce no event.
fn ingest_message_stop(state: &mut StreamState) -> Option<types::ChatResponseChunk> {
    let usage = state.totals.take()?;
    Some(types::ChatResponseChunk {
        id: state.id.clone(),
        model: state.model.clone(),
        created: None,
        // The usage-only shape the IR already models, and the one chat
        // completions puts a stream's totals on.
        completions: Vec::new(),
        usage: Some(usage),
        ext: types::ProviderExt::new(),
        source: None,
    })
}

/// An event whose `type` this gateway does not recognize.
fn unknown_chunk(value: Value, state: &StreamState) -> types::ChatResponseChunk {
    let mut chunk_ext = UnknownFields::new();
    chunk_ext.insert("event".into(), value);
    types::ChatResponseChunk {
        id: state.id.clone(),
        model: state.model.clone(),
        created: None,
        completions: Vec::new(),
        usage: None,
        ext: namespaced(chunk_ext),
        source: None,
    }
}

/// The shape every content-bearing chunk after `message_start` shares.
fn chunk(
    state: &StreamState,
    content: ContentDelta,
    block_ext: UnknownFields,
) -> types::ChatResponseChunk {
    types::ChatResponseChunk {
        id: state.id.clone(),
        model: state.model.clone(),
        created: None,
        completions: vec![types::CompletionDelta {
            delta: types::MessageDelta {
                // Sent once, on the chunk that opened the choice.
                role: None,
                content: vec![content],
                ext: namespaced(block_ext),
            },
            finish_reason: None,
            finish_reason_raw: None,
            ext: types::ProviderExt::new(),
        }],
        usage: None,
        ext: types::ProviderExt::new(),
        source: None,
    }
}

/// A `message_delta`'s usage, completed from what `message_start` remembered.
///
/// The same normalization `response::ingest_usage` does and for the same
/// reason: Anthropic's `input_tokens` counts only what came AFTER the last
/// cache breakpoint, while the IR's is the whole input. A protocol whose wire
/// value is exclusive has to add the cache counts back on ingest.
///
/// This event may restate the input counts, and when it does they WIN — a
/// restated total is the provider correcting itself, and preferring the
/// remembered one would report a stale number.
fn delta_usage(usage: &MessageDeltaUsage, state: &StreamState) -> types::Usage {
    let input = usage.input_tokens.flatten().or(state.input_tokens);
    let read = usage
        .cache_read_input_tokens
        .flatten()
        .or(state.cache_read_tokens);
    let write = usage
        .cache_creation_input_tokens
        .flatten()
        .or(state.cache_write_tokens);

    let cached = u64::from(read.unwrap_or(0)) + u64::from(write.unwrap_or(0));
    let total_input = input.map(|count| u64::from(count) + cached);
    types::Usage {
        input_tokens: total_input,
        output_tokens: Some(u64::from(usage.output_tokens)),
        total_tokens: total_input.map(|input| input + u64::from(usage.output_tokens)),
        // A streamed reply reports no thinking breakdown; only the unstreamed
        // `usage.output_tokens_details` carries one.
        reasoning_tokens: None,
        cache_read_tokens: read.map(u64::from),
        cache_write_tokens: write.map(u64::from),
        ext: types::ProviderExt::new(),
    }
}

/// The inverse of [`ingest_event`], complete for a chunk this protocol
/// produced and deliberately incomplete for one it did not — see this module's
/// note on what an Anthropic-shaped ingress owes.
///
/// `None` where a chunk maps to no event: the usage-only and keepalive shapes
/// another protocol produces have no spelling here.
///
/// Reads only this protocol's namespace, through [`merged_ext`] — never
/// `ext.get`, which would bypass [`Protocol::may_read`] and let another
/// protocol's retained fields onto this wire.
pub(crate) fn render_event(
    chunk: &types::ChatResponseChunk,
    provider: &str,
) -> Option<MessageStreamEvent> {
    let mut chunk_ext = merged_ext(&chunk.ext, provider);
    if let Some(event) = chunk_ext.remove("event") {
        return Some(MessageStreamEvent::Unknown(event));
    }
    if let Some(message) = take_typed(&mut chunk_ext, "message") {
        return Some(MessageStreamEvent::MessageStart(MessageStartEvent {
            message,
            unknown_fields: chunk_ext,
        }));
    }
    let completion = chunk.completions.first()?;
    if completion.finish_reason_raw.is_some() || chunk_ext.contains_key("usage") {
        return Some(render_message_delta(chunk, completion, chunk_ext, provider));
    }
    render_content_event(completion, provider)
}

fn render_message_delta(
    chunk: &types::ChatResponseChunk,
    completion: &types::CompletionDelta,
    mut chunk_ext: UnknownFields,
    provider: &str,
) -> MessageStreamEvent {
    let mut completion_ext = merged_ext(&completion.ext, provider);
    MessageStreamEvent::MessageDelta(MessageDeltaEvent {
        delta: WireMessageDelta {
            stop_reason: match completion.finish_reason_raw.clone() {
                Some(raw) => Some(Some(raw)),
                // An explicit null and an absent key are different states, and
                // only the residue remembers which this was.
                None => completion_ext.remove("stop_reason").map(|_| None),
            },
            stop_sequence: take_nullable_string(&mut completion_ext, "stop_sequence"),
            unknown_fields: completion_ext,
        },
        // Retained residue first, then the GATEWAY form — which is the same
        // order every renderer here uses, and which was missing: reading only
        // the residue meant a chunk from another protocol reported
        // `output_tokens: 0` however many it had actually spent.
        usage: take_typed(&mut chunk_ext, "usage")
            .or_else(|| chunk.usage.as_ref().map(delta_usage_of))
            .unwrap_or_else(unknown_usage),
        unknown_fields: chunk_ext,
    })
}

/// The gateway's totals as this event's, the inverse of [`delta_usage`].
///
/// Anthropic's `input_tokens` counts only what came AFTER the last cache
/// breakpoint while the gateway's is the whole input, so the cache counts come
/// back OUT here. Saturating rather than wrapping: a gateway form built by hand
/// with a cached count larger than its input is nonsense, and zero is the inert
/// answer rather than four billion.
fn delta_usage_of(usage: &types::Usage) -> MessageDeltaUsage {
    let cached = usage.cache_read_tokens.unwrap_or(0) + usage.cache_write_tokens.unwrap_or(0);
    MessageDeltaUsage {
        output_tokens: usage.output_tokens.unwrap_or(0) as u32,
        input_tokens: usage
            .input_tokens
            .map(|input| Some(input.saturating_sub(cached) as u32)),
        cache_creation_input_tokens: usage.cache_write_tokens.map(|count| Some(count as u32)),
        cache_read_input_tokens: usage.cache_read_tokens.map(|count| Some(count as u32)),
        unknown_fields: UnknownFields::new(),
    }
}

/// The totals for a chunk that stated none.
///
/// `output_tokens` is REQUIRED on this event, so something has to go there and
/// zero is the only inert value. It is a real limitation rather than a good
/// answer: chat completions puts its totals on a trailing usage-only chunk that
/// carries no choices, and one chunk cannot become two events here — see this
/// module's note on what an Anthropic-shaped ingress owes.
fn unknown_usage() -> MessageDeltaUsage {
    MessageDeltaUsage {
        output_tokens: 0,
        input_tokens: None,
        cache_creation_input_tokens: None,
        cache_read_input_tokens: None,
        unknown_fields: UnknownFields::new(),
    }
}

/// A content-bearing chunk as the event it came from.
///
/// The retained `content_block` and `delta` decide which event this was, since
/// a start and a delta both carry one `ContentDelta` and only the residue tells
/// them apart. A chunk from ANOTHER protocol has neither, and renders from the
/// delta alone.
fn render_content_event(
    completion: &types::CompletionDelta,
    provider: &str,
) -> Option<MessageStreamEvent> {
    let mut block_ext = merged_ext(&completion.delta.ext, provider);
    let index = block_ext
        .remove("index")
        .and_then(|value| value.as_u64())
        .map(|index| index as u32);

    if let Some(content_block) = take_typed(&mut block_ext, "content_block") {
        return Some(MessageStreamEvent::ContentBlockStart(
            ContentBlockStartEvent {
                index: index.unwrap_or(0),
                content_block,
                unknown_fields: block_ext,
            },
        ));
    }
    if let Some(delta) = take_typed(&mut block_ext, "delta") {
        return Some(MessageStreamEvent::ContentBlockDelta(
            ContentBlockDeltaEvent {
                index: index.unwrap_or_else(|| delta_index(completion)),
                delta,
                unknown_fields: block_ext,
            },
        ));
    }
    render_foreign_event(
        completion.delta.content.first()?,
        index.unwrap_or_else(|| delta_index(completion)),
        block_ext,
    )
}

/// A content fragment from ANOTHER protocol as the event this one spells it
/// with — which is not always a delta.
///
/// A tool call that carries its `id` and `name` is an OPENER, and this protocol
/// says an opener with `content_block_start`, not with a fragment. Rendering it
/// as an `input_json_delta` dropped both fields, and a chat-completions opener
/// carries no argument text either, so the whole event came back `None` — the
/// one event that names a tool call, rendered as silence.
///
/// A fragment carrying an opener AND argument text is the case this cannot
/// fully express: it is two events here and one chunk there. The start wins,
/// because `id` and `name` are what nothing later can supply while argument
/// text arrives again on the next fragment. Buffering the pair is part of what
/// an Anthropic-shaped ingress owes; see this module's docs.
fn render_foreign_event(
    content: &ContentDelta,
    index: u32,
    unknown_fields: UnknownFields,
) -> Option<MessageStreamEvent> {
    if let ContentDelta::ToolCall {
        id: Some(id),
        name: Some(name),
        ..
    } = content
    {
        return Some(MessageStreamEvent::ContentBlockStart(
            ContentBlockStartEvent {
                index,
                content_block: WireBlock::ToolUse(ToolUseBlock {
                    id: id.clone(),
                    name: name.clone(),
                    // The arguments arrive as fragments after this, exactly as
                    // Anthropic's own opener carries an empty object.
                    input: Value::Object(serde_json::Map::new()),
                    cache_control: None,
                    unknown_fields: UnknownFields::new(),
                }),
                unknown_fields,
            },
        ));
    }
    Some(MessageStreamEvent::ContentBlockDelta(
        ContentBlockDeltaEvent {
            index,
            delta: render_delta(content)?,
            unknown_fields,
        },
    ))
}

/// A gateway content delta as this protocol's, for a chunk that arrived with no
/// retained wire delta — i.e. one from another protocol.
fn render_delta(content: &ContentDelta) -> Option<ContentBlockDelta> {
    Some(match content {
        ContentDelta::Text { text } => ContentBlockDelta::TextDelta(TextDelta {
            text: text.clone(),
            unknown_fields: UnknownFields::new(),
        }),
        ContentDelta::Reasoning { text, .. } => ContentBlockDelta::ThinkingDelta(ThinkingDelta {
            thinking: text.clone(),
            unknown_fields: UnknownFields::new(),
        }),
        ContentDelta::ToolCall { arguments, .. } => {
            ContentBlockDelta::InputJsonDelta(InputJsonDelta {
                partial_json: arguments.clone()?,
                unknown_fields: UnknownFields::new(),
            })
        }
        ContentDelta::Unknown(value) => ContentBlockDelta::Unknown(value.clone()),
    })
}

/// The index a foreign chunk's delta carries, when there is no retained one.
fn delta_index(completion: &types::CompletionDelta) -> u32 {
    match completion.delta.content.first() {
        Some(ContentDelta::ToolCall { index, .. }) => *index,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::gateway::types::FinishReason;

    fn event(body: Value) -> MessageStreamEvent {
        serde_json::from_value(body).expect("fixture is a valid event")
    }

    /// A stream opened by a caller who did NOT ask for the totals, which is
    /// chat completions' default and the case most of these tests are about.
    fn started() -> (StreamState, types::ChatResponseChunk) {
        start(false)
    }

    /// A stream opened by a caller who DID ask.
    fn started_asking() -> (StreamState, types::ChatResponseChunk) {
        start(true)
    }

    fn start(report_usage: bool) -> (StreamState, types::ChatResponseChunk) {
        let mut state = StreamState::new(report_usage);
        let chunk = ingest_event(
            event(json!({
                "type": "message_start",
                "message": {
                    "id": "msg_1", "type": "message", "role": "assistant",
                    "model": "claude-opus-5", "content": [],
                    "stop_reason": null, "stop_sequence": null,
                    "usage": {"input_tokens": 21, "output_tokens": 0,
                              "cache_read_input_tokens": 8192,
                              "cache_creation_input_tokens": 1024}
                }
            })),
            &mut state,
        )
        .expect("message_start is a chunk");
        (state, chunk)
    }

    fn ingest(body: Value, state: &mut StreamState) -> Option<types::ChatResponseChunk> {
        ingest_event(event(body), state)
    }

    /// Identity arrives once and every later chunk carries it, which is the
    /// whole reason [`StreamState`] exists.
    #[test]
    fn message_start_establishes_identity_for_every_later_chunk() {
        let (mut state, first) = started();
        assert_eq!(first.id, "msg_1");
        assert_eq!(first.model, "claude-opus-5");
        assert_eq!(
            first.completions[0].delta.role,
            Some(types::Role::Assistant)
        );

        let later = ingest(
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "text_delta", "text": "hi"}}),
            &mut state,
        )
        .unwrap();
        assert_eq!(later.id, "msg_1");
        assert_eq!(later.model, "claude-opus-5");
    }

    /// The events that say nothing a chunk can hold produce nothing. An empty
    /// chunk would be a value the provider never sent.
    #[test]
    fn the_contentless_events_produce_no_chunk() {
        let (mut state, _) = started();
        for body in [
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "message_stop"}),
            json!({"type": "ping"}),
        ] {
            assert!(ingest(body.clone(), &mut state).is_none(), "{body}");
        }
    }

    /// An `error` event is not a chunk. It ends the stream as an error, and
    /// producing a chunk for it would let a consumer that reads only content
    /// miss the failure entirely.
    #[test]
    fn an_error_event_is_not_a_chunk() {
        let (mut state, _) = started();
        assert!(
            ingest(
                json!({"type": "error",
                       "error": {"type": "overloaded_error", "message": "overloaded"}}),
                &mut state
            )
            .is_none()
        );
    }

    /// The full event table, one row at a time, so a change to any single arm
    /// fails on that arm rather than inside a transcript.
    #[test]
    fn the_content_event_table() {
        let (mut state, _) = started();

        let text = ingest(
            json!({"type": "content_block_start", "index": 0,
                   "content_block": {"type": "text", "text": ""}}),
            &mut state,
        )
        .unwrap();
        assert!(matches!(
            &text.completions[0].delta.content[0],
            ContentDelta::Text { text } if text.is_empty()
        ));

        let call = ingest(
            json!({"type": "content_block_start", "index": 1,
                   "content_block": {"type": "tool_use", "id": "tu_1",
                                     "name": "weather", "input": {}}}),
            &mut state,
        )
        .unwrap();
        assert!(matches!(
            &call.completions[0].delta.content[0],
            ContentDelta::ToolCall { index: 1, id: Some(id), name: Some(name), arguments: None, .. }
                if id == "tu_1" && name == "weather"
        ));

        let fragment = ingest(
            json!({"type": "content_block_delta", "index": 1,
                   "delta": {"type": "input_json_delta", "partial_json": "{\"city\":"}}),
            &mut state,
        )
        .unwrap();
        assert!(matches!(
            &fragment.completions[0].delta.content[0],
            ContentDelta::ToolCall { index: 1, id: None, name: None, arguments: Some(text), .. }
                if text == "{\"city\":"
        ));

        let thinking = ingest(
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "thinking_delta", "thinking": "step"}}),
            &mut state,
        )
        .unwrap();
        assert!(matches!(
            &thinking.completions[0].delta.content[0],
            ContentDelta::Reasoning { text, provenance: Some(p) }
                if text == "step" && p == "anthropic"
        ));

        // A signature covers a whole block, so it is not a fragment of one and
        // no `ContentDelta` holds it. It rides residue instead of being lost.
        let signature = ingest(
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "signature_delta", "signature": "sig"}}),
            &mut state,
        )
        .unwrap();
        assert!(matches!(
            &signature.completions[0].delta.content[0],
            ContentDelta::Unknown(_)
        ));
    }

    /// Anthropic's index counts ALL blocks and chat completions' counts only
    /// tool calls. Carrying Anthropic's verbatim is a documented divergence,
    /// so it gets a test rather than a comment alone.
    #[test]
    fn the_block_index_is_anthropics_not_remapped() {
        let (mut state, _) = started();
        // Text at 0, so the FIRST tool call is at 1 — a chat-completions
        // reader would have called it 0.
        let call = ingest(
            json!({"type": "content_block_start", "index": 1,
                   "content_block": {"type": "tool_use", "id": "tu_1",
                                     "name": "w", "input": {}}}),
            &mut state,
        )
        .unwrap();
        assert!(matches!(
            call.completions[0].delta.content[0],
            ContentDelta::ToolCall { index: 1, .. }
        ));
    }

    /// `message_start`'s counts are partial — output has not happened — so
    /// reporting usage there would double-count against `message_delta`, which
    /// restates the totals at the end.
    #[test]
    fn usage_is_reported_once_and_includes_the_cached_input() {
        let (mut state, first) = started_asking();
        assert!(first.usage.is_none(), "partial totals were reported early");

        assert!(
            ingest(
                json!({"type": "message_delta",
                       "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                       "usage": {"output_tokens": 16}}),
                &mut state,
            )
            .unwrap()
            .usage
            .is_none(),
            "the totals rode the chunk that carries the stop reason"
        );

        let usage = ingest(json!({"type": "message_stop"}), &mut state)
            .expect("a caller who asked gets a chunk for the totals")
            .usage
            .unwrap();
        // 21 uncached + 8192 read + 1024 written, per the IR's rule that the
        // cached counts are a BREAKDOWN of the input, never addends to it.
        assert_eq!(usage.input_tokens, Some(9237));
        assert_eq!(usage.output_tokens, Some(16));
        assert_eq!(usage.total_tokens, Some(9253));
        assert_eq!(usage.cache_read_tokens, Some(8192));
        assert_eq!(usage.cache_write_tokens, Some(1024));
    }

    /// A restated input count is the provider correcting itself; the remembered
    /// one is stale the moment it does.
    #[test]
    fn a_restated_input_count_wins_over_the_remembered_one() {
        let (mut state, _) = started_asking();
        ingest(
            json!({"type": "message_delta",
                   "delta": {"stop_reason": "end_turn"},
                   "usage": {"output_tokens": 16, "input_tokens": 30,
                             "cache_read_input_tokens": 0,
                             "cache_creation_input_tokens": 0}}),
            &mut state,
        )
        .unwrap();
        let end = ingest(json!({"type": "message_stop"}), &mut state).unwrap();
        assert_eq!(end.usage.unwrap().input_tokens, Some(30));
    }

    /// Anthropic states a stream's totals whether or not anyone asked. The IR
    /// says the field is present only when someone did, so this is the gate
    /// that keeps an OpenAI caller from receiving a `usage` it never requested
    /// on a protocol whose contract is that it is null.
    #[test]
    fn the_totals_stay_off_the_stream_when_nobody_asked() {
        let (mut state, _) = started();
        let end = ingest(
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                   "usage": {"output_tokens": 16}}),
            &mut state,
        )
        .unwrap();
        assert!(end.usage.is_none(), "unasked-for totals reached the caller");
        assert!(
            ingest(json!({"type": "message_stop"}), &mut state).is_none(),
            "a chunk was invented for totals nobody asked for"
        );
    }

    /// The totals arrive on a chunk of their own with no choices, which is
    /// where chat completions puts them — not beside the stop reason, which is
    /// where this protocol does.
    #[test]
    fn the_totals_ride_a_chunk_that_carries_nothing_else() {
        let (mut state, _) = started_asking();
        ingest(
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                   "usage": {"output_tokens": 16}}),
            &mut state,
        )
        .unwrap();
        let totals = ingest(json!({"type": "message_stop"}), &mut state).unwrap();
        assert!(totals.completions.is_empty());
        assert!(totals.usage.is_some());
        assert_eq!(totals.id, "msg_1");
        assert_eq!(totals.model, "claude-opus-5");
        // Taken, not copied: a stream reports its totals once however many
        // times the event that carries them arrives.
        assert!(ingest(json!({"type": "message_stop"}), &mut state).is_none());
    }

    /// The gate costs the same-protocol round trip nothing, because the wire
    /// counts are retained rather than promoted. Written from the state a
    /// caller who asked for NOTHING produces, so the promotion cannot be what
    /// makes it pass.
    #[test]
    fn a_message_delta_replays_its_usage_even_when_nobody_asked() {
        let (mut state, _) = started();
        let end = ingest(
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                   "usage": {"output_tokens": 16, "input_tokens": 30}}),
            &mut state,
        )
        .unwrap();
        assert!(end.usage.is_none());
        let MessageStreamEvent::MessageDelta(rendered) =
            render_event(&end, "anthropic").expect("a stop reason is an event")
        else {
            panic!("a message_delta chunk rendered as something else");
        };
        assert_eq!(rendered.usage.output_tokens, 16);
        assert_eq!(rendered.usage.input_tokens, Some(Some(30)));
    }

    /// And the chunk the totals ride renders to nothing, because `message_stop`
    /// says nothing this wire holds. Without it the stream would carry a second
    /// `message_delta` reporting the same counts twice.
    #[test]
    fn the_chunk_the_totals_ride_is_not_an_event() {
        let (mut state, _) = started_asking();
        ingest(
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                   "usage": {"output_tokens": 16}}),
            &mut state,
        )
        .unwrap();
        let totals = ingest(json!({"type": "message_stop"}), &mut state).unwrap();
        assert!(render_event(&totals, "anthropic").is_none());
    }

    /// The stop reason crosses through the same table the unstreamed reply
    /// uses — one vocabulary, so the streamed and unstreamed forms of a request
    /// cannot end differently.
    #[test]
    fn the_stop_reason_uses_the_unstreamed_table() {
        let (mut state, _) = started();
        for (raw, expected) in [
            ("end_turn", FinishReason::Stop),
            ("max_tokens", FinishReason::Length),
            ("tool_use", FinishReason::ToolCalls),
            ("refusal", FinishReason::ContentFilter),
            ("pause_turn", FinishReason::Other),
        ] {
            let end = ingest(
                json!({"type": "message_delta", "delta": {"stop_reason": raw},
                       "usage": {"output_tokens": 1}}),
                &mut state,
            )
            .unwrap();
            assert_eq!(end.completions[0].finish_reason, Some(expected), "{raw}");
            assert_eq!(end.completions[0].finish_reason_raw.as_deref(), Some(raw));
        }
    }

    /// The three-state fields on a `message_delta`, every combination.
    ///
    /// An absent key, an explicit null and a value are three distinct states
    /// and `Option<Option<_>>` exists to hold all three. Flattening
    /// `stop_reason` reported an explicit null as absent, and retaining
    /// `stop_sequence` unconditionally reported an absent key as an explicit
    /// null — both silent, and both invisible to a round-trip test that
    /// happens to pick the one combination that works.
    #[test]
    fn the_three_state_fields_on_a_message_delta_round_trip() {
        let states = [None, Some(json!(null)), Some(json!("STOP"))];
        for reason in &states {
            for sequence in &states {
                let mut delta = serde_json::Map::new();
                if let Some(reason) = reason {
                    delta.insert("stop_reason".into(), reason.clone());
                }
                if let Some(sequence) = sequence {
                    delta.insert("stop_sequence".into(), sequence.clone());
                }
                let body = json!({
                    "type": "message_delta",
                    "delta": Value::Object(delta),
                    "usage": {"output_tokens": 3}
                });

                let mut state = StreamState::default();
                let chunk = ingest(body.clone(), &mut state).expect("carries content");
                assert_eq!(
                    plain(&render_event(&chunk, "anthropic").expect("renders back")),
                    body,
                    "reason={reason:?} sequence={sequence:?}"
                );
            }
        }
    }

    /// A named event this gateway does not model keeps its whole payload, so a
    /// same-protocol render puts it back unchanged.
    #[test]
    fn an_unmodelled_event_rides_the_envelope_and_renders_back() {
        let (mut state, _) = started();
        let body = json!({"type": "message_pause", "reason": "handoff"});
        let chunk = ingest(body.clone(), &mut state).unwrap();
        assert_eq!(chunk.id, "msg_1");
        assert_eq!(plain(&render_event(&chunk, "anthropic").unwrap()), body);
    }

    /// The round trip this module CAN promise, event by event. Reassembly
    /// equivalence is the transcript-level claim and lives in `reassembly.rs`;
    /// this is the per-event half of it.
    #[test]
    fn every_content_bearing_event_round_trips() {
        let (mut state, _) = started();
        for body in [
            json!({"type": "content_block_start", "index": 0,
                   "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_start", "index": 1,
                   "content_block": {"type": "tool_use", "id": "tu_1",
                                     "name": "weather", "input": {}}}),
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "text_delta", "text": "hi"}}),
            json!({"type": "content_block_delta", "index": 1,
                   "delta": {"type": "input_json_delta", "partial_json": "{\"a\":"}}),
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "thinking_delta", "thinking": "step"}}),
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "signature_delta", "signature": "sig"}}),
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "citations_delta", "citation": {"x": 1}}}),
            json!({"type": "message_delta",
                   "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                   "usage": {"output_tokens": 16}}),
        ] {
            let chunk = ingest(body.clone(), &mut state).expect("carries content");
            assert_eq!(
                plain(&render_event(&chunk, "anthropic").expect("renders back")),
                body,
                "round trip changed the event"
            );
        }
    }

    /// `message_start` carries a whole `Message`, so its round trip is the one
    /// that exercises the retained envelope rather than a single block.
    #[test]
    fn message_start_round_trips_with_its_whole_envelope() {
        let body = json!({
            "type": "message_start",
            "message": {
                "id": "msg_1", "type": "message", "role": "assistant",
                "model": "claude-opus-5", "content": [],
                "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": 21, "output_tokens": 0},
                "vendor_tag": "vt"
            }
        });
        let mut state = StreamState::default();
        let chunk = ingest(body.clone(), &mut state).unwrap();
        assert_eq!(plain(&render_event(&chunk, "anthropic").unwrap()), body);
    }

    /// The layer rule, on the streamed path. Another protocol's retained
    /// fields must not reach this wire — and going through `merged_ext` rather
    /// than `ext.get` is what enforces it, since `Protocol::may_read` denies
    /// this renderer the openai_compat bag however the provider is named.
    #[test]
    fn another_protocols_residue_never_reaches_this_wire() {
        let (mut state, _) = started();
        let mut chunk = ingest(
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "text_delta", "text": "hi"}}),
            &mut state,
        )
        .unwrap();
        chunk.completions[0].delta.ext.insert(
            Protocol::OpenAiCompat.as_str().into(),
            json!({"delta": {"type": "text_delta", "text": "SMUGGLED"}, "index": 99}),
        );
        let rendered = plain(&render_event(&chunk, Protocol::OpenAiCompat.as_str()).unwrap());
        assert_eq!(rendered["delta"]["text"], json!("hi"));
        assert_eq!(rendered["index"], json!(0));
    }

    /// A chunk carrying only a gateway form, as another protocol's would be.
    fn foreign(
        content: Vec<ContentDelta>,
        usage: Option<types::Usage>,
    ) -> types::ChatResponseChunk {
        types::ChatResponseChunk {
            id: "chatcmpl-1".into(),
            model: "gpt-4o".into(),
            created: Some(1),
            completions: vec![types::CompletionDelta {
                delta: types::MessageDelta {
                    role: None,
                    content,
                    ext: types::ProviderExt::new(),
                },
                finish_reason: usage.as_ref().map(|_| FinishReason::Stop),
                finish_reason_raw: usage.as_ref().map(|_| "stop".to_string()),
                ext: types::ProviderExt::new(),
            }],
            usage,
            ext: types::ProviderExt::new(),
            source: None,
        }
    }

    /// A tool call that names itself is an OPENER, and this protocol spells one
    /// with `content_block_start`. Rendering it as an `input_json_delta` threw
    /// away the `id` and the `name`, and since a chat-completions opener
    /// carries no argument text, the whole event came back `None` — the one
    /// event that names a tool call, rendered as silence.
    #[test]
    fn a_foreign_tool_call_opener_renders_as_a_block_start() {
        let rendered = plain(
            &render_event(
                &foreign(
                    vec![ContentDelta::ToolCall {
                        index: 0,
                        id: Some("call_abc".into()),
                        kind: Some("function".into()),
                        name: Some("get_weather".into()),
                        arguments: None,
                    }],
                    None,
                ),
                "openai",
            )
            .expect("an opener is not silence"),
        );
        assert_eq!(rendered["type"], json!("content_block_start"));
        assert_eq!(rendered["content_block"]["type"], json!("tool_use"));
        assert_eq!(rendered["content_block"]["id"], json!("call_abc"));
        assert_eq!(rendered["content_block"]["name"], json!("get_weather"));

        // ...and a fragment that names nothing is still a fragment.
        let fragment = plain(
            &render_event(
                &foreign(
                    vec![ContentDelta::ToolCall {
                        index: 0,
                        id: None,
                        kind: None,
                        name: None,
                        arguments: Some("{\"a\":".into()),
                    }],
                    None,
                ),
                "openai",
            )
            .unwrap(),
        );
        assert_eq!(fragment["delta"]["type"], json!("input_json_delta"));
        assert_eq!(fragment["delta"]["partial_json"], json!("{\"a\":"));
    }

    /// A foreign chunk's totals live on the TYPED field, not in this protocol's
    /// residue, and reading only the residue reported every one of them as
    /// zero. The cache counts come back out of the input on the way, because
    /// Anthropic's `input_tokens` excludes what the gateway's includes.
    #[test]
    fn a_foreign_chunks_usage_reaches_the_wire() {
        let rendered = plain(
            &render_event(
                &foreign(
                    Vec::new(),
                    Some(types::Usage {
                        input_tokens: Some(9237),
                        output_tokens: Some(16),
                        total_tokens: Some(9253),
                        reasoning_tokens: None,
                        cache_read_tokens: Some(8192),
                        cache_write_tokens: Some(1024),
                        ext: types::ProviderExt::new(),
                    }),
                ),
                "openai",
            )
            .expect("a chunk with a stop reason is an event"),
        );
        assert_eq!(rendered["type"], json!("message_delta"));
        assert_eq!(rendered["usage"]["output_tokens"], json!(16));
        // 9237 total minus the 9216 that were cached.
        assert_eq!(rendered["usage"]["input_tokens"], json!(21));
        assert_eq!(rendered["usage"]["cache_read_input_tokens"], json!(8192));
        assert_eq!(
            rendered["usage"]["cache_creation_input_tokens"],
            json!(1024)
        );
    }

    /// A chunk from another protocol has no retained wire delta, so it renders
    /// from the gateway form alone — which is the path an OpenAI-shaped stream
    /// takes when it is forwarded to Anthropic.
    #[test]
    fn a_foreign_chunk_renders_from_the_gateway_form() {
        let chunk = types::ChatResponseChunk {
            id: "chatcmpl-1".into(),
            model: "gpt-4o".into(),
            created: Some(1),
            completions: vec![types::CompletionDelta {
                delta: types::MessageDelta {
                    role: None,
                    content: vec![ContentDelta::Text {
                        text: "hello".into(),
                    }],
                    ext: types::ProviderExt::new(),
                },
                finish_reason: None,
                finish_reason_raw: None,
                ext: types::ProviderExt::new(),
            }],
            usage: None,
            ext: types::ProviderExt::new(),
            source: None,
        };
        assert_eq!(
            plain(&render_event(&chunk, "openai").unwrap()),
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "text_delta", "text": "hello"}})
        );
    }
}
