//! Chunk conversions: OpenAI-compatible wire → gateway (ingest) and gateway →
//! wire (render), one value per chunk. The whole-reply counterpart, and the
//! model for everything here, is [`super::response`].
//!
//! Round trips are lossless with zero permitted transformations, which for a
//! chunk means holding three distinct states of the same field: a key the
//! provider never sent stays absent, an explicit `null` comes back as one, and
//! a value comes back as itself. OpenAI streams `"logprobs": null` on every
//! chunk, `"usage": null` on every chunk but the last, and opens with
//! `"content": ""` — all three appear in one live stream.
//!
//! Protocol-specific fields (`object`, `service_tier`, choice `index`,
//! `refusal`, `obfuscation`, …) ride `ext["openai_compat"]` at their nesting
//! level and are restored verbatim.

use serde_json::Value;

use crate::gateway::types::{self, ContentDelta, FinishReason, IngestSource};
use crate::protocol::openai_compat::chat_completions::types::{
    ChatCompletionMessageToolCallChunk, ChatCompletionStreamResponseDelta,
    CreateChatCompletionStreamResponse, StreamChoice, ToolCallChunkFunction, UnknownFields,
};
use crate::types::Protocol;

use super::response::{
    ingest_usage, plain, render_finish_reason, render_usage, take_nullable_string, take_string,
    take_typed,
};
use crate::gateway::openai_compat::{merged_ext, namespaced, role_from_wire, role_to_wire};

/// Permissive and infallible; the exhaustive destructures at every level make
/// dropping a newly-typed wire field a compile error.
pub(crate) fn ingest_chunk(chunk: CreateChatCompletionStreamResponse) -> types::ChatResponseChunk {
    // Wire structs are plain serde data; serialization cannot fail.
    let body = serde_json::to_value(&chunk).expect("wire chunk serializes");
    let CreateChatCompletionStreamResponse {
        id,
        choices,
        created,
        model,
        object,
        moderation,
        service_tier,
        system_fingerprint,
        usage,
        unknown_fields,
    } = chunk;
    let mut compat = UnknownFields::new();
    compat.insert("object".into(), Value::String(object));
    if let Some(moderation) = moderation {
        compat.insert("moderation".into(), plain(&moderation));
    }
    if let Some(tier) = service_tier {
        // The outer `Some` is "the key was there"; an explicit null is stashed
        // as one so the renderer can put it back as one.
        compat.insert(
            "service_tier".into(),
            tier.map_or(Value::Null, Value::String),
        );
    }
    if let Some(fingerprint) = system_fingerprint {
        compat.insert("system_fingerprint".into(), Value::String(fingerprint));
    }
    // `usage` is optional AND nullable, and the null is the common case: it
    // rides on every chunk of a stream that asked for totals except the one
    // that carries them. Stashing the null keeps that chunk byte-identical.
    let usage = match usage {
        Some(Some(totals)) => Some(ingest_usage(totals)),
        Some(None) => {
            compat.insert("usage".into(), Value::Null);
            None
        }
        None => None,
    };
    compat.extend(unknown_fields);
    types::ChatResponseChunk {
        id,
        model,
        created: Some(created),
        completions: choices.into_iter().map(ingest_stream_choice).collect(),
        usage,
        ext: namespaced(compat),
        source: Some(IngestSource {
            protocol: Protocol::OpenAiCompat,
            body,
        }),
    }
}

fn ingest_stream_choice(choice: StreamChoice) -> types::CompletionDelta {
    let StreamChoice {
        delta,
        finish_reason,
        index,
        logprobs,
        unknown_fields,
    } = choice;
    let mut compat = UnknownFields::new();
    compat.insert("index".into(), Value::from(index));
    if let Some(logprobs) = logprobs {
        compat.insert("logprobs".into(), plain(&logprobs));
    }
    compat.extend(unknown_fields);
    types::CompletionDelta {
        delta: ingest_delta(delta),
        finish_reason: finish_reason.as_deref().map(FinishReason::from_raw),
        finish_reason_raw: finish_reason,
        ext: namespaced(compat),
    }
}

fn ingest_delta(delta: ChatCompletionStreamResponseDelta) -> types::MessageDelta {
    let ChatCompletionStreamResponseDelta {
        content,
        function_call,
        refusal,
        role,
        tool_calls,
        unknown_fields,
    } = delta;
    let mut compat = UnknownFields::new();
    let role = role.map(|raw| {
        let (role, raw_role) = role_from_wire(raw);
        if let Some(raw) = raw_role {
            compat.insert("role".into(), Value::String(raw));
        }
        role
    });
    if let Some(refusal) = refusal {
        compat.insert("refusal".into(), refusal.map_or(Value::Null, Value::String));
    }
    if let Some(function_call) = function_call {
        compat.insert("function_call".into(), plain(&function_call));
    }
    // Reasoning first, since it precedes the answer it led to — the same order
    // `ingest_message` produces for a whole reply.
    let mut deltas: Vec<ContentDelta> = reasoning_delta(&unknown_fields).into_iter().collect();
    match content {
        // An empty string is a value, not an absence: a stream opens with
        // `"content": ""`, and dropping it would lose the chunk that says the
        // answer has started.
        Some(Some(text)) => deltas.push(ContentDelta::Text { text }),
        Some(None) => {
            compat.insert("content".into(), Value::Null);
        }
        None => {}
    }
    match tool_calls {
        // `Some([])` is distinguishable from absent; stash it so render
        // re-emits the empty array byte-for-byte.
        Some(calls) if calls.is_empty() => {
            compat.insert("tool_calls".into(), Value::Array(Vec::new()));
        }
        Some(calls) => deltas.extend(calls.into_iter().map(ingest_tool_call_chunk)),
        None => {}
    }
    compat.extend(unknown_fields);
    types::MessageDelta {
        role,
        content: deltas,
        ext: namespaced(compat),
    }
}

/// A fragment of `reasoning_content`, the chain-of-thought sibling of
/// `content`, as a typed delta.
///
/// PROMOTE BUT RETAIN, exactly as [`super::response`] does it for a whole
/// reply: the caller leaves `unknown_fields` intact, so the field still rides
/// `ext["openai_compat"]` and the renderer replays it verbatim. Nothing could
/// rebuild it from the delta — `render_delta` drops Reasoning, because the
/// protocol has no field to render it into.
///
/// The emptiness rule differs from the whole-reply one deliberately. There, an
/// empty string means "the provider sent no thinking" and is not promoted.
/// Here a provider may well open a reasoning run with an empty fragment the way
/// it opens `content` with one, so only a non-string or an explicit null is
/// declined — and the retained ext copy makes the choice unobservable on a
/// same-protocol round trip either way.
fn reasoning_delta(fields: &UnknownFields) -> Option<ContentDelta> {
    Some(ContentDelta::Reasoning {
        text: fields.get("reasoning_content")?.as_str()?.to_string(),
        // The same one string the ext namespace is keyed by, so the two cannot
        // disagree about where this fragment came from.
        provenance: Some(Protocol::OpenAiCompat.as_str().to_string()),
    })
}

/// A plain function fragment becomes a typed delta; anything else — a
/// nonstandard `type`, a fragment with no `function` at all, or unknown fields
/// at either level — passes through as an `Unknown` delta, re-emitted verbatim
/// in array order.
///
/// `type` is carried rather than assumed: it arrives on the fragment that opens
/// a call and is absent on every fragment after it, so restoring a constant
/// would put a key on the wire the provider never sent.
fn ingest_tool_call_chunk(call: ChatCompletionMessageToolCallChunk) -> ContentDelta {
    match &call {
        ChatCompletionMessageToolCallChunk {
            index,
            id,
            function: Some(function),
            r#type,
            unknown_fields,
        } if r#type.as_deref().is_none_or(|kind| kind == "function")
            && unknown_fields.is_empty()
            && function.unknown_fields.is_empty()
            // `function: {}` carries presence this variant cannot express, so
            // it takes the verbatim path rather than being flattened away.
            && (function.name.is_some() || function.arguments.is_some()) =>
        {
            ContentDelta::ToolCall {
                index: *index,
                id: id.clone(),
                kind: r#type.clone(),
                name: function.name.clone(),
                arguments: function.arguments.clone(),
            }
        }
        _ => ContentDelta::Unknown(plain(&call)),
    }
}

/// Infallible: protocol fields restore from ext (a hook that corrupted a
/// stashed value beyond its wire type falls back to dropping that field).
/// One stream's map from the IR's tool-call correlation key to the ordinal
/// this protocol requires.
///
/// The two are not the same number, and that is the whole reason this exists.
/// [`ContentDelta::ToolCall::index`](types::ContentDelta) is documented as a
/// stable CORRELATION KEY — it says which call a fragment belongs to and
/// nothing more — while chat completions' `tool_calls[].index` is a 0-based
/// index INTO the array a client assembles. Anthropic numbers every content
/// block, so a reply that opens with text puts its first tool call at key 1;
/// copying that through leaves a hole at 0, and openai-python's own streaming
/// accumulator indexes `tool_calls` by that value — an out-of-range failure,
/// not a cosmetic mismatch.
///
/// Scoped per CHOICE as well as per stream, because `tool_calls[].index` is
/// numbered within one choice: two choices each opening a call would otherwise
/// have the second one renumbered to 1. Anthropic has no multi-choice concept
/// so its path never exercises that, but this renderer also serves `n > 1` on
/// its own protocol, where it would be wrong.
///
/// On a same-protocol stream this is the IDENTITY, which is what makes it safe
/// to apply everywhere rather than only on a translated path. Chat completions'
/// own indices are already dense and 0-based in first-seen order, so mapping
/// first-seen to position returns them unchanged.
///
/// Lives here rather than in the Anthropic gateway because the number being
/// computed is THIS protocol's — the same seam `render_finish_reason` and
/// `render_tool_call_kind` sit on. It is owned by
/// [`ChatCompletionStream`](crate::ChatCompletionStream), the only scope that
/// outlives a chunk.
#[derive(Debug, Default)]
pub(crate) struct ToolCallOrdinals {
    /// `(choice, correlation key)` in first-seen order. An entry's POSITION
    /// among the entries of its own choice IS the ordinal, so nothing is
    /// counted or stored twice.
    ///
    /// A `Vec` rather than a `HashMap`: one reply holds a handful of tool
    /// calls, so the scan is over that handful, and the length is bounded by
    /// the reply rather than accumulating across them. A map would still need
    /// a per-choice counter beside it to answer the same question.
    seen: Vec<(u32, u32)>,
}

impl ToolCallOrdinals {
    /// This key's ordinal within `choice`, assigning the next one if the key
    /// is new. One pass, because finding the entry and counting the entries
    /// before it are the same walk.
    fn ordinal(&mut self, choice: u32, key: u32) -> u32 {
        let mut ordinal = 0;
        for &(seen_choice, seen_key) in &self.seen {
            if seen_choice != choice {
                continue;
            }
            if seen_key == key {
                return ordinal;
            }
            ordinal += 1;
        }
        self.seen.push((choice, key));
        ordinal
    }
}

pub(crate) fn render_chunk(
    chunk: &types::ChatResponseChunk,
    provider: &str,
    ordinals: &mut ToolCallOrdinals,
) -> CreateChatCompletionStreamResponse {
    let mut unknown_fields = merged_ext(&chunk.ext, provider);
    let object = take_string(&mut unknown_fields, "object")
        .unwrap_or_else(|| "chat.completion.chunk".to_string());
    // A stashed null and a typed total are the two ways `usage` can be
    // present; the typed one wins, since only ingest could have produced both.
    let stashed_usage = unknown_fields.remove("usage");
    let usage = match (&chunk.usage, stashed_usage) {
        (Some(usage), _) => Some(Some(render_usage(usage, provider))),
        (None, Some(Value::Null)) => Some(None),
        (None, _) => None,
    };
    CreateChatCompletionStreamResponse {
        id: chunk.id.clone(),
        choices: chunk
            .completions
            .iter()
            .enumerate()
            .map(|(position, completion)| {
                render_stream_choice(completion, position, provider, ordinals)
            })
            .collect(),
        created: chunk.created.unwrap_or(0),
        model: chunk.model.clone(),
        object,
        moderation: take_typed(&mut unknown_fields, "moderation"),
        service_tier: take_nullable_string(&mut unknown_fields, "service_tier"),
        system_fingerprint: take_string(&mut unknown_fields, "system_fingerprint"),
        usage,
        unknown_fields,
    }
}

fn render_stream_choice(
    completion: &types::CompletionDelta,
    position: usize,
    provider: &str,
    ordinals: &mut ToolCallOrdinals,
) -> StreamChoice {
    let mut unknown_fields = merged_ext(&completion.ext, provider);
    let index = unknown_fields
        .remove("index")
        .and_then(|v| v.as_u64())
        .unwrap_or(position as u64) as u32;
    StreamChoice {
        // The choice index goes DOWN into the delta as well as onto the wire:
        // a tool call's ordinal is numbered within its own choice. See
        // [`ToolCallOrdinals`].
        delta: render_delta(&completion.delta, provider, index, ordinals),
        finish_reason: render_delta_finish_reason(completion),
        index,
        logprobs: take_typed(&mut unknown_fields, "logprobs"),
        unknown_fields,
    }
}

/// [`render_finish_reason`] for a chunk, where the field is absent on every
/// chunk of a choice but the last.
///
/// A chunk needs the same translation a whole reply does, and for a sharper
/// reason: a streamed call and its unstreamed twin must END THE SAME WAY. If
/// only the whole-reply renderer spoke this protocol's vocabulary, the same
/// request to the same model would report `tool_calls` unstreamed and
/// `tool_use` streamed, and a caller branching on it would take a different
/// path for no reason it could see.
fn render_delta_finish_reason(completion: &types::CompletionDelta) -> Option<String> {
    let raw = completion.finish_reason_raw.as_deref()?;
    // The pair is documented to be absent and present together. `Other` is what
    // a raw string this crate does not recognize reads as anyway, so it is the
    // inert answer if they ever came apart — the helper echoes `raw` for it.
    Some(render_finish_reason(
        raw,
        completion.finish_reason.unwrap_or(FinishReason::Other),
    ))
}

/// A tool call's `kind` in THIS protocol's vocabulary.
///
/// The same rule [`render_finish_reason`] states, applied to the other field
/// where the two protocols disagree on a word: Anthropic calls a tool call
/// `tool_use`, and this wire's `type` has no such value — every official SDK
/// reads it as unrecognized, and a client matching on `"function"` silently
/// drops the call.
///
/// `None` stays `None`, which is load-bearing rather than incidental: the type
/// arrives on the fragment that OPENS a call and is absent on the rest, so a
/// form that supplied a default here would put a key on the wire the provider
/// never sent. An unrecognized word echoes — this protocol has `custom` tools
/// too, and inventing `function` for a word that might be one is worse than
/// passing through what a future SDK may well know.
fn render_tool_call_kind(kind: Option<&str>) -> Option<String> {
    Some(match kind? {
        "tool_use" => "function".to_string(),
        known => known.to_string(),
    })
}

fn render_delta(
    delta: &types::MessageDelta,
    provider: &str,
    choice: u32,
    ordinals: &mut ToolCallOrdinals,
) -> ChatCompletionStreamResponseDelta {
    let mut unknown_fields = merged_ext(&delta.ext, provider);
    let role = match unknown_fields.remove("role") {
        Some(Value::String(raw)) => Some(raw),
        _ => delta.role.map(|role| role_to_wire(role).to_string()),
    };
    let mut content = None;
    let mut tool_calls = Vec::new();
    for fragment in &delta.content {
        match fragment {
            ContentDelta::Text { text } => content = Some(Some(text.clone())),
            ContentDelta::ToolCall {
                index,
                id,
                kind,
                name,
                arguments,
            } => tool_calls.push(ChatCompletionMessageToolCallChunk {
                // The IR's value is a correlation key; this protocol's field is
                // an ordinal. See [`ToolCallOrdinals`] for why they differ and
                // why translating is an identity on a same-protocol stream.
                index: ordinals.ordinal(choice, *index),
                id: id.clone(),
                function: Some(ToolCallChunkFunction {
                    arguments: arguments.clone(),
                    name: name.clone(),
                    unknown_fields: UnknownFields::new(),
                }),
                r#type: render_tool_call_kind(kind.as_deref()),
                unknown_fields: UnknownFields::new(),
            }),
            // Retained residue from THIS protocol, replayed verbatim — its
            // `index` was already an ordinal when it arrived, and
            // `Protocol::may_read` is what guarantees another protocol's
            // residue never reaches here to be mistaken for one.
            //
            // It still has to CLAIM that ordinal. A passthrough call that took
            // slot 0 without registering it left the slot free, so the next
            // typed call was numbered 0 as well and the client merged two
            // distinct calls into one array element. Registering is the
            // identity for residue — first-seen order on a same-protocol
            // stream is the order the indices already have — so this reserves
            // the slot without renumbering anything.
            ContentDelta::Unknown(value) => {
                if let Ok(mut call) =
                    serde_json::from_value::<ChatCompletionMessageToolCallChunk>(value.clone())
                {
                    call.index = ordinals.ordinal(choice, call.index);
                    tool_calls.push(call);
                }
            }
            // A Reasoning fragment was lifted out of `reasoning_content`,
            // which is still sitting in ext and about to be emitted verbatim —
            // PROMOTE BUT RETAIN, read from the other end, so dropping it
            // loses nothing.
            ContentDelta::Reasoning { .. } => {}
        }
    }
    // A stashed null only stands in for content nothing else claimed.
    if content.is_none() && matches!(unknown_fields.get("content"), Some(Value::Null)) {
        content = Some(None);
    }
    unknown_fields.remove("content");
    if !tool_calls.is_empty() {
        // Typed fragments are authoritative over any stashed empty array.
        unknown_fields.remove("tool_calls");
    }
    ChatCompletionStreamResponseDelta {
        content,
        function_call: take_typed(&mut unknown_fields, "function_call"),
        refusal: take_nullable_string(&mut unknown_fields, "refusal"),
        role,
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        unknown_fields,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::gateway::types::Role;

    fn parse(body: Value) -> CreateChatCompletionStreamResponse {
        serde_json::from_value(body).unwrap()
    }

    fn ending_chunk(raw: &str, reason: FinishReason) -> types::ChatResponseChunk {
        types::ChatResponseChunk {
            id: "chunk-1".into(),
            model: "claude-opus-5".into(),
            created: None,
            completions: vec![types::CompletionDelta {
                delta: types::MessageDelta::default(),
                finish_reason: Some(reason),
                finish_reason_raw: Some(raw.to_string()),
                ext: types::ProviderExt::new(),
            }],
            usage: None,
            ext: types::ProviderExt::new(),
            source: None,
        }
    }

    /// A chunk that came in on ANOTHER protocol must end in this one's
    /// vocabulary, exactly as a whole reply does — and for a sharper reason:
    /// the streamed and unstreamed forms of one request must end the same way,
    /// or a caller branching on `finish_reason` takes a different path for a
    /// difference it cannot see.
    #[test]
    fn a_foreign_finish_reason_is_rendered_in_this_protocols_words() {
        let rows = [
            ("tool_use", FinishReason::ToolCalls, "tool_calls"),
            ("end_turn", FinishReason::Stop, "stop"),
            ("max_tokens", FinishReason::Length, "length"),
            ("refusal", FinishReason::ContentFilter, "content_filter"),
        ];
        for (raw, reason, expected) in rows {
            let wire = render_chunk(
                &ending_chunk(raw, reason),
                "anthropic",
                &mut ToolCallOrdinals::default(),
            );
            assert_eq!(
                wire.choices[0].finish_reason.as_deref(),
                Some(expected),
                "{raw}"
            );
        }
    }

    /// ...and the two really do agree, which is the property the shared helper
    /// exists for. Comparing them directly is what would catch one renderer
    /// being changed without the other.
    #[test]
    fn a_chunk_and_a_whole_reply_end_the_same_way() {
        for (raw, reason) in [
            ("tool_use", FinishReason::ToolCalls),
            ("end_turn", FinishReason::Stop),
            ("stop", FinishReason::Stop),
            ("function_call", FinishReason::Other),
        ] {
            let chunk = render_chunk(
                &ending_chunk(raw, reason),
                "anthropic",
                &mut ToolCallOrdinals::default(),
            );
            assert_eq!(
                chunk.choices[0].finish_reason,
                Some(super::super::response::render_finish_reason(raw, reason)),
                "{raw}"
            );
        }
    }

    /// Every chunk of a choice but the last carries no finish reason at all,
    /// and must not grow one.
    #[test]
    fn a_chunk_that_does_not_end_a_choice_carries_no_finish_reason() {
        let mut chunk = ending_chunk("stop", FinishReason::Stop);
        chunk.completions[0].finish_reason = None;
        chunk.completions[0].finish_reason_raw = None;
        assert_eq!(
            render_chunk(&chunk, "anthropic", &mut ToolCallOrdinals::default()).choices[0]
                .finish_reason,
            None
        );
    }

    fn round_trip(body: Value, provider: &str) {
        let normalized = ingest_chunk(parse(body.clone()));
        let rendered = render_chunk(&normalized, provider, &mut ToolCallOrdinals::default());
        assert_eq!(serde_json::to_value(&rendered).unwrap(), body);
    }

    /// The maximal chunk: every documented field, unknown fields at every
    /// nesting level, and a tool-call fragment.
    fn maximal_chunk() -> Value {
        json!({
            "id": "chatcmpl-123",
            "choices": [{
                "delta": {
                    "content": "Hello",
                    "function_call": {"arguments": "{\"q\":", "name": "legacy_fn"},
                    "refusal": null,
                    "role": "assistant",
                    "tool_calls": [{
                        "index": 0,
                        "id": "call-1",
                        "function": {"arguments": "{\"z\":1,\"a\":2}", "name": "search"},
                        "type": "function"
                    }],
                    "reasoning_content": "step by step"
                },
                "finish_reason": "tool_calls",
                "index": 3,
                "logprobs": {
                    "content": [{
                        "token": "Hello",
                        "bytes": [72, 101, 108, 108, 111],
                        "logprob": -0.1,
                        "top_logprobs": [{"token": "Hello", "bytes": null, "logprob": -0.1}]
                    }],
                    "refusal": null
                },
                "new_choice_field": true
            }],
            "created": 1_700_000_000,
            "model": "gpt-5.6",
            "object": "chat.completion.chunk",
            "service_tier": "default",
            "system_fingerprint": "fp_44709d6fcb",
            "usage": {
                "completion_tokens": 12,
                "prompt_tokens": 9,
                "total_tokens": 21,
                "prompt_tokens_details": {"cached_tokens": 3}
            },
            "obfuscation": "8Xk2p"
        })
    }

    /// A chunk carrying one tool-call fragment per `(choice, key)` pair given.
    fn call_chunk(fragments: &[(u32, u32)]) -> types::ChatResponseChunk {
        types::ChatResponseChunk {
            id: "msg_1".into(),
            model: "claude-opus-5".into(),
            created: None,
            completions: fragments
                .iter()
                .map(|&(choice, key)| {
                    let mut ext = UnknownFields::new();
                    ext.insert("index".into(), json!(choice));
                    types::CompletionDelta {
                        delta: types::MessageDelta {
                            role: None,
                            content: vec![ContentDelta::ToolCall {
                                index: key,
                                id: Some(format!("tu_{key}")),
                                kind: Some("tool_use".into()),
                                name: Some("get_weather".into()),
                                arguments: None,
                            }],
                            ext: types::ProviderExt::new(),
                        },
                        finish_reason: None,
                        finish_reason_raw: None,
                        ext: namespaced(ext),
                    }
                })
                .collect(),
            usage: None,
            ext: types::ProviderExt::new(),
            source: None,
        }
    }

    /// The wire `tool_calls[].index` of every fragment in one rendered chunk.
    fn wire_indices(fragments: &[(u32, u32)], ordinals: &mut ToolCallOrdinals) -> Vec<Option<u32>> {
        render_chunk(&call_chunk(fragments), "anthropic", ordinals)
            .choices
            .iter()
            .map(|choice| choice.delta.tool_calls.as_ref().unwrap()[0].index.into())
            .collect()
    }

    /// THE bug this whole map exists for. Anthropic numbers every content
    /// block, so a reply that opens with text puts its first tool call at
    /// key 1 — and `tool_calls[].index` is an index INTO the array a client
    /// assembles, so copying 1 through leaves a hole at 0. openai-python's own
    /// streaming accumulator indexes `tool_calls` by that value, which makes it
    /// an out-of-range failure rather than a cosmetic mismatch.
    #[test]
    fn a_tool_calls_index_is_an_ordinal_not_the_correlation_key() {
        let mut ordinals = ToolCallOrdinals::default();
        assert_eq!(
            wire_indices(&[(0, 1)], &mut ordinals),
            vec![Some(0)],
            "Anthropic's block index reached an OpenAI caller as an array index"
        );
    }

    /// Every fragment of ONE call answers with the same ordinal, which is what
    /// makes the ordinal usable as the array slot it claims to be. The opener
    /// and its argument fragments arrive as separate chunks.
    #[test]
    fn every_fragment_of_one_call_keeps_its_ordinal() {
        let mut ordinals = ToolCallOrdinals::default();
        for _ in 0..3 {
            assert_eq!(wire_indices(&[(0, 1)], &mut ordinals), vec![Some(0)]);
        }
    }

    /// A second call takes the next slot, in the order the calls were first
    /// seen rather than in the order of their keys.
    #[test]
    fn each_new_call_takes_the_next_slot() {
        let mut ordinals = ToolCallOrdinals::default();
        // Keys 5 then 2, deliberately not ascending: first-seen is the rule,
        // so a version that sorted or subtracted a base would answer 3 and 0.
        assert_eq!(wire_indices(&[(0, 5)], &mut ordinals), vec![Some(0)]);
        assert_eq!(wire_indices(&[(0, 2)], &mut ordinals), vec![Some(1)]);
        assert_eq!(wire_indices(&[(0, 5)], &mut ordinals), vec![Some(0)]);
    }

    /// On a SAME-protocol stream the map is the identity, which is what makes
    /// it safe to apply on every path rather than only on a translated one.
    /// This protocol's own indices are already dense and 0-based in first-seen
    /// order, so first-seen-to-position returns them unchanged.
    #[test]
    fn a_same_protocol_streams_indices_are_unchanged() {
        let mut ordinals = ToolCallOrdinals::default();
        for key in 0..4 {
            assert_eq!(
                wire_indices(&[(0, key)], &mut ordinals),
                vec![Some(key)],
                "key {key} was renumbered on a stream that needed no remap"
            );
        }
    }

    /// One tool-call fragment on choice 0, as it arrives on the wire.
    fn tool_call_chunk(call: Value) -> Value {
        json!({
            "id": "chatcmpl_1",
            "created": 1_700_000_000,
            "model": "gpt-5.6",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"tool_calls": [call]}}]
        })
    }

    /// The wire index a chunk's single tool call is rendered at, through the
    /// REAL ingest path rather than a hand-built IR delta — which is the point
    /// here, since the bug is about which path ingest chooses.
    fn round_tripped_index(call: Value, ordinals: &mut ToolCallOrdinals) -> u32 {
        render_chunk(
            &ingest_chunk(parse(tool_call_chunk(call))),
            "openai",
            ordinals,
        )
        .choices[0]
            .delta
            .tool_calls
            .as_ref()
            .expect("the chunk carried a tool call")[0]
            .index
    }

    /// A passthrough tool call OCCUPIES the slot it is replayed into.
    ///
    /// A `custom` tool is a `type` this protocol has but `ContentDelta::ToolCall`
    /// cannot express, so ingest sends it down the verbatim path — and sent
    /// whole in one chunk, NO fragment of that call ever takes the typed path
    /// to register its key. A version that replayed it without claiming index 0
    /// left the slot free, and the next typed call was numbered 0 as well:
    /// two distinct calls merged into one element of the client's array.
    #[test]
    fn a_passthrough_tool_call_claims_its_ordinal() {
        let mut ordinals = ToolCallOrdinals::default();
        assert_eq!(
            round_tripped_index(
                json!({
                    "index": 0,
                    "id": "call_a",
                    "type": "custom",
                    "function": {"name": "run", "arguments": "{}"}
                }),
                &mut ordinals
            ),
            0,
            "a verbatim replay renumbered the index it arrived with"
        );
        assert_eq!(
            round_tripped_index(
                json!({
                    "index": 1,
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": ""}
                }),
                &mut ordinals
            ),
            1,
            "a typed call took the slot a passthrough call already held"
        );
    }

    /// `tool_calls[].index` is numbered within ONE choice, so two choices each
    /// opening a call are both slot 0. A map scoped per stream alone would
    /// renumber the second choice's call to 1 and put it in a slot its own
    /// array does not have.
    ///
    /// Anthropic has no multi-choice concept so its path never reaches this,
    /// but the same renderer serves `n > 1` on its own protocol.
    #[test]
    fn ordinals_are_numbered_within_a_choice() {
        let mut ordinals = ToolCallOrdinals::default();
        // DIFFERENT keys per choice, and that is the whole design of this test.
        // With the same key in both, a map that ignored the choice entirely
        // would still answer 0 twice — the second lookup would find the first
        // entry by key alone — and this would pass while proving nothing.
        // Distinct keys are what force the scoping to be real.
        assert_eq!(
            wire_indices(&[(0, 5), (1, 9)], &mut ordinals),
            vec![Some(0), Some(0)],
            "a per-stream map renumbered a call that belongs to another choice"
        );
        // And within each of those choices, numbering still advances.
        assert_eq!(
            wire_indices(&[(0, 6), (1, 5)], &mut ordinals),
            vec![Some(1), Some(1)]
        );
    }

    /// The mandated test, chunk-side: an OpenAI-compatible chunk must survive
    /// normalization and come back out with ZERO permitted transformations.
    #[test]
    fn openai_compat_chunk_round_trip_is_lossless() {
        round_trip(maximal_chunk(), "openai");

        // Spot-check the typed views the IR exposes along the way.
        let normalized = ingest_chunk(parse(maximal_chunk()));
        assert_eq!(normalized.id, "chatcmpl-123");
        assert_eq!(normalized.created, Some(1_700_000_000));
        let completion = &normalized.completions[0];
        assert_eq!(completion.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(completion.finish_reason_raw.as_deref(), Some("tool_calls"));
        assert_eq!(completion.ext["openai_compat"]["index"], json!(3));
        assert_eq!(completion.delta.role, Some(Role::Assistant));
        assert_eq!(
            normalized.usage.as_ref().unwrap().cache_read_tokens,
            Some(3)
        );
        assert_eq!(
            normalized.ext["openai_compat"]["obfuscation"],
            json!("8Xk2p")
        );
    }

    /// The chunk a provider that sends nothing optional would send.
    #[test]
    fn a_minimal_chunk_round_trips() {
        round_trip(
            json!({
                "id": "chatcmpl-1",
                "choices": [{"delta": {}, "finish_reason": null, "index": 0}],
                "created": 1_700_000_000,
                "model": "gpt-5.6",
                "object": "chat.completion.chunk"
            }),
            "openai",
        );
    }

    /// The three states of `content` a live stream sends, each of which has to
    /// come back out as itself.
    #[test]
    fn absent_null_and_empty_content_stay_distinct() {
        let chunk = |delta: Value| {
            json!({
                "id": "chatcmpl-1",
                "choices": [{"delta": delta, "finish_reason": null, "index": 0}],
                "created": 1_700_000_000,
                "model": "gpt-5.6",
                "object": "chat.completion.chunk"
            })
        };

        for delta in [json!({}), json!({"content": null}), json!({"content": ""})] {
            round_trip(chunk(delta), "openai");
        }

        // ...and the IR tells them apart, not merely the bytes.
        let absent = ingest_chunk(parse(chunk(json!({}))));
        assert!(absent.completions[0].delta.content.is_empty());

        let nulled = ingest_chunk(parse(chunk(json!({"content": null}))));
        assert!(nulled.completions[0].delta.content.is_empty());
        assert_eq!(
            nulled.completions[0].delta.ext["openai_compat"]["content"],
            Value::Null
        );

        let empty = ingest_chunk(parse(chunk(json!({"content": ""}))));
        assert!(matches!(
            &empty.completions[0].delta.content[0],
            ContentDelta::Text { text } if text.is_empty()
        ));
    }

    /// The vocabulary rule [`render_finish_reason`] states, on the other field
    /// the two protocols spell differently. Anthropic's `tool_use` has no
    /// meaning in this wire's `type`, so echoing it hands a client a value its
    /// own SDK reads as unrecognized and drops.
    ///
    /// The `None` row is the one that would be easiest to get wrong: a
    /// continuation fragment carries no type, and supplying one here would put
    /// a key on the wire the provider never sent.
    /// Rendered through `render_chunk` rather than by calling the helper
    /// directly: the bug this guards against was the CALL SITE echoing `kind`,
    /// and a test of the helper alone passes while the call site ignores it.
    #[test]
    fn a_tool_calls_kind_is_translated_into_this_protocols_vocabulary() {
        let rendered = |kind: Option<&str>| {
            let chunk = types::ChatResponseChunk {
                id: "msg_1".into(),
                model: "claude-opus-5".into(),
                created: None,
                completions: vec![types::CompletionDelta {
                    delta: types::MessageDelta {
                        role: None,
                        content: vec![ContentDelta::ToolCall {
                            index: 0,
                            id: Some("tu_1".into()),
                            kind: kind.map(str::to_string),
                            name: Some("get_weather".into()),
                            arguments: None,
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
            plain(&render_chunk(
                &chunk,
                "anthropic",
                &mut ToolCallOrdinals::default(),
            ))["choices"][0]["delta"]["tool_calls"][0]["type"]
                .clone()
        };

        assert_eq!(
            rendered(Some("tool_use")),
            json!("function"),
            "Anthropic's spelling reached an OpenAI caller, whose SDK has no such value"
        );
        assert_eq!(
            rendered(Some("function")),
            json!("function"),
            "this protocol's own spelling must survive unchanged"
        );
        assert_eq!(
            rendered(Some("custom")),
            json!("custom"),
            "a word this protocol has must not be rewritten to `function`"
        );
        assert_eq!(
            rendered(None),
            json!(null),
            "a fragment that carried no type must not gain one"
        );
    }

    /// The opener names the call and carries `type`; the fragments after it
    /// carry only an index and more argument text, and must not gain a `type`
    /// the provider never sent.
    #[test]
    fn tool_call_fragments_keep_their_shape() {
        let fragment = |call: Value| {
            json!({
                "id": "chatcmpl-1",
                "choices": [{
                    "delta": {"tool_calls": [call]},
                    "finish_reason": null,
                    "index": 0
                }],
                "created": 1_700_000_000,
                "model": "deepseek-v4-flash",
                "object": "chat.completion.chunk"
            })
        };
        let opener = json!({
            "index": 0,
            "id": "call_0_5f3a91c2",
            "type": "function",
            "function": {"name": "get_weather", "arguments": ""}
        });
        let continuation = json!({"index": 0, "function": {"arguments": "{\"city\":"}});

        round_trip(fragment(opener.clone()), "deepseek");
        round_trip(fragment(continuation.clone()), "deepseek");

        let normalized = ingest_chunk(parse(fragment(opener)));
        assert!(matches!(
            &normalized.completions[0].delta.content[0],
            ContentDelta::ToolCall { index: 0, id: Some(id), name: Some(name), .. }
                if id == "call_0_5f3a91c2" && name == "get_weather"
        ));

        let normalized = ingest_chunk(parse(fragment(continuation)));
        assert!(matches!(
            &normalized.completions[0].delta.content[0],
            ContentDelta::ToolCall { index: 0, id: None, kind: None, name: None, arguments: Some(a) }
                if a == "{\"city\":"
        ));
    }

    /// A fragment this conversion has no typed shape for passes through
    /// verbatim rather than being flattened into one that loses its residue.
    #[test]
    fn a_nonstandard_tool_call_fragment_passes_through_verbatim() {
        let chunk = |call: Value| {
            json!({
                "id": "chatcmpl-1",
                "choices": [{
                    "delta": {"tool_calls": [call]},
                    "finish_reason": null,
                    "index": 0
                }],
                "created": 1_700_000_000,
                "model": "gpt-5.6",
                "object": "chat.completion.chunk"
            })
        };
        let odd = [
            json!({"index": 0, "type": "custom", "function": {"arguments": "x"}}),
            json!({"index": 0, "function": {"arguments": "x"}, "vendor_extra": true}),
            json!({"index": 0, "function": {}}),
            json!({"index": 0, "id": "call-1"}),
        ];
        for call in odd {
            round_trip(chunk(call.clone()), "openai");
            let normalized = ingest_chunk(parse(chunk(call.clone())));
            assert!(
                matches!(&normalized.completions[0].delta.content[0], ContentDelta::Unknown(v) if *v == call),
                "{call} was not passed through verbatim"
            );
        }
    }

    /// `usage: null` rides every chunk of a stream that asked for totals
    /// except the one that carries them, so the null and the totals are both
    /// ordinary and neither may turn into the other.
    #[test]
    fn a_null_usage_is_not_an_absent_one() {
        let chunk = |usage: Value| {
            let mut body = json!({
                "id": "chatcmpl-1",
                "choices": [],
                "created": 1_700_000_000,
                "model": "gpt-5.6",
                "object": "chat.completion.chunk"
            });
            if !usage.is_null() {
                body.as_object_mut().unwrap().insert("usage".into(), usage);
            }
            body
        };

        round_trip(chunk(Value::Null), "openai");
        round_trip(chunk(json!(null)), "openai");
        round_trip(
            chunk(json!({"completion_tokens": 3, "prompt_tokens": 11, "total_tokens": 14})),
            "openai",
        );

        let mut nulled = chunk(Value::Null);
        nulled
            .as_object_mut()
            .unwrap()
            .insert("usage".into(), Value::Null);
        let normalized = ingest_chunk(parse(nulled.clone()));
        assert!(normalized.usage.is_none());
        assert_eq!(normalized.ext["openai_compat"]["usage"], Value::Null);
        assert_eq!(
            serde_json::to_value(render_chunk(
                &normalized,
                "openai",
                &mut ToolCallOrdinals::default()
            ))
            .unwrap(),
            nulled
        );
    }

    /// The usage-only chunk `stream_options.include_usage` adds: no choices at
    /// all, and the totals for the whole request.
    #[test]
    fn a_usage_only_chunk_round_trips() {
        round_trip(
            json!({
                "id": "chatcmpl-1",
                "choices": [],
                "created": 1_700_000_000,
                "model": "gpt-5.6",
                "object": "chat.completion.chunk",
                "usage": {
                    "completion_tokens": 3,
                    "prompt_tokens": 11,
                    "total_tokens": 14,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                    "prompt_tokens_details": {"cached_tokens": 0}
                }
            }),
            "openai",
        );
    }

    /// Promoted for any provider on this protocol, ordered before the answer
    /// it led to, and still replayed from ext — the same PROMOTE BUT RETAIN
    /// contract the whole-reply path states.
    #[test]
    fn reasoning_content_becomes_a_delta_before_the_answer() {
        let normalized = ingest_chunk(parse(maximal_chunk()));
        let content = &normalized.completions[0].delta.content;

        assert!(matches!(
            &content[0],
            ContentDelta::Reasoning { text, provenance }
                if text == "step by step" && provenance.as_deref() == Some("openai_compat")
        ));
        assert!(
            matches!(&content[1], ContentDelta::Text { .. }),
            "the answer must still follow the reasoning: {content:?}"
        );

        let rendered = render_chunk(&normalized, "deepseek", &mut ToolCallOrdinals::default());
        assert_eq!(
            rendered.choices[0].delta.unknown_fields["reasoning_content"],
            json!("step by step")
        );
    }

    /// A non-string `reasoning_content` must not panic and must not be
    /// promoted — ingest does not type this field.
    #[test]
    fn only_a_string_reasoning_content_is_promoted() {
        for value in [json!(null), json!(["not", "a", "string"]), json!(7)] {
            let mut body = maximal_chunk();
            body["choices"][0]["delta"]["reasoning_content"] = value.clone();
            let normalized = ingest_chunk(parse(body.clone()));
            assert!(
                !matches!(
                    normalized.completions[0].delta.content[0],
                    ContentDelta::Reasoning { .. }
                ),
                "{value} was promoted"
            );
            round_trip(body, "deepseek");
        }
    }

    /// An empty `tool_calls` array is distinguishable from an absent one, and
    /// has to come back as the empty array it arrived as.
    #[test]
    fn an_empty_tool_calls_array_survives() {
        round_trip(
            json!({
                "id": "chatcmpl-1",
                "choices": [{
                    "delta": {"content": "hi", "tool_calls": []},
                    "finish_reason": null,
                    "index": 0
                }],
                "created": 1_700_000_000,
                "model": "gpt-5.6",
                "object": "chat.completion.chunk"
            }),
            "openai",
        );
    }

    /// An unrecognized role keeps its exact wire spelling, so a chunk that
    /// opens on one still round-trips.
    #[test]
    fn an_unknown_role_keeps_its_spelling() {
        let body = json!({
            "id": "chatcmpl-1",
            "choices": [{
                "delta": {"role": "critic"},
                "finish_reason": null,
                "index": 0
            }],
            "created": 1_700_000_000,
            "model": "gpt-5.6",
            "object": "chat.completion.chunk"
        });
        round_trip(body.clone(), "openai");
        assert_eq!(
            ingest_chunk(parse(body)).completions[0].delta.ext["openai_compat"]["role"],
            json!("critic")
        );
    }

    #[test]
    fn ingest_populates_source() {
        let wire = parse(maximal_chunk());
        let normalized = ingest_chunk(wire.clone());
        let source = normalized.source.as_ref().unwrap();
        assert_eq!(source.protocol, Protocol::OpenAiCompat);
        assert_eq!(source.body, serde_json::to_value(&wire).unwrap());
    }

    /// The load-bearing one: whole recorded streams, event by event, rather
    /// than chunks written to suit the conversion.
    ///
    /// A hand-built fixture proves the shapes hold what its author thought to
    /// put in it. These are the transcripts `tests/protocol/` already checks
    /// the WIRE types against, so running the same bytes through normalization
    /// is what says the IR holds a real stream and not just a plausible one —
    /// including the parts nobody designs for: an opening `content: ""`, a
    /// `logprobs: null` on every chunk, tool-call arguments split mid-JSON, and
    /// OpenAI's per-chunk `obfuscation`, which no specification names.
    ///
    /// These live here rather than beside the wire-level transcript tests
    /// because `ingest_chunk` and `render_chunk` are crate-internal; the
    /// fixtures are shared with `tests/` deliberately, so a captured transcript
    /// that replaces one is checked from both ends at once.
    #[test]
    fn every_recorded_transcript_survives_normalization() {
        for name in ["openai-text", "deepseek-tool-call"] {
            let sse = std::fs::read_to_string(format!(
                "{}/tests/protocol/openai_compat/chat_completions/fixtures/transcript/{name}.sse",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            let payloads: Vec<&str> = sse
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter(|payload| *payload != "[DONE]")
                .collect();
            assert!(payloads.len() > 1, "{name} has no events");

            let provider = if name.starts_with("deepseek") {
                "deepseek"
            } else {
                "openai"
            };
            for payload in payloads {
                let body: Value = serde_json::from_str(payload).unwrap();
                let normalized = ingest_chunk(parse(body.clone()));
                assert_eq!(
                    serde_json::to_value(render_chunk(
                        &normalized,
                        provider,
                        &mut ToolCallOrdinals::default()
                    ))
                    .unwrap(),
                    body,
                    "{name} lost something on the way through the IR"
                );
            }
        }
    }

    /// ...and the IR of a real transcript is worth reading, not merely worth
    /// re-serializing: the pieces a consumer needs are typed, in order, and
    /// where a fold would look for them.
    #[test]
    fn a_recorded_tool_call_is_typed_fragment_by_fragment() {
        let sse = std::fs::read_to_string(format!(
            "{}/tests/protocol/openai_compat/chat_completions/fixtures/transcript/\
             deepseek-tool-call.sse",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let normalized: Vec<_> = sse
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|payload| *payload != "[DONE]")
            .map(|payload| ingest_chunk(serde_json::from_str(payload).unwrap()))
            .collect();

        // The reasoning that opened the stream, typed rather than left in ext.
        assert!(matches!(
            &normalized[0].completions[0].delta.content[0],
            ContentDelta::Reasoning { text, .. } if text == "the user wants today's weather"
        ));
        assert_eq!(
            normalized[0].completions[0].delta.role,
            Some(crate::gateway::types::Role::Assistant)
        );

        // Every fragment of the call, correlated by index, with the id on the
        // opener alone — concatenating the arguments yields the whole.
        let fragments: Vec<_> = normalized
            .iter()
            .flat_map(|chunk| &chunk.completions)
            .flat_map(|completion| &completion.delta.content)
            .filter_map(|delta| match delta {
                ContentDelta::ToolCall {
                    index,
                    id,
                    arguments,
                    ..
                } => Some((*index, id.clone(), arguments.clone().unwrap_or_default())),
                _ => None,
            })
            .collect();
        assert_eq!(fragments.len(), 3, "the call arrived in three pieces");
        assert!(fragments.iter().all(|(index, ..)| *index == 0));
        assert_eq!(fragments[0].1.as_deref(), Some("call_0_5f3a91c2"));
        assert!(fragments[1].1.is_none(), "the id arrives once");
        assert_eq!(
            fragments
                .iter()
                .map(|(_, _, args)| args.as_str())
                .collect::<String>(),
            r#"{"city":"Seoul"}"#
        );

        // The totals ride the last chunk, after the choice already finished.
        let last = normalized.last().unwrap();
        assert_eq!(last.usage.as_ref().unwrap().total_tokens, Some(111));
        assert_eq!(last.usage.as_ref().unwrap().cache_read_tokens, Some(64));
        assert_eq!(
            last.completions[0].finish_reason,
            Some(FinishReason::ToolCalls)
        );
    }
}
