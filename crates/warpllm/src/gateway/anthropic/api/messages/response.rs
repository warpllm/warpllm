//! Response conversions: Anthropic's [`Message`] → gateway (ingest) and back
//! (render). Round trips are lossless: fields the gateway form has no home for
//! — `type`, `stop_sequence`, the usage breakdowns — ride `ext["anthropic"]` at
//! their nesting level and are restored verbatim.
//!
//! # PROMOTE BUT RETAIN, including the whole content array
//!
//! The gateway's content blocks carry no `ext` of their own, so a KNOWN block's
//! residue has nowhere to be filed one field at a time: a text block's
//! `citations`, a source's vendor extensions, an explicit `null` where the
//! gateway holds only present-or-absent. Promotion alone loses all of it
//! precisely BECAUSE the block parsed — a shape warpllm does not recognize
//! reaches the `Unknown` arm and survives whole, while one it does recognize
//! would be rebuilt from the fields it could hold.
//!
//! So `ingest_response` retains the entire `content` array under the message's
//! ext and `render_response` prefers it. What the blocks promote to is then the
//! CROSS-protocol view, and the retained copy is the same-protocol one; the
//! tests below check each separately, because retention alone would make a
//! round trip pass however wrong the promotion was.
//!
//! # One normalization, and it is arithmetic
//!
//! `Usage::input_tokens` is the only value not carried across unchanged: this
//! protocol's excludes the cached counts and the gateway's includes them. See
//! [`ingest_usage`].
//!
//! # Two shapes this protocol does not have
//!
//! **No `created`.** Anthropic stamps no timestamp, so `ChatResponse::created`
//! is `None` and rendering back emits nothing. A caller that needs one reads
//! its own clock; inventing one here would be a number nobody measured.
//!
//! **No choices.** One request yields one reply, so `completions` always holds
//! exactly one element. That is why nothing here has a choice-level `ext`: the
//! reply IS the completion, and its residue is the response's.
//!
//! # `render_response` has no caller yet
//!
//! Nothing answers in this protocol — warpllm speaks Anthropic to a provider
//! and answers its own callers in theirs — so [`render_response`] exists as the
//! round trip's other half rather than as a code path. That is not idle
//! symmetry: a residue that is stashed but cannot be restored looks identical
//! to a lossless ingest from the gateway side alone, and only rendering back
//! tells the two apart. The tests below are what it is for.

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::gateway::anthropic::{
    cache_control, cache_hint, merged_ext, namespaced, role_from_wire, role_to_wire,
};
use crate::gateway::types::{
    self, ContentBlock, FinishReason, IngestSource, MediaSource, RawJson, ReasoningDetail,
};
use crate::protocol::UnknownFields;
use crate::protocol::anthropic::messages::types::{
    Base64Source, CacheControl, ContentBlock as WireBlock, DocumentBlock, FileSource, ImageBlock,
    Message, OutputTokensDetails, RedactedThinkingBlock, Source, TextBlock, ThinkingBlock,
    ToolResultBlock, ToolResultContent, ToolUseBlock, UrlSource, Usage,
};
use crate::types::Protocol;

/// Permissive and infallible; the exhaustive destructures at every level make
/// dropping a newly-typed wire field a compile error.
pub(crate) fn ingest_response(response: Message) -> types::ChatResponse {
    // Wire structs are plain serde data; serialization cannot fail.
    let body = plain(&response);
    let Message {
        id,
        r#type,
        role,
        content,
        model,
        stop_reason,
        stop_sequence,
        usage,
        unknown_fields,
    } = response;
    let mut anthropic = UnknownFields::new();
    anthropic.insert("type".into(), Value::String(r#type));
    // Required AND nullable upstream, so it is stashed either way — as a string
    // or as the null it arrived as — and always emitted again.
    anthropic.insert(
        "stop_sequence".into(),
        stop_sequence.map_or(Value::Null, Value::String),
    );
    anthropic.extend(unknown_fields);

    let (role, raw_role) = role_from_wire(role);
    let mut message_ext = UnknownFields::new();
    if let Some(raw) = raw_role {
        message_ext.insert("role".into(), Value::String(raw));
    }
    // PROMOTE BUT RETAIN, and here it is the ONLY way a reply round trips.
    //
    // The gateway's content blocks have no `ext` of their own, so anything a
    // known block carries that they cannot hold has nowhere else to go: a text
    // block's `citations`, a source's vendor fields, the difference between an
    // absent `cache_control` and an explicit null. Promoting alone would let
    // `render_response` rebuild the block with an empty residue and silently
    // drop documented metadata, which is exactly what the losslessness claim
    // above forbids. Retaining the whole array is what makes the claim true.
    message_ext.insert("content".into(), plain(&content));
    types::ChatResponse {
        id,
        model,
        // See this module's docs: there is no timestamp to report.
        created: None,
        completions: vec![types::Completion {
            message: types::Message {
                role,
                content: content.iter().map(ingest_block).collect(),
                ext: namespaced(message_ext),
            },
            finish_reason: finish_reason(stop_reason.as_deref()),
            // A null `stop_reason` means the reply is still being generated,
            // which a whole reply never is — so the empty string here is
            // unreachable from a completed exchange, and rendering reads it
            // back as the null it was. Anthropic sends no empty stop reasons.
            finish_reason_raw: stop_reason.unwrap_or_default(),
            ext: types::ProviderExt::new(),
        }],
        usage: Some(ingest_usage(usage)),
        ext: namespaced(anthropic),
        source: Some(IngestSource {
            protocol: Protocol::Anthropic,
            body,
        }),
    }
}

/// Anthropic's `stop_reason` → the gateway's programmatic enum.
///
/// Its own table rather than [`FinishReason::from_raw`], which reads OpenAI's
/// spellings and shares not one of these. `finish_reason_raw` stays
/// authoritative on render, so this only has to be right for callers matching
/// on the enum.
fn finish_reason(stop_reason: Option<&str>) -> FinishReason {
    match stop_reason {
        Some("end_turn" | "stop_sequence") => FinishReason::Stop,
        // Both ran out of room: one hit the caller's ceiling, the other the
        // model's window. A caller does the same thing about either — send
        // less — which is what this enum is for.
        Some("max_tokens" | "model_context_window_exceeded") => FinishReason::Length,
        Some("tool_use") => FinishReason::ToolCalls,
        Some("refusal") => FinishReason::ContentFilter,
        // `pause_turn` is deliberately NOT `Stop`: the turn was interrupted and
        // is meant to be continued, which is not the claim `Stop` makes. A
        // caller that treats it as a finished answer stops a reply mid-thought.
        _ => FinishReason::Other,
    }
}

/// Anthropic's blocks → the gateway's.
///
/// A block this protocol has and the gateway model does not — a text source, a
/// server tool's result, whatever ships next — becomes `Unknown` carrying the
/// WHOLE wire block rather than a lossy approximation of it. That is what makes
/// a same-protocol round trip exact for shapes warpllm does not understand.
pub(super) fn ingest_block(block: &WireBlock) -> ContentBlock {
    match block {
        WireBlock::Text(text) => ContentBlock::Text {
            text: text.text.clone(),
            cache: hint_of(&text.cache_control),
        },
        WireBlock::Image(image) => match media_source(&image.source) {
            Some(source) => ContentBlock::Image {
                source,
                // Anthropic has no resolution hint on an image block, so this
                // is only ever filled by an ingest from a protocol that does.
                detail: None,
                cache: hint_of(&image.cache_control),
            },
            None => ContentBlock::Unknown(plain(block)),
        },
        WireBlock::Document(document) => match media_source(&document.source) {
            Some(source) => ContentBlock::Document {
                source,
                title: document.title.clone().flatten(),
                cache: hint_of(&document.cache_control),
            },
            None => ContentBlock::Unknown(plain(block)),
        },
        WireBlock::ToolUse(call) => ContentBlock::ToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            // An OBJECT here, a string of JSON on chat completions. Serializing
            // once at the boundary is what `RawJson::from_value` is for, and
            // `preserve_order` keeps Anthropic's key order — so the text is
            // exact modulo whitespace, which is the one documented seam in the
            // tool path.
            arguments: RawJson::from_value(&call.input),
        },
        WireBlock::ToolResult(result) => ContentBlock::ToolResult {
            call_id: result.tool_use_id.clone(),
            content: match &result.content {
                Some(ToolResultContent::Text(text)) => vec![ContentBlock::Text {
                    text: text.clone(),
                    cache: None,
                }],
                Some(ToolResultContent::Blocks(blocks)) => {
                    blocks.iter().map(ingest_block).collect()
                }
                None => Vec::new(),
            },
            is_error: result.is_error.unwrap_or(false),
        },
        WireBlock::Thinking(thinking) => ContentBlock::Reasoning {
            detail: ReasoningDetail::Text {
                text: thinking.thinking.clone(),
                signature: thinking.signature.clone(),
            },
            provenance: Some(Protocol::Anthropic.as_str().to_string()),
            id: None,
        },
        WireBlock::RedactedThinking(redacted) => ContentBlock::Reasoning {
            detail: ReasoningDetail::Encrypted {
                data: redacted.data.clone(),
            },
            provenance: Some(Protocol::Anthropic.as_str().to_string()),
            id: None,
        },
        WireBlock::Unknown(value) => ContentBlock::Unknown(value.clone()),
    }
}

/// `Some` for the three sources the gateway model names; `None` for a source it
/// does not, which makes the whole block `Unknown`.
///
/// Anthropic's `text` source — a document's contents inline — is the one with
/// no counterpart: `MediaSource` is a URL, base64 bytes, or a provider file id,
/// and none of those is "here is the text". Squeezing it into `Base64` would
/// claim the payload was encoded when it was not.
fn media_source(source: &Source) -> Option<MediaSource> {
    match source {
        Source::Base64(base64) => Some(MediaSource::Base64 {
            media_type: base64.media_type.clone(),
            data: base64.data.clone(),
        }),
        Source::Url(url) => Some(MediaSource::Url {
            url: url.url.clone(),
        }),
        Source::File(file) => Some(MediaSource::ProviderFile {
            id: file.file_id.clone(),
        }),
        Source::Text(_) | Source::Unknown(_) => None,
    }
}

/// The inverse of [`media_source`]. Fallible in the other direction for the
/// mirror reason: nothing in this protocol carries a media type for a provider
/// file, and a bare URL is not something every block accepts — but both of
/// those the wire DOES have shapes for, so only the media type is guessed at,
/// never the kind.
pub(super) fn render_source(source: &MediaSource) -> Source {
    match source {
        MediaSource::Base64 { media_type, data } => Source::Base64(Base64Source {
            media_type: media_type.clone(),
            data: data.clone(),
            unknown_fields: UnknownFields::new(),
        }),
        MediaSource::Url { url } => Source::Url(UrlSource {
            url: url.clone(),
            unknown_fields: UnknownFields::new(),
        }),
        MediaSource::ProviderFile { id } => Source::File(FileSource {
            file_id: id.clone(),
            unknown_fields: UnknownFields::new(),
        }),
    }
}

/// A wire block's three-state `cache_control` as the gateway's plain
/// `Option`. An explicit `null` and an absent key both mean no breakpoint;
/// which one it was rides the message's retained `content`.
fn hint_of(control: &Option<Option<CacheControl>>) -> Option<types::CacheHint> {
    control.as_ref().and_then(Option::as_ref).map(cache_hint)
}

pub(super) fn ingest_usage(usage: Usage) -> types::Usage {
    let Usage {
        input_tokens,
        output_tokens,
        cache_creation_input_tokens,
        cache_read_input_tokens,
        output_tokens_details,
        unknown_fields,
    } = usage;
    let mut anthropic = UnknownFields::new();
    let cache_write_tokens = lift_count(
        &mut anthropic,
        "cache_creation_input_tokens",
        cache_creation_input_tokens,
    );
    let cache_read_tokens = lift_count(
        &mut anthropic,
        "cache_read_input_tokens",
        cache_read_input_tokens,
    );
    let mut reasoning_tokens = None;
    if let Some(details) = output_tokens_details {
        let residue = match details {
            Some(details) => {
                let mut residue = object(plain(&details));
                // Only a NUMBER is lifted. An explicit `null` stays in the
                // residue, because the gateway's `Option<u64>` cannot tell one
                // from an absent key and rendering would turn it into one.
                if residue.get("thinking_tokens").is_some_and(Value::is_number) {
                    reasoning_tokens = residue.remove("thinking_tokens").and_then(|v| v.as_u64());
                }
                Value::Object(residue)
            }
            None => Value::Null,
        };
        anthropic.insert("output_tokens_details".into(), residue);
    }
    anthropic.extend(unknown_fields);
    // NORMALIZED, and this is the one arithmetic decision in the module.
    //
    // Anthropic's `input_tokens` counts only the tokens after the last cache
    // breakpoint — "not read from or used to create a cache" — where the
    // gateway's means the WHOLE input with the cached parts included, which is
    // what `Usage`'s own doc requires and what OpenAI's `prompt_tokens` already
    // is. Storing Anthropic's number unchanged would make one conversation
    // report a different prompt size depending on which backend served it, and
    // would let `cached_tokens` come out LARGER than the prompt it is part of.
    let cached = cache_read_tokens.unwrap_or(0) + cache_write_tokens.unwrap_or(0);
    let input_total = u64::from(input_tokens) + cached;
    // PROMOTE BUT RETAIN, so the normalization is reversible: the wire value
    // rides the residue and `render_usage` prefers it over subtracting back.
    if cached > 0 {
        anthropic.insert("input_tokens".into(), Value::from(input_tokens));
    }
    types::Usage {
        input_tokens: Some(input_total),
        output_tokens: Some(u64::from(output_tokens)),
        // COMPUTED, because Anthropic sends no total. Input plus output and
        // nothing else — the cache counts are already inside `input_total`, so
        // adding them here would double-count every cached request.
        total_tokens: Some(input_total + u64::from(output_tokens)),
        // Thinking tokens are likewise a BREAKDOWN of `output_tokens`, already
        // counted in it.
        reasoning_tokens,
        cache_read_tokens,
        cache_write_tokens,
        ext: namespaced(anthropic),
    }
}

/// Lifts an optional-and-nullable count into the gateway's plain `Option<u64>`,
/// stashing an explicit `null` so rendering can restore one.
///
/// Three wire states do not fit in the gateway's two. Collapsing them would
/// turn `"cache_read_input_tokens": null` — which a cache-aware reply really
/// does send — into an absent key on the way back out.
fn lift_count(residue: &mut UnknownFields, key: &str, value: Option<Option<u32>>) -> Option<u64> {
    match value {
        Some(Some(count)) => Some(u64::from(count)),
        Some(None) => {
            residue.insert(key.to_string(), Value::Null);
            None
        }
        None => None,
    }
}

/// The inverse of [`lift_count`]. A stashed `null` is restored only when the
/// gateway field is empty; a count that is set is authoritative over it.
fn restore_count(
    residue: &mut UnknownFields,
    key: &str,
    count: Option<u64>,
) -> Option<Option<u32>> {
    match (count, residue.remove(key)) {
        (Some(count), _) => Some(Some(count as u32)),
        (None, Some(Value::Null)) => Some(None),
        (None, _) => None,
    }
}

/// Infallible: wire fields restore from ext, and anything corrupted past its
/// wire type falls back to dropping that field rather than failing a render.
pub(crate) fn render_response(response: &types::ChatResponse, provider: &str) -> Message {
    let mut unknown_fields = merged_ext(&response.ext, provider);
    // One reply, one completion — see this module's docs. A response carrying
    // several could only have come from a protocol with choices, and Anthropic
    // has no shape for the others; the first is the answer, and dropping the
    // rest is the whole of what "no multi-choice concept" costs.
    let completion = response.completions.first();
    let mut message_fields = completion
        .map(|completion| merged_ext(&completion.message.ext, provider))
        .unwrap_or_default();
    Message {
        id: response.id.clone(),
        r#type: take_string(&mut unknown_fields, "type").unwrap_or_else(|| "message".into()),
        role: match message_fields.remove("role") {
            Some(Value::String(raw)) => raw,
            _ => completion
                .map_or("assistant", |completion| {
                    role_to_wire(completion.message.role)
                })
                .to_string(),
        },
        // What arrived wins over what the blocks can be rebuilt into; see
        // `ingest_response`. A reply from ANOTHER protocol has no residue here
        // and is rebuilt, which is the only path `render_blocks` serves.
        content: take_typed(&mut message_fields, "content").unwrap_or_else(|| {
            completion
                .map(|completion| render_blocks(&completion.message.content))
                .unwrap_or_default()
        }),
        model: response.model.clone(),
        stop_reason: completion
            .map(|completion| completion.finish_reason_raw.clone())
            .filter(|raw| !raw.is_empty()),
        stop_sequence: match unknown_fields.remove("stop_sequence") {
            Some(Value::String(sequence)) => Some(sequence),
            _ => None,
        },
        usage: response
            .usage
            .as_ref()
            .map_or_else(zero_usage, |usage| render_usage(usage, provider)),
        unknown_fields,
    }
}

/// Blocks that have no wire shape are DROPPED rather than approximated. Only
/// two can be: reasoning that did not come from Anthropic (see
/// [`render_reasoning`]) and audio, which `request::ensure_renderable` refuses
/// before anything reaches here.
fn render_blocks(blocks: &[ContentBlock]) -> Vec<WireBlock> {
    blocks.iter().filter_map(render_block).collect()
}

/// The inverse of [`ingest_block`] for the blocks a REPLY carries.
///
/// Deliberately not shared with the request renderer: a request's blocks can
/// fail to render (a tool call whose arguments will not parse) and a reply's
/// cannot, because everything in a reply came off this protocol's own wire.
fn render_block(block: &ContentBlock) -> Option<WireBlock> {
    match block {
        ContentBlock::Text { text, cache } => Some(WireBlock::Text(TextBlock {
            text: text.clone(),
            cache_control: control_of(cache),
            unknown_fields: UnknownFields::new(),
        })),
        ContentBlock::Image { source, cache, .. } => Some(WireBlock::Image(ImageBlock {
            source: render_source(source),
            cache_control: control_of(cache),
            unknown_fields: UnknownFields::new(),
        })),
        ContentBlock::Document {
            source,
            title,
            cache,
        } => Some(WireBlock::Document(DocumentBlock {
            source: render_source(source),
            title: title.clone().map(Some),
            cache_control: control_of(cache),
            unknown_fields: UnknownFields::new(),
        })),
        ContentBlock::ToolCall {
            id,
            name,
            arguments,
        } => Some(WireBlock::ToolUse(ToolUseBlock {
            id: id.clone(),
            name: name.clone(),
            // A reply's arguments came from this protocol as an object, so
            // they parse. A REQUEST's may not, which is why that renderer
            // reports the failure instead of falling back — here the fallback
            // is unreachable and an empty object is the inert value.
            input: arguments.parse().unwrap_or(Value::Null),
            cache_control: None,
            unknown_fields: UnknownFields::new(),
        })),
        ContentBlock::ToolResult {
            call_id,
            content,
            is_error,
        } => Some(WireBlock::ToolResult(ToolResultBlock {
            tool_use_id: call_id.clone(),
            content: Some(render_result_content(content)),
            is_error: is_error.then_some(true),
            cache_control: None,
            unknown_fields: UnknownFields::new(),
        })),
        ContentBlock::Reasoning {
            detail, provenance, ..
        } => render_reasoning(detail, provenance.as_deref()),
        ContentBlock::Unknown(value) => Some(WireBlock::Unknown(value.clone())),
        // No audio block exists on this protocol. Reaching here would mean the
        // request gate let one through; a reply cannot contain one.
        ContentBlock::Audio { .. } => None,
    }
}

/// Reasoning back onto the wire, but only reasoning THIS protocol produced.
///
/// `provenance` is what makes that decision possible, and it is what the
/// field's own doc says it is for: "renderers can reconstitute the native shape
/// (or know they can't)". A thinking block carries a cryptographic signature
/// Anthropic verifies, and one produced elsewhere has none that would pass — so
/// forwarding another provider's reasoning would not preserve it, it would
/// reject the turn. A summary has no Anthropic shape at all.
///
/// Dropping is safe in a way that forwarding is not: Anthropic requires a run
/// of its OWN thinking blocks to come back untouched, and this returns exactly
/// those unchanged.
pub(super) fn render_reasoning(
    detail: &ReasoningDetail,
    provenance: Option<&str>,
) -> Option<WireBlock> {
    if provenance != Some(Protocol::Anthropic.as_str()) {
        return None;
    }
    match detail {
        ReasoningDetail::Text { text, signature } => Some(WireBlock::Thinking(ThinkingBlock {
            thinking: text.clone(),
            signature: signature.clone(),
            unknown_fields: UnknownFields::new(),
        })),
        ReasoningDetail::Encrypted { data } => {
            Some(WireBlock::RedactedThinking(RedactedThinkingBlock {
                data: data.clone(),
                unknown_fields: UnknownFields::new(),
            }))
        }
        // Anthropic never produces one, so this is unreachable under the
        // provenance check above and stays a drop rather than a guess.
        ReasoningDetail::Summary { .. } => None,
    }
}

/// A tool result's payload. A lone text block renders as the bare string form,
/// which is what a result almost always is and what Anthropic's own examples
/// show; anything else renders as blocks.
pub(super) fn render_result_content(blocks: &[ContentBlock]) -> ToolResultContent {
    match blocks {
        [ContentBlock::Text { text, cache: None }] => ToolResultContent::Text(text.clone()),
        _ => ToolResultContent::Blocks(render_blocks(blocks)),
    }
}

/// The inverse of [`hint_of`]. A hint always renders as a present breakpoint,
/// never as an explicit `null` — the null form only ever comes back through a
/// message's retained content.
pub(super) fn control_of(hint: &Option<types::CacheHint>) -> Option<Option<CacheControl>> {
    hint.as_ref().map(|hint| Some(cache_control(hint)))
}

fn zero_usage() -> Usage {
    Usage {
        input_tokens: 0,
        output_tokens: 0,
        cache_creation_input_tokens: None,
        cache_read_input_tokens: None,
        output_tokens_details: None,
        unknown_fields: UnknownFields::new(),
    }
}

fn render_usage(usage: &types::Usage, provider: &str) -> Usage {
    let mut unknown_fields = merged_ext(&usage.ext, provider);
    let details = match unknown_fields.remove("output_tokens_details") {
        Some(Value::Object(mut residue)) => {
            if let Some(tokens) = usage.reasoning_tokens {
                residue.insert("thinking_tokens".into(), Value::from(tokens));
            }
            Some(Some(from_fields(residue)))
        }
        Some(Value::Null) => Some(None),
        // No residue: the breakdown is emitted only if there is a count to put
        // in it, so a reply that reported none does not grow an empty object.
        _ => usage.reasoning_tokens.map(|tokens| {
            Some(OutputTokensDetails {
                thinking_tokens: Some(Some(tokens as u32)),
                unknown_fields: UnknownFields::new(),
            })
        }),
    };
    Usage {
        // The inverse of `ingest_usage`'s normalization: the gateway count
        // INCLUDES the cached tokens and this protocol's excludes them. The
        // retained wire value wins where there is one; subtracting is the
        // fallback for a reply that arrived on another protocol, and saturates
        // rather than wrapping because nothing guarantees a hand-built `Usage`
        // keeps the parts smaller than the whole.
        input_tokens: match unknown_fields
            .remove("input_tokens")
            .and_then(|v| v.as_u64())
        {
            Some(retained) => retained as u32,
            None => usage
                .input_tokens
                .unwrap_or(0)
                .saturating_sub(usage.cache_read_tokens.unwrap_or(0))
                .saturating_sub(usage.cache_write_tokens.unwrap_or(0)) as u32,
        },
        output_tokens: usage.output_tokens.unwrap_or(0) as u32,
        cache_creation_input_tokens: restore_count(
            &mut unknown_fields,
            "cache_creation_input_tokens",
            usage.cache_write_tokens,
        ),
        cache_read_input_tokens: restore_count(
            &mut unknown_fields,
            "cache_read_input_tokens",
            usage.cache_read_tokens,
        ),
        output_tokens_details: details,
        unknown_fields,
    }
}

fn from_fields<T: DeserializeOwned + Default>(fields: UnknownFields) -> T {
    serde_json::from_value(Value::Object(fields)).unwrap_or_default()
}

pub(super) fn take_string(fields: &mut UnknownFields, key: &str) -> Option<String> {
    match fields.remove(key) {
        Some(Value::String(value)) => Some(value),
        _ => None,
    }
}

pub(super) fn take_typed<T: DeserializeOwned>(fields: &mut UnknownFields, key: &str) -> Option<T> {
    fields
        .remove(key)
        .and_then(|value| serde_json::from_value(value).ok())
}

/// Wire structs are plain serde data; serialization cannot fail.
pub(super) fn plain<T: serde::Serialize>(value: &T) -> Value {
    serde_json::to_value(value).expect("wire data serializes")
}

pub(super) fn object(value: Value) -> UnknownFields {
    match value {
        Value::Object(fields) => fields,
        _ => UnknownFields::new(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn parse(body: Value) -> Message {
        serde_json::from_value(body).unwrap()
    }

    fn round_trip(body: Value) -> Value {
        plain(&render_response(&ingest_response(parse(body)), "anthropic"))
    }

    /// Every documented field, unknown fields at every nesting level, and one
    /// block of each kind — including two the gateway model has no shape for.
    fn maximal_body() -> Value {
        json!({
            "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [
                {"type": "thinking", "thinking": "Let me count.", "signature": "sig_abc"},
                {"type": "redacted_thinking", "data": "EncryptedBase64=="},
                {"type": "text", "text": "There are 3."},
                {"type": "tool_use", "id": "toolu_01", "name": "counter", "input": {"z": 1, "a": 2}},
                {"type": "server_tool_use", "id": "srvtoolu_01", "name": "web_search"}
            ],
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 2095,
                "output_tokens": 503,
                "cache_creation_input_tokens": 2051,
                "cache_read_input_tokens": null,
                "output_tokens_details": {"thinking_tokens": 120},
                "service_tier": "standard",
                "cache_creation": {"ephemeral_5m_input_tokens": 2051}
            },
            "container": null
        })
    }

    #[test]
    fn a_maximal_reply_round_trips_byte_for_byte() {
        let body = maximal_body();
        assert_eq!(round_trip(body.clone()), body);
    }

    /// The gateway view, so a change to the mapping fails here rather than only
    /// inside a round trip that would still pass if BOTH halves changed.
    #[test]
    fn the_gateway_view_of_a_maximal_reply() {
        let response = ingest_response(parse(maximal_body()));
        assert_eq!(response.id, "msg_013Zva2CMHLNnXjNJJKqJ2EF");
        assert!(response.created.is_none(), "Anthropic stamps no timestamp");
        assert_eq!(response.completions.len(), 1, "one reply, one completion");

        let completion = &response.completions[0];
        assert_eq!(completion.finish_reason, FinishReason::ToolCalls);
        assert_eq!(completion.finish_reason_raw, "tool_use");
        assert_eq!(completion.message.role, types::Role::Assistant);

        let blocks = &completion.message.content;
        assert!(matches!(
            &blocks[0],
            ContentBlock::Reasoning {
                detail: ReasoningDetail::Text { signature: Some(sig), .. },
                provenance: Some(p),
                ..
            } if sig == "sig_abc" && p == "anthropic"
        ));
        assert!(matches!(
            &blocks[1],
            ContentBlock::Reasoning {
                detail: ReasoningDetail::Encrypted { data },
                ..
            } if data == "EncryptedBase64=="
        ));
        assert!(matches!(&blocks[2], ContentBlock::Text { text, .. } if text == "There are 3."));
        // Key order survives the object → text conversion (preserve_order).
        assert!(matches!(
            &blocks[3],
            ContentBlock::ToolCall { id, arguments, .. }
                if id == "toolu_01" && arguments.as_str() == r#"{"z":1,"a":2}"#
        ));
        // A block warpllm does not model stays whole rather than being
        // approximated as something it is not.
        assert!(
            matches!(&blocks[4], ContentBlock::Unknown(value) if value["type"] == "server_tool_use")
        );
    }

    /// The arithmetic, spelled out. Anthropic's `input_tokens` counts only what
    /// came after the last cache breakpoint; the gateway's counts the WHOLE
    /// input with the cached parts inside it, so ingest adds them back. Storing
    /// the wire number unchanged would make the cached counts look like a
    /// separate pool rather than a breakdown.
    #[test]
    fn the_input_count_is_normalized_to_include_the_cached_tokens() {
        let usage = ingest_usage(
            serde_json::from_value(json!({
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_creation_input_tokens": 300,
                "cache_read_input_tokens": 4000,
                "output_tokens_details": {"thinking_tokens": 7}
            }))
            .unwrap(),
        );
        assert_eq!(usage.input_tokens, Some(4310), "10 + 300 + 4000");
        assert_eq!(usage.output_tokens, Some(20));
        assert_eq!(usage.cache_write_tokens, Some(300));
        assert_eq!(usage.cache_read_tokens, Some(4000));
        // Input plus output and nothing else: the cache counts are already
        // inside the input, so adding them again would double-count them.
        assert_eq!(usage.total_tokens, Some(4330));
        // Likewise a breakdown of `output_tokens`, already inside it.
        assert_eq!(usage.reasoning_tokens, Some(7));
    }

    /// THE reason the normalization exists, checked where it is observable: at
    /// the OpenAI surface a caller actually reads.
    ///
    /// Without it a cached reply comes out self-contradicting — `cached_tokens`
    /// larger than the `prompt_tokens` it is a subset of, and a `total_tokens`
    /// that is not `prompt + completion` — and every one of those numbers feeds
    /// cost accounting that has no way to notice.
    #[test]
    fn a_cached_reply_reports_consistent_openai_usage() {
        let reply = ingest_response(parse(json!({
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": "claude-opus-5", "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {
                "input_tokens": 21,
                "output_tokens": 16,
                "cache_creation_input_tokens": 1024,
                "cache_read_input_tokens": 8192
            }
        })));
        let openai = plain(
            &crate::gateway::openai_compat::api::chat_completions::render_response(
                &reply,
                "anthropic",
            ),
        );
        let usage = &openai["usage"];
        assert_eq!(usage["prompt_tokens"], json!(9237), "21 + 1024 + 8192");
        assert_eq!(usage["completion_tokens"], json!(16));
        assert_eq!(
            usage["total_tokens"],
            json!(9253),
            "OpenAI's total is prompt + completion"
        );
        assert_eq!(
            usage["total_tokens"].as_u64().unwrap(),
            usage["prompt_tokens"].as_u64().unwrap() + usage["completion_tokens"].as_u64().unwrap()
        );
        // The cached count is a SUBSET of the prompt, so it cannot exceed it.
        let cached = usage["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap();
        assert_eq!(cached, 8192);
        assert!(cached <= usage["prompt_tokens"].as_u64().unwrap());
    }

    /// ...and the normalization is reversible, so an Anthropic round trip still
    /// hands back the exclusive number the protocol actually uses.
    #[test]
    fn the_wire_input_count_survives_the_normalization() {
        let body = json!({
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": "claude-opus-5", "content": [],
            "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {
                "input_tokens": 21, "output_tokens": 16,
                "cache_creation_input_tokens": 1024,
                "cache_read_input_tokens": 8192
            }
        });
        assert_eq!(round_trip(body.clone()), body);
    }

    /// An explicit `null` count is not an absent one, and the gateway's plain
    /// `Option` cannot hold the difference — so it rides the residue. Without
    /// this the null comes back as a missing key.
    #[test]
    fn an_explicit_null_count_survives_the_round_trip() {
        let body = json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5",
            "content": [], "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {
                "input_tokens": 1, "output_tokens": 2,
                "cache_creation_input_tokens": null,
                "cache_read_input_tokens": null,
                "output_tokens_details": {"thinking_tokens": null}
            }
        });
        assert_eq!(round_trip(body.clone()), body);
    }

    /// A reply that reports no breakdown must not grow an empty one.
    #[test]
    fn an_absent_breakdown_stays_absent() {
        let body = json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 2}
        });
        assert_eq!(round_trip(body.clone()), body);
    }

    /// Every row of the table, including the two the plan for this work did not
    /// have and the one that must NOT read as a finished answer.
    #[test]
    fn the_stop_reason_table() {
        let rows = [
            ("end_turn", FinishReason::Stop),
            ("stop_sequence", FinishReason::Stop),
            ("max_tokens", FinishReason::Length),
            ("model_context_window_exceeded", FinishReason::Length),
            ("tool_use", FinishReason::ToolCalls),
            ("refusal", FinishReason::ContentFilter),
            ("pause_turn", FinishReason::Other),
            ("something_new", FinishReason::Other),
        ];
        for (raw, expected) in rows {
            assert_eq!(finish_reason(Some(raw)), expected, "{raw}");
        }
        assert_eq!(finish_reason(None), FinishReason::Other);
    }

    /// `pause_turn` on its own, because the cost of getting it wrong is not a
    /// mislabelled enum: a caller that reads it as `Stop` presents an
    /// interrupted turn as the finished answer and never continues it.
    #[test]
    fn a_paused_turn_is_not_a_stop() {
        assert_ne!(finish_reason(Some("pause_turn")), FinishReason::Stop);
    }

    /// A `stop_reason` Anthropic adds tomorrow keeps its exact spelling, which
    /// is what `finish_reason_raw` is for — the enum is for matching, the
    /// string is authoritative.
    #[test]
    fn an_unknown_stop_reason_keeps_its_spelling() {
        let body = json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5",
            "content": [], "stop_reason": "model_context_window_exceeded", "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 2}
        });
        let response = ingest_response(parse(body.clone()));
        assert_eq!(
            response.completions[0].finish_reason_raw,
            "model_context_window_exceeded"
        );
        assert_eq!(round_trip(body.clone()), body);
    }

    /// Reasoning from ANOTHER provider must not be forwarded: it carries no
    /// signature Anthropic would accept, so sending it rejects the whole turn
    /// rather than preserving anything.
    #[test]
    fn only_anthropics_own_reasoning_renders_back() {
        let detail = ReasoningDetail::Text {
            text: "thinking".into(),
            signature: Some("sig".into()),
        };
        assert!(render_reasoning(&detail, Some("anthropic")).is_some());
        assert!(render_reasoning(&detail, Some("openai_compat")).is_none());
        assert!(render_reasoning(&detail, None).is_none());
        assert!(
            render_reasoning(
                &ReasoningDetail::Summary {
                    summary: "s".into()
                },
                Some("anthropic")
            )
            .is_none()
        );
    }

    /// A KNOWN block carrying residue the gateway blocks cannot hold. This is
    /// the case promotion alone loses: the block parses, so it never reaches
    /// the `Unknown` arm that would have kept it whole, and the gateway's
    /// `ContentBlock` has no `ext` to put a citation or a vendor field in.
    /// Retaining the array is what holds it — and the explicit `null` on a
    /// three-state field is the same loss in miniature.
    #[test]
    fn residue_on_a_known_block_survives_the_round_trip() {
        let body = json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5",
            "content": [
                {
                    "type": "text",
                    "text": "The grass is green.",
                    "citations": [{
                        "type": "char_location",
                        "cited_text": "The grass is green.",
                        "document_index": 0,
                        "start_char_index": 0,
                        "end_char_index": 19
                    }]
                },
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://e.com/a.png", "vendor_hint": 1},
                    "cache_control": null
                },
                {"type": "text", "text": "and more", "cache_control": null}
            ],
            "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 2}
        });
        assert_eq!(round_trip(body.clone()), body);
    }

    /// The other half of the retention design, and the reason the test above is
    /// not the only one that matters: a reply with NO residue — one built by
    /// hand, or ingested from another protocol — must still render its blocks
    /// from the gateway form. Without this, `render_blocks` could break
    /// entirely and every round-trip test would still pass, because retention
    /// answers before it is ever called.
    #[test]
    fn a_reply_with_no_retained_content_still_renders_its_blocks() {
        let response = types::ChatResponse {
            id: "msg_1".into(),
            model: "claude-opus-5".into(),
            created: None,
            completions: vec![types::Completion {
                message: types::Message {
                    role: types::Role::Assistant,
                    content: vec![
                        ContentBlock::Text {
                            text: "hi".into(),
                            cache: None,
                        },
                        ContentBlock::ToolCall {
                            id: "toolu_1".into(),
                            name: "counter".into(),
                            arguments: RawJson::new(r#"{"z":1}"#),
                        },
                    ],
                    ext: types::ProviderExt::new(),
                },
                finish_reason: FinishReason::ToolCalls,
                finish_reason_raw: "tool_use".into(),
                ext: types::ProviderExt::new(),
            }],
            usage: None,
            ext: types::ProviderExt::new(),
            source: None,
        };
        assert_eq!(
            plain(&render_response(&response, "anthropic"))["content"],
            json!([
                {"type": "text", "text": "hi"},
                {"type": "tool_use", "id": "toolu_1", "name": "counter", "input": {"z": 1}}
            ])
        );
    }

    /// A source shape the gateway model cannot name keeps its whole wire block
    /// rather than being flattened into something it is not.
    #[test]
    fn a_text_source_document_stays_whole() {
        let body = json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5",
            "content": [{
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": "grass is green"},
                "title": "notes"
            }],
            "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 2}
        });
        let response = ingest_response(parse(body.clone()));
        assert!(matches!(
            &response.completions[0].message.content[0],
            ContentBlock::Unknown(value) if value["source"]["type"] == "text"
        ));
        assert_eq!(round_trip(body.clone()), body);
    }
}
