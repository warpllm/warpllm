//! Response conversions: OpenAI-compatible wire → gateway (ingest) and
//! gateway → wire (render). Round trips are lossless with zero
//! permitted transformations: protocol-specific fields (`object`,
//! `service_tier`, choice `index`, `refusal`, …) ride
//! `ext["openai_compat"]` at their nesting level and are restored
//! verbatim.

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::gateway::types::{
    self, ContentBlock, FinishReason, IngestSource, RawJson, ReasoningDetail,
};
use crate::protocol::openai_compat::chat_completions::types::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCallUnion,
    ChatCompletionResponseMessage, Choice, CompletionUsage, CreateChatCompletionResponse, Function,
    UnknownFields,
};
use crate::types::Protocol;

use crate::gateway::openai_compat::{merged_ext, namespaced, role_from_wire, role_to_wire};

/// Permissive and infallible; the exhaustive destructures at every level
/// make dropping a newly-typed wire field a compile error.
pub(crate) fn ingest_response(response: CreateChatCompletionResponse) -> types::ChatResponse {
    // Wire structs are plain serde data; serialization cannot fail.
    let body = serde_json::to_value(&response).expect("wire response serializes");
    let CreateChatCompletionResponse {
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
    } = response;
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
    compat.extend(unknown_fields);
    types::ChatResponse {
        id,
        model,
        created: Some(created),
        completions: choices.into_iter().map(ingest_choice).collect(),
        usage: usage.map(ingest_usage),
        ext: namespaced(compat),
        source: Some(IngestSource {
            protocol: Protocol::OpenAiCompat,
            body,
        }),
    }
}

fn ingest_choice(choice: Choice) -> types::Completion {
    let Choice {
        finish_reason,
        index,
        logprobs,
        message,
        unknown_fields,
    } = choice;
    let mut compat = UnknownFields::new();
    compat.insert("index".into(), Value::from(index));
    if let Some(logprobs) = logprobs {
        compat.insert("logprobs".into(), plain(&logprobs));
    }
    compat.extend(unknown_fields);
    types::Completion {
        message: ingest_message(message),
        finish_reason: FinishReason::from_raw(&finish_reason),
        finish_reason_raw: finish_reason,
        ext: namespaced(compat),
    }
}

fn ingest_message(message: ChatCompletionResponseMessage) -> types::Message {
    let ChatCompletionResponseMessage {
        content,
        refusal,
        role,
        annotations,
        audio,
        function_call,
        tool_calls,
        unknown_fields,
    } = message;
    let (role, raw_role) = role_from_wire(role);
    let mut compat = UnknownFields::new();
    if let Some(raw) = raw_role {
        compat.insert("role".into(), Value::String(raw));
    }
    if let Some(refusal) = refusal {
        compat.insert("refusal".into(), Value::String(refusal));
    }
    if let Some(annotations) = annotations {
        compat.insert("annotations".into(), plain(&annotations));
    }
    if let Some(audio) = audio {
        compat.insert("audio".into(), plain(&audio));
    }
    if let Some(function_call) = function_call {
        compat.insert("function_call".into(), plain(&function_call));
    }
    // Reasoning first, since it precedes the answer it led to.
    let mut blocks: Vec<ContentBlock> = reasoning_block(&unknown_fields)
        .into_iter()
        .chain(content.map(|text| ContentBlock::Text { text, cache: None }))
        .collect();
    match tool_calls {
        // `Some([])` is distinguishable from absent; stash it so render
        // re-emits the empty array byte-for-byte.
        Some(calls) if calls.is_empty() => {
            compat.insert("tool_calls".into(), Value::Array(Vec::new()));
        }
        Some(calls) => blocks.extend(calls.into_iter().map(ingest_tool_call)),
        None => {}
    }
    compat.extend(unknown_fields);
    types::Message {
        role,
        content: blocks,
        ext: namespaced(compat),
    }
}

/// `reasoning_content`, a sibling of `content` carrying chain-of-thought, as a
/// typed block.
///
/// OpenAI has no such field, but it is not one provider's extension either:
/// DeepSeek, vLLM, SGLang and others all emit it from their OpenAI-compatible
/// surfaces, which makes it part of what `openai_compat` means as a permissive
/// superset. So it is promoted for EVERY provider here rather than per-provider
/// — which providers emit it is not a list warpllm should have to keep in Rust,
/// and gating it would mean a new roster entry silently lost the mapping until
/// someone wrote code for it.
///
/// PROMOTE BUT RETAIN: the caller keeps `unknown_fields` intact, so the field
/// still rides `ext["openai_compat"]` and the renderer replays it verbatim.
/// Nothing could rebuild it from the block — `render_message` drops Reasoning
/// blocks, because the protocol has no field to render them into. `ext` is what
/// the provider said; the block is what it means.
///
/// `provenance` records the protocol rather than a hand-written identifier:
/// [`Protocol::as_str`] is the same one string the namespace is keyed by, so
/// the two cannot disagree about where this field came from.
fn reasoning_block(fields: &UnknownFields) -> Option<ContentBlock> {
    let text = fields
        .get("reasoning_content")?
        .as_str()
        // Absent, null, empty, or a non-string all mean "no reasoning". An
        // empty block would claim the provider sent thinking it did not.
        .filter(|text| !text.is_empty())?;
    Some(ContentBlock::Reasoning {
        detail: ReasoningDetail::Text {
            text: text.to_string(),
            // No provider on this protocol publishes a signature for it.
            signature: None,
        },
        provenance: Some(Protocol::OpenAiCompat.as_str().to_string()),
        id: None,
    })
}

/// A plain function call becomes a typed block; anything else — custom
/// tool calls, or calls carrying unknown fields at either level — passes
/// through as an `Unknown` block, re-emitted verbatim in array order.
fn ingest_tool_call(call: ChatCompletionMessageToolCallUnion) -> ContentBlock {
    match &call {
        ChatCompletionMessageToolCallUnion::Function(function_call)
            if function_call.r#type == "function"
                && function_call.unknown_fields.is_empty()
                && function_call.function.unknown_fields.is_empty() =>
        {
            ContentBlock::ToolCall {
                id: function_call.id.clone(),
                name: function_call.function.name.clone(),
                arguments: RawJson::new(function_call.function.arguments.clone()),
            }
        }
        _ => ContentBlock::Unknown(plain(&call)),
    }
}

pub(super) fn ingest_usage(usage: CompletionUsage) -> types::Usage {
    let CompletionUsage {
        completion_tokens,
        prompt_tokens,
        total_tokens,
        completion_tokens_details,
        prompt_tokens_details,
        unknown_fields,
    } = usage;
    let mut compat = UnknownFields::new();
    let mut reasoning_tokens = None;
    let mut cache_read_tokens = None;
    let mut cache_write_tokens = None;
    // A details residue is stashed iff the wire object was present (even
    // empty), so presence itself survives the round trip.
    if let Some(details) = prompt_tokens_details {
        let mut residue = object(plain(&details));
        cache_read_tokens = residue.remove("cached_tokens").and_then(|v| v.as_u64());
        cache_write_tokens = residue
            .remove("cache_write_tokens")
            .and_then(|v| v.as_u64());
        compat.insert("prompt_tokens_details".into(), Value::Object(residue));
    }
    if let Some(details) = completion_tokens_details {
        let mut residue = object(plain(&details));
        reasoning_tokens = residue.remove("reasoning_tokens").and_then(|v| v.as_u64());
        compat.insert("completion_tokens_details".into(), Value::Object(residue));
    }
    compat.extend(unknown_fields);
    types::Usage {
        input_tokens: Some(u64::from(prompt_tokens)),
        output_tokens: Some(u64::from(completion_tokens)),
        total_tokens: Some(u64::from(total_tokens)),
        reasoning_tokens,
        cache_read_tokens,
        cache_write_tokens,
        ext: namespaced(compat),
    }
}

/// Infallible: protocol fields restore from ext (a hook that corrupted a
/// stashed value beyond its wire type falls back to dropping that field).
pub(crate) fn render_response(
    response: &types::ChatResponse,
    provider: &str,
) -> CreateChatCompletionResponse {
    let mut unknown_fields = merged_ext(&response.ext, provider);
    let object =
        take_string(&mut unknown_fields, "object").unwrap_or_else(|| "chat.completion".to_string());
    let choices = response
        .completions
        .iter()
        .enumerate()
        .map(|(position, completion)| render_choice(completion, position, provider))
        .collect();
    CreateChatCompletionResponse {
        id: response.id.clone(),
        choices,
        created: response.created.unwrap_or(0),
        model: response.model.clone(),
        object,
        moderation: take_typed(&mut unknown_fields, "moderation"),
        service_tier: take_nullable_string(&mut unknown_fields, "service_tier"),
        system_fingerprint: take_string(&mut unknown_fields, "system_fingerprint"),
        usage: response.usage.as_ref().map(|u| render_usage(u, provider)),
        unknown_fields,
    }
}

fn render_choice(completion: &types::Completion, position: usize, provider: &str) -> Choice {
    let mut unknown_fields = merged_ext(&completion.ext, provider);
    let index = unknown_fields
        .remove("index")
        .and_then(|v| v.as_u64())
        .unwrap_or(position as u64) as u32;
    Choice {
        finish_reason: completion.finish_reason_raw.clone(),
        index,
        logprobs: take_typed(&mut unknown_fields, "logprobs"),
        message: render_message(&completion.message, provider),
        unknown_fields,
    }
}

fn render_message(message: &types::Message, provider: &str) -> ChatCompletionResponseMessage {
    let mut unknown_fields = merged_ext(&message.ext, provider);
    let role = match unknown_fields.remove("role") {
        Some(Value::String(raw)) => raw,
        _ => role_to_wire(message.role).to_string(),
    };
    let mut texts = Vec::new();
    let mut tool_calls = Vec::new();
    for block in &message.content {
        match block {
            ContentBlock::Text { text, .. } => texts.push(text.as_str()),
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => tool_calls.push(ChatCompletionMessageToolCallUnion::Function(
                ChatCompletionMessageToolCall {
                    id: id.clone(),
                    r#type: "function".into(),
                    function: Function {
                        arguments: arguments.as_str().to_string(),
                        name: name.clone(),
                        unknown_fields: UnknownFields::new(),
                    },
                    unknown_fields: UnknownFields::new(),
                },
            )),
            ContentBlock::Unknown(value) => {
                if let Ok(call) = serde_json::from_value(value.clone()) {
                    tool_calls.push(call);
                }
            }
            // A Reasoning block was lifted out of `reasoning_content`,
            // which is still sitting in ext and about to be emitted
            // verbatim — PROMOTE BUT RETAIN, read from the other end, so
            // dropping the block loses nothing. Media and tool results have
            // no rendering on this protocol at all; they can only arise
            // cross-protocol, which warpllm does not do yet.
            _ => {}
        }
    }
    if !tool_calls.is_empty() {
        // Typed tool calls are authoritative over any stashed empty array.
        unknown_fields.remove("tool_calls");
    }
    ChatCompletionResponseMessage {
        // Same-protocol messages carry at most one text block, so the join
        // is exact; joining >1 only occurs cross-protocol.
        content: (!texts.is_empty()).then(|| texts.join("\n")),
        refusal: take_string(&mut unknown_fields, "refusal"),
        role,
        annotations: take_typed(&mut unknown_fields, "annotations"),
        audio: take_typed(&mut unknown_fields, "audio"),
        function_call: take_typed(&mut unknown_fields, "function_call"),
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        unknown_fields,
    }
}

pub(super) fn render_usage(usage: &types::Usage, provider: &str) -> CompletionUsage {
    let mut unknown_fields = merged_ext(&usage.ext, provider);
    let prompt_details = render_details(
        unknown_fields.remove("prompt_tokens_details"),
        &[
            ("cached_tokens", usage.cache_read_tokens),
            ("cache_write_tokens", usage.cache_write_tokens),
        ],
    );
    let completion_details = render_details(
        unknown_fields.remove("completion_tokens_details"),
        &[("reasoning_tokens", usage.reasoning_tokens)],
    );
    CompletionUsage {
        completion_tokens: usage.output_tokens.unwrap_or(0) as u32,
        prompt_tokens: usage.input_tokens.unwrap_or(0) as u32,
        total_tokens: usage.total_tokens.unwrap_or(0) as u32,
        completion_tokens_details: completion_details,
        prompt_tokens_details: prompt_details,
        unknown_fields,
    }
}

/// Rebuilds a details object from its ext residue plus the typed fields
/// lifted out at ingest. Emitted iff the residue was present (preserving
/// wire presence) or a lifted field is set.
fn render_details<T: DeserializeOwned>(
    residue: Option<Value>,
    lifted: &[(&str, Option<u64>)],
) -> Option<T> {
    let residue = match residue {
        Some(Value::Object(fields)) => Some(fields),
        _ => None,
    };
    if residue.is_none() && lifted.iter().all(|(_, value)| value.is_none()) {
        return None;
    }
    let mut fields = residue.unwrap_or_default();
    for (key, value) in lifted {
        if let Some(tokens) = value {
            fields.insert((*key).to_string(), Value::from(*tokens));
        }
    }
    serde_json::from_value(Value::Object(fields)).ok()
}

pub(super) fn take_string(fields: &mut UnknownFields, key: &str) -> Option<String> {
    match fields.remove(key) {
        Some(Value::String(value)) => Some(value),
        _ => None,
    }
}

/// [`take_string`] for a field the protocol lets be optional AND nullable:
/// absent comes back absent, and an explicit `null` comes back as one. The
/// typed fields get this for free — `take_typed::<Option<T>>` reads a stashed
/// `null` as `Some(None)` — and only a bare string needs it spelled out.
pub(super) fn take_nullable_string(
    fields: &mut UnknownFields,
    key: &str,
) -> Option<Option<String>> {
    match fields.remove(key) {
        Some(Value::String(value)) => Some(Some(value)),
        Some(Value::Null) => Some(None),
        // Absent, or corrupted past its wire type: drop the field.
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

fn object(value: Value) -> UnknownFields {
    match value {
        Value::Object(fields) => fields,
        _ => UnknownFields::new(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn parse(body: Value) -> CreateChatCompletionResponse {
        serde_json::from_value(body).unwrap()
    }

    /// The maximal body: every documented field, unknown fields at every
    /// nesting level, function + custom tool calls, both moderation arms.
    fn maximal_body() -> Value {
        json!({
            "id": "chatcmpl-123",
            "choices": [{
                "finish_reason": "tool_calls",
                "index": 3,
                "logprobs": {
                    "content": [{
                        "token": "Hi",
                        "bytes": [72, 105],
                        "logprob": -0.1,
                        "top_logprobs": [{"token": "Hi", "bytes": null, "logprob": -0.1}]
                    }],
                    "refusal": []
                },
                "message": {
                    "content": "Hello there!",
                    "refusal": "no thanks",
                    "role": "assistant",
                    "annotations": [{
                        "type": "url_citation",
                        "url_citation": {
                            "end_index": 5,
                            "start_index": 0,
                            "title": "Example",
                            "url": "https://example.com"
                        }
                    }],
                    "audio": {
                        "id": "audio-1",
                        "data": "aGk=",
                        "expires_at": 1700000600,
                        "transcript": "hi"
                    },
                    "function_call": {"arguments": "{}", "name": "legacy_fn"},
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"arguments": "{\"z\":1,\"a\":2}", "name": "search"}
                        },
                        {
                            "id": "call-2",
                            "type": "custom",
                            "custom": {"input": "raw text", "name": "my_tool"}
                        },
                        {
                            "id": "call-3",
                            "type": "function",
                            "function": {"arguments": "{}", "name": "extended"},
                            "vendor_extra": true
                        }
                    ],
                    "reasoning_content": "step by step"
                },
                "new_choice_field": true
            }],
            "created": 1_700_000_000,
            "model": "gpt-5.6-2024-08-06",
            "object": "chat.completion",
            "moderation": {
                "input": {
                    "type": "moderation_results",
                    "model": "omni-moderation-latest",
                    "results": [{
                        "categories": {"violence": false},
                        "category_applied_input_types": {"violence": ["text"]},
                        "category_scores": {"violence": 0.001},
                        "flagged": false,
                        "model": "omni-moderation-latest",
                        "type": "moderation_result"
                    }]
                },
                "output": {"type": "error", "code": "moderation_unavailable", "message": "try again"}
            },
            "service_tier": "default",
            "system_fingerprint": "fp_44709d6fcb",
            "usage": {
                "completion_tokens": 12,
                "prompt_tokens": 9,
                "total_tokens": 21,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 5,
                    "rejected_prediction_tokens": 0
                },
                "prompt_tokens_details": {
                    "audio_tokens": 0,
                    "cache_write_tokens": 2,
                    "cached_tokens": 3
                },
                "new_usage_field": 7
            },
            "new_top_level_field": "surprise"
        })
    }

    /// The mandated test: an OpenAI-compatible response must survive
    /// normalization and come back out with ZERO permitted
    /// transformations.
    #[test]
    fn openai_compat_response_round_trip_is_lossless() {
        let body = maximal_body();
        let normalized = ingest_response(parse(body.clone()));
        let rendered = render_response(&normalized, "openai");
        assert_eq!(serde_json::to_value(&rendered).unwrap(), body);

        // Spot-check the typed views the IR exposes along the way.
        assert_eq!(normalized.id, "chatcmpl-123");
        assert_eq!(normalized.model, "gpt-5.6-2024-08-06");
        assert_eq!(normalized.created, Some(1_700_000_000));
        let completion = &normalized.completions[0];
        assert_eq!(completion.finish_reason, FinishReason::ToolCalls);
        assert_eq!(completion.finish_reason_raw, "tool_calls");
        assert_eq!(completion.ext["openai_compat"]["index"], json!(3));
    }

    #[test]
    fn minimal_response_round_trips() {
        // Optional fields absent; content/refusal explicit null contrast.
        let body = json!({
            "id": "chatcmpl-1",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": null, "refusal": null, "role": "assistant"}
            }],
            "created": 1_700_000_000,
            "model": "deepseek-v4-flash",
            "object": "chat.completion"
        });
        let normalized = ingest_response(parse(body.clone()));
        assert!(normalized.completions[0].message.content.is_empty());
        let rendered = render_response(&normalized, "deepseek");
        assert_eq!(serde_json::to_value(&rendered).unwrap(), body);
    }

    #[test]
    fn plain_function_tool_call_becomes_toolcall_block() {
        let normalized = ingest_response(parse(maximal_body()));
        let content = &normalized.completions[0].message.content;
        // The block order this protocol produces: promoted reasoning, then the
        // text, then tool calls in array order.
        assert!(matches!(&content[0], ContentBlock::Reasoning { .. }));
        assert!(matches!(&content[1], ContentBlock::Text { text, .. } if text == "Hello there!"));
        match &content[2] {
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => {
                assert_eq!(id, "call-1");
                assert_eq!(name, "search");
                // Byte-exact, key order untouched.
                assert_eq!(arguments.as_str(), "{\"z\":1,\"a\":2}");
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
        // Custom and extended calls pass through as Unknown, verbatim.
        assert!(
            matches!(&content[3], ContentBlock::Unknown(v) if v["id"] == "call-2" && v["custom"]["input"] == "raw text")
        );
        assert!(
            matches!(&content[4], ContentBlock::Unknown(v) if v["id"] == "call-3" && v["vendor_extra"] == true)
        );
    }

    #[test]
    fn empty_tool_calls_array_survives() {
        let body = json!({
            "id": "chatcmpl-1",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "hi",
                    "refusal": null,
                    "role": "assistant",
                    "tool_calls": []
                }
            }],
            "created": 1_700_000_000,
            "model": "gpt-5.6",
            "object": "chat.completion"
        });
        let normalized = ingest_response(parse(body.clone()));
        let rendered = render_response(&normalized, "openai");
        assert_eq!(serde_json::to_value(&rendered).unwrap(), body);
    }

    #[test]
    fn usage_maps_typed_token_fields() {
        let normalized = ingest_response(parse(maximal_body()));
        let usage = normalized.usage.as_ref().unwrap();
        assert_eq!(usage.input_tokens, Some(9));
        assert_eq!(usage.output_tokens, Some(12));
        assert_eq!(usage.total_tokens, Some(21));
        assert_eq!(usage.reasoning_tokens, Some(5));
        assert_eq!(usage.cache_read_tokens, Some(3));
        assert_eq!(usage.cache_write_tokens, Some(2));
        assert_eq!(usage.ext["openai_compat"]["new_usage_field"], json!(7));
    }

    #[test]
    fn empty_details_object_presence_survives() {
        let body = json!({
            "id": "chatcmpl-1",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": "hi", "refusal": null, "role": "assistant"}
            }],
            "created": 1_700_000_000,
            "model": "gpt-5.6",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 1,
                "prompt_tokens": 2,
                "total_tokens": 3,
                "prompt_tokens_details": {}
            }
        });
        let normalized = ingest_response(parse(body.clone()));
        let rendered = render_response(&normalized, "openai");
        assert_eq!(serde_json::to_value(&rendered).unwrap(), body);
    }

    /// Promoted for any provider on this protocol — which providers emit it is
    /// not a list warpllm keeps — and ordered before the answer it led to.
    #[test]
    fn reasoning_content_becomes_a_reasoning_block_before_the_answer() {
        let normalized = ingest_response(parse(maximal_body()));
        let content = &normalized.completions[0].message.content;

        match &content[0] {
            ContentBlock::Reasoning {
                detail: ReasoningDetail::Text { text, signature },
                provenance,
                id,
            } => {
                assert_eq!(text, "step by step");
                assert_eq!(*signature, None);
                assert_eq!(provenance.as_deref(), Some("openai_compat"));
                assert_eq!(*id, None);
            }
            other => panic!("expected a Reasoning block first, got {other:?}"),
        }
        assert!(
            matches!(&content[1], ContentBlock::Text { .. }),
            "the answer must still follow the reasoning: {content:?}"
        );
    }

    /// Absent, empty, and non-string all mean "no reasoning" — a promoted empty
    /// block would claim the provider sent thinking it never did, and a
    /// non-string must not panic since ingest does not type this field.
    #[test]
    fn only_a_non_empty_string_reasoning_content_is_promoted() {
        for value in [json!(""), json!(null), json!(["not", "a", "string"])] {
            let mut body = maximal_body();
            body["choices"][0]["message"]["reasoning_content"] = value.clone();
            let normalized = ingest_response(parse(body));
            let content = &normalized.completions[0].message.content;
            assert!(
                !matches!(content[0], ContentBlock::Reasoning { .. }),
                "{value} was promoted"
            );
        }
    }

    /// A message with no `reasoning_content` at all gains no block.
    #[test]
    fn a_message_without_reasoning_content_gains_no_block() {
        let mut body = maximal_body();
        body["choices"][0]["message"]
            .as_object_mut()
            .unwrap()
            .remove("reasoning_content");
        let normalized = ingest_response(parse(body));

        assert!(
            !normalized.completions[0]
                .message
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::Reasoning { .. }))
        );
    }

    #[test]
    fn protocol_extras_land_in_ext() {
        let normalized = ingest_response(parse(maximal_body()));
        let ext = &normalized.ext["openai_compat"];
        assert_eq!(ext["object"], "chat.completion");
        assert_eq!(ext["service_tier"], "default");
        assert_eq!(ext["system_fingerprint"], "fp_44709d6fcb");
        assert_eq!(ext["new_top_level_field"], "surprise");
        let message = &normalized.completions[0].message;
        assert_eq!(message.ext["openai_compat"]["refusal"], "no thanks");
        assert_eq!(
            message.ext["openai_compat"]["reasoning_content"],
            "step by step"
        );
    }

    #[test]
    fn ingest_populates_source() {
        let wire = parse(maximal_body());
        let normalized = ingest_response(wire.clone());
        let source = normalized.source.as_ref().unwrap();
        assert_eq!(source.protocol, Protocol::OpenAiCompat);
        assert_eq!(source.body, serde_json::to_value(&wire).unwrap());
    }

    /// A block this protocol has no field for is dropped, and the rest of
    /// the message still renders. Only reachable cross-protocol, which
    /// warpllm does not do yet — pinned so the behavior is stated rather
    /// than discovered.
    #[test]
    fn a_block_with_no_rendering_is_dropped_and_the_rest_survives() {
        let mut normalized = ingest_response(parse(maximal_body()));
        normalized.completions[0].message.content = vec![
            ContentBlock::Text {
                text: "here it is".into(),
                cache: None,
            },
            ContentBlock::Image {
                source: crate::gateway::types::MediaSource::Url {
                    url: "https://example.com/a.png".into(),
                },
                detail: None,
                cache: None,
            },
        ];

        let rendered = render_response(&normalized, "openai");
        assert_eq!(
            rendered.choices[0].message.content.as_deref(),
            Some("here it is")
        );
    }

    /// PROMOTE BUT RETAIN, read from the render side: the Reasoning block
    /// is dropped, but the `reasoning_content` it was lifted out of is
    /// still in ext and comes back verbatim — which is why dropping the
    /// block loses nothing.
    #[test]
    fn a_dropped_reasoning_block_still_renders_from_ext() {
        let normalized = ingest_response(parse(maximal_body()));
        assert!(
            normalized.completions[0]
                .message
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::Reasoning { .. })),
            "the fixture no longer exercises the retained-reasoning path"
        );

        let rendered = render_response(&normalized, "openai");
        assert_eq!(
            rendered.choices[0].message.unknown_fields["reasoning_content"],
            json!("step by step")
        );
    }
}
