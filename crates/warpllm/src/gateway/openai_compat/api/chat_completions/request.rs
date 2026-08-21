//! Request conversions: OpenAI-compatible wire → gateway (ingest) and
//! gateway → wire (render).
//!
//! # Promote but retain
//!
//! Every typed field this module lifts into the gateway form is ALSO kept in
//! `ext["openai_compat"]` exactly as it arrived, and [`render_request`] prefers
//! the retained value over re-rendering the gateway form. `reasoning_content`
//! in `response.rs` already works this way; the reason is the same here and
//! matters more, because content parts and tools carry residue the gateway
//! shapes have no field for — a part's `type` spelling, a tool's vendor
//! extensions, the difference between `"hi"` and `[{"type":"text",...}]`.
//!
//! So a request that arrives on this protocol and leaves on it is byte-exact,
//! and one that arrives on ANOTHER protocol has no residue here and is
//! rendered from the gateway form. Both paths are real: the first is every
//! request to an OpenAI-compatible backend, the second is what makes a second
//! protocol reachable at all.
//!
//! What "exactly as it arrived" rests on: [`retain`] stores the PARSED value
//! re-serialized, not the original bytes — by the time this runs the caller
//! has handed over a struct and the bytes are gone. So the residue is faithful
//! only as far as the wire types round trip losslessly, and a field that read
//! an explicit `null` as absent would lose it here no matter what this module
//! did. That is why every request field the vendor documents as nullable is
//! `Option<Option<T>>` behind `present_or_null` rather than a plain `Option`.
//!
//! The caveat, and it is the same one `reasoning_content` has: something that
//! edited `ChatRequest::tools` between ingest and render would find the
//! retained value winning. Nothing does that today, and the moment something
//! does it will want to drop the residue deliberately rather than have this
//! guess.

use serde_json::Value;

use crate::error::{Error, Result};
use crate::gateway::types::{self, ContentBlock, IngestSource, MediaSource, RawJson};
use crate::protocol::openai_compat::chat_completions::types::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCallUnion,
    ChatCompletionNamedToolChoice, ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContent, ChatCompletionRequestMessageContentPart,
    ChatCompletionRequestMessageContentPartAudio, ChatCompletionRequestMessageContentPartFile,
    ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionResponseFormat, ChatCompletionStop, ChatCompletionStreamOptions,
    ChatCompletionTool, ChatCompletionToolChoiceOption, CreateChatCompletionRequest, FileContent,
    Function, FunctionObject, ImageUrl, InputAudio, JsonSchemaDefinition, ResponseFormatJsonSchema,
    ResponseFormatSimple, ToolChoiceFunction, UnknownFields,
};
use crate::types::Protocol;

use super::response::{plain, take_nullable_string, take_string, take_typed};
use crate::gateway::openai_compat::{merged_ext, namespaced, role_from_wire, role_to_wire};

/// Permissive and infallible: capture, don't validate. `model` is the
/// prefix-stripped name from `parse_model`; `stream` is resolved before
/// ingest (`Some(true)` already rejected; false/`None` are equivalent).
///
/// The destructuring is deliberately exhaustive (no `..`): adding a typed
/// field to the wire request without mapping it here is a compile error,
/// so normalization can never silently drop data. Everything without a
/// typed wire field (seed, top_k, ...) rides `ext["openai_compat"]`
/// verbatim.
pub(crate) fn ingest_request(
    request: CreateChatCompletionRequest,
    model: &str,
) -> types::ChatRequest {
    // Wire structs are plain serde data; serialization cannot fail.
    let body = serde_json::to_value(&request).expect("wire request serializes");
    let CreateChatCompletionRequest {
        model: _,
        messages,
        temperature,
        max_tokens,
        top_p,
        stop,
        stream,
        stream_options,
        tools,
        tool_choice,
        response_format,
        reasoning_effort,
        unknown_fields,
    } = request;

    let mut compat = unknown_fields;
    retain(&mut compat, "temperature", temperature.as_ref());
    retain(&mut compat, "max_tokens", max_tokens.as_ref());
    retain(&mut compat, "top_p", top_p.as_ref());
    retain(&mut compat, "stop", stop.as_ref());
    retain(&mut compat, "stream_options", stream_options.as_ref());
    retain(&mut compat, "tools", tools.as_ref());
    retain(&mut compat, "tool_choice", tool_choice.as_ref());
    retain(&mut compat, "response_format", response_format.as_ref());
    retain(&mut compat, "reasoning_effort", reasoning_effort.as_ref());

    types::ChatRequest {
        model: model.to_string(),
        messages: messages.into_iter().map(ingest_message).collect(),
        tools: tools
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(ingest_tool)
            .collect(),
        tool_choice: tool_choice.as_ref().and_then(ingest_tool_choice),
        params: types::GenerationParams {
            // Both an absent field and an explicit `null` mean "no value" to
            // the gateway IR; which one it was rides the residue above so
            // rendering can restore it.
            max_tokens: max_tokens.flatten(),
            temperature: temperature.flatten(),
            top_p: top_p.flatten(),
            // Both wire spellings mean the same list of sequences; the one
            // that arrived is retained above so rendering can restore it.
            stop: stop
                .as_ref()
                .and_then(Option::as_ref)
                .map(ChatCompletionStop::sequences)
                .unwrap_or_default(),
        },
        response_format: response_format.as_ref().and_then(ingest_response_format),
        // Only the effort half: chat completions has no token budget to read,
        // and inventing one from a word would be a number nobody wrote.
        reasoning: reasoning_effort
            .flatten()
            .map(|effort| types::ReasoningConfig {
                effort: Some(effort),
                ..Default::default()
            }),
        stream: stream.unwrap_or(false),
        stream_include_usage: stream_options
            .flatten()
            .and_then(|options| options.include_usage),
        ext: namespaced(compat),
        source: Some(IngestSource {
            protocol: Protocol::OpenAiCompat,
            body,
        }),
        ..Default::default()
    }
}

/// Keeps a typed field's wire value under this protocol's namespace, so
/// [`render_request`] can hand back exactly what arrived. See this module's
/// docs for why the promoted form is not enough on its own.
fn retain<T: serde::Serialize>(compat: &mut UnknownFields, key: &str, value: Option<&T>) {
    if let Some(value) = value {
        compat.insert(key.to_string(), plain(value));
    }
}

fn ingest_message(message: ChatCompletionRequestMessage) -> types::Message {
    let ChatCompletionRequestMessage {
        role,
        content,
        tool_calls,
        tool_call_id,
        unknown_fields,
    } = message;
    let (role, raw_role) = role_from_wire(role);
    let mut compat = unknown_fields;
    if let Some(raw) = raw_role {
        // Unrecognized roles (e.g. "developer") pass through verbatim:
        // render restores the raw spelling from ext.
        compat.insert("role".into(), Value::String(raw));
    }
    retain(&mut compat, "content", content.as_ref());
    retain(&mut compat, "tool_calls", tool_calls.as_ref());
    retain(&mut compat, "tool_call_id", tool_call_id.as_ref());

    // An absent `content` and an explicit `null` both carry no blocks; which
    // one it was rides the residue above, so rendering restores the right one.
    let mut blocks: Vec<ContentBlock> = match content.as_ref().and_then(Option::as_ref) {
        Some(ChatCompletionRequestMessageContent::Text(text)) => vec![ContentBlock::Text {
            text: text.clone(),
            cache: None,
        }],
        Some(ChatCompletionRequestMessageContent::Parts(parts)) => {
            parts.iter().map(ingest_part).collect()
        }
        None => Vec::new(),
    };
    // A tool result is one block carrying the id of the call it answers,
    // which is what lets a protocol that groups results into a single turn
    // (Anthropic) reassemble them. `tool_call_id` is what says this is one.
    if let Some(call_id) = tool_call_id {
        blocks = vec![ContentBlock::ToolResult {
            call_id,
            content: blocks,
            // Chat completions has no way to mark a result as failed; the
            // model reads the text. Only a protocol that flags it can set this.
            is_error: false,
        }];
    }
    blocks.extend(tool_calls.into_iter().flatten().map(ingest_tool_call));

    types::Message {
        role,
        content: blocks,
        ext: namespaced(compat),
    }
}

fn ingest_part(part: &ChatCompletionRequestMessageContentPart) -> ContentBlock {
    match part {
        ChatCompletionRequestMessageContentPart::Text(part) => ContentBlock::Text {
            text: part.text.clone(),
            cache: None,
        },
        ChatCompletionRequestMessageContentPart::Image(part) => ContentBlock::Image {
            source: media_source(&part.image_url.url),
            detail: part.image_url.detail.clone(),
            cache: None,
        },
        ChatCompletionRequestMessageContentPart::Audio(part) => ContentBlock::Audio {
            // The format IS the subtype for every container OpenAI documents,
            // and it is the only thing on the wire that could name a media
            // type — so it is composed rather than guessed at.
            source: MediaSource::Base64 {
                media_type: format!("audio/{}", part.input_audio.format),
                data: part.input_audio.data.clone(),
            },
            format: Some(part.input_audio.format.clone()),
        },
        ChatCompletionRequestMessageContentPart::File(part) => match &part.file {
            FileContent {
                file_id: Some(id), ..
            } => ContentBlock::Document {
                source: MediaSource::ProviderFile { id: id.clone() },
                title: part.file.filename.clone(),
                cache: None,
            },
            FileContent {
                file_data: Some(data),
                ..
            } => ContentBlock::Document {
                // RAW base64, not a `data:` URL. The vendor documents this
                // field as "the base64 encoded file data" flat out, where
                // `image_url.url` is "either a URL of the image OR the base64
                // encoded image data" — so the two fields take different
                // things and only the image one is worth parsing.
                //
                // The media type is empty because the file part has no field
                // for one. A target that requires it (Anthropic wants
                // `application/pdf`) has to say so rather than be handed a
                // guess made from the filename.
                source: MediaSource::Base64 {
                    media_type: String::new(),
                    data: data.clone(),
                },
                title: part.file.filename.clone(),
                cache: None,
            },
            // Neither a reference nor bytes: nothing to point a renderer at.
            _ => ContentBlock::Unknown(plain(part)),
        },
        // A refusal is the model declining, not content the caller sent. The
        // gateway form has no block for it, and inventing one would put it in
        // front of a renderer as if it were something to say.
        ChatCompletionRequestMessageContentPart::Refusal(part) => {
            ContentBlock::Unknown(plain(part))
        }
        ChatCompletionRequestMessageContentPart::Unknown(value) => {
            ContentBlock::Unknown(value.clone())
        }
    }
}

/// Splits a `data:` URL into its media type and payload, so a protocol that
/// wants them apart (Anthropic) does not have to re-parse a string. Anything
/// else is a plain URL, which is the other thing OpenAI accepts here.
fn media_source(url: &str) -> MediaSource {
    match url
        .strip_prefix("data:")
        .and_then(|rest| rest.split_once(";base64,"))
    {
        Some((media_type, data)) => MediaSource::Base64 {
            media_type: media_type.to_string(),
            data: data.to_string(),
        },
        None => MediaSource::Url {
            url: url.to_string(),
        },
    }
}

fn ingest_tool_call(call: ChatCompletionMessageToolCallUnion) -> ContentBlock {
    match call {
        ChatCompletionMessageToolCallUnion::Function(call) => ContentBlock::ToolCall {
            id: call.id,
            name: call.function.name,
            // Model-generated, so not necessarily valid JSON. `RawJson` holds
            // the exact text either way; whoever renders it decides what to do
            // about that.
            arguments: RawJson::new(&call.function.arguments),
        },
        // A custom tool's `input` is free text against a grammar, not JSON
        // arguments, so it is not a `ToolCall` — calling it one would put text
        // where every renderer expects an object.
        ChatCompletionMessageToolCallUnion::Custom(call) => ContentBlock::Unknown(plain(&call)),
    }
}

fn ingest_tool(tool: &ChatCompletionTool) -> types::ToolDef {
    let mut ext = UnknownFields::new();
    ext.insert("type".into(), Value::String(tool.r#type.clone()));
    match &tool.function {
        Some(function) => types::ToolDef {
            name: function.name.clone(),
            description: function.description.clone(),
            // Verbatim: providers that reject some JSON Schema keywords prune
            // in their own renderer, never here.
            input_schema: function.parameters.clone().unwrap_or(Value::Null),
            strict: function.strict.flatten(),
            cache: None,
            ext: namespaced(ext),
        },
        // A hosted tool (`{"type": "web_search"}`) or a custom one: no
        // arguments schema exists to translate. It keeps its `type` as a name
        // and carries the whole wire object in ext, which is what lets a
        // renderer for another protocol refuse it by name instead of sending
        // something invented. A neutral representation for these is #47's.
        None => types::ToolDef {
            name: tool.r#type.clone(),
            description: None,
            input_schema: Value::Null,
            strict: None,
            cache: None,
            ext: namespaced(object(plain(tool))),
        },
    }
}

fn ingest_tool_choice(choice: &ChatCompletionToolChoiceOption) -> Option<types::ToolChoice> {
    match choice {
        ChatCompletionToolChoiceOption::Named(named) => Some(types::ToolChoice::Tool {
            name: named.function.name.clone(),
        }),
        ChatCompletionToolChoiceOption::Mode(mode) => match mode.as_str() {
            "auto" => Some(types::ToolChoice::Auto),
            "none" => Some(types::ToolChoice::None),
            "required" => Some(types::ToolChoice::Required),
            // A mode this gateway has no word for. It still reaches an
            // OpenAI-compatible target through the retained residue; there is
            // just nothing to tell another protocol.
            _ => None,
        },
        // A choice shape with no gateway word either — `allowed_tools`, or a
        // custom tool. Same deal as an unrecognized mode.
        ChatCompletionToolChoiceOption::Unknown(_) => None,
    }
}

fn ingest_response_format(format: &ChatCompletionResponseFormat) -> Option<types::ResponseFormat> {
    match format {
        ChatCompletionResponseFormat::JsonSchema(format) => {
            Some(types::ResponseFormat::JsonSchema {
                name: format.json_schema.name.clone(),
                schema: format.json_schema.schema.clone().unwrap_or(Value::Null),
                strict: format.json_schema.strict.flatten(),
            })
        }
        ChatCompletionResponseFormat::Simple(format) => match format.r#type.as_str() {
            "text" => Some(types::ResponseFormat::Text),
            "json_object" => Some(types::ResponseFormat::JsonObject),
            _ => None,
        },
    }
}

/// Renders the gateway request back onto the wire. Mechanical by
/// design: nothing is filtered against what the target documents — the
/// provider is the authority on its own parameters and rejects what it
/// doesn't accept. Same-protocol round trips are lossless modulo three
/// documented transformations: the provider prefix is stripped from
/// `model`, `stream: false` is dropped, and an empty `stop` list is omitted,
/// each of them exactly reversible or resolved before ingest.
///
/// `stream` is emitted only when it is true. The wire field is optional and
/// defaults to false, so an absent key and an explicit `false` ask the
/// provider for the same thing — and ingest already reads them as the same
/// thing, which is what keeps dropping it lossless.
pub(crate) fn render_request(
    request: &types::ChatRequest,
    provider: &'static str,
) -> Result<CreateChatCompletionRequest> {
    ensure_renderable(request)?;
    let mut unknown_fields = merged_ext(&request.ext, provider);
    let params = &request.params;
    // One gateway message can be several wire ones; see [`render_message`].
    let mut messages = Vec::with_capacity(request.messages.len());
    for message in &request.messages {
        messages.extend(render_message(message, provider)?);
    }
    // Hoisted rather than inlined below: rendering a tool can fail, and `?`
    // has nowhere to go inside an `or_else` closure returning `Option`.
    let tools = match take_typed(&mut unknown_fields, "tools") {
        Some(tools) => Some(tools),
        None if request.tools.is_empty() => None,
        None => Some(
            request
                .tools
                .iter()
                .map(render_tool)
                .collect::<Result<Vec<_>>>()?,
        ),
    };
    Ok(CreateChatCompletionRequest {
        model: request.model.clone(),
        messages,
        // Each of these prefers what arrived over what the gateway IR can
        // reconstruct: a null the caller sent survives the round trip.
        temperature: take_typed(&mut unknown_fields, "temperature")
            .or_else(|| params.temperature.map(Some)),
        max_tokens: take_typed(&mut unknown_fields, "max_tokens")
            .or_else(|| params.max_tokens.map(Some)),
        top_p: take_typed(&mut unknown_fields, "top_p").or_else(|| params.top_p.map(Some)),
        stop: take_typed(&mut unknown_fields, "stop").or_else(|| {
            (!params.stop.is_empty()).then(|| Some(ChatCompletionStop::Many(params.stop.clone())))
        }),
        stream: request.stream.then_some(true),
        // Each of these prefers what arrived over what the gateway form can
        // reconstruct; see this module's docs.
        stream_options: take_typed(&mut unknown_fields, "stream_options").or_else(|| {
            request.stream_include_usage.map(|include_usage| {
                Some(ChatCompletionStreamOptions {
                    include_usage: Some(include_usage),
                    unknown_fields: UnknownFields::new(),
                })
            })
        }),
        tools,
        tool_choice: take_typed(&mut unknown_fields, "tool_choice")
            .or_else(|| request.tool_choice.as_ref().map(render_tool_choice)),
        response_format: take_typed(&mut unknown_fields, "response_format")
            .or_else(|| request.response_format.as_ref().map(render_response_format)),
        reasoning_effort: take_nullable_string(&mut unknown_fields, "reasoning_effort").or_else(
            || {
                request
                    .reasoning
                    .as_ref()
                    .and_then(|reasoning| reasoning.effort.clone())
                    .map(Some)
            },
        ),
        unknown_fields,
    })
}

/// What the gateway form can hold and this protocol cannot say.
///
/// Narrower than the gate it replaces, because tools, a tool choice and a
/// response format all became renderable — but not empty, and the remainder is
/// not a staging list. Chat completions has no cache-breakpoint vocabulary
/// warpllm models, and its only reasoning control is `reasoning_effort`, so a
/// token budget or a "reason but do not return it" flag has nowhere to go.
///
/// Rendering these as silence would CHANGE the request rather than translate
/// it: a caller who asked for a 4k thinking budget would get an ordinary
/// completion back with nothing saying the budget was ignored. Refusing names
/// the field instead.
///
/// Nothing on this protocol's own ingest path sets any of them, so this can
/// only fire on a request that arrived somewhere else.
fn ensure_renderable(request: &types::ChatRequest) -> Result<()> {
    if request.cache.is_some()
        || request.tools.iter().any(|tool| tool.cache.is_some())
        || request
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .any(has_cache_hint)
    {
        return Err(Error::NotImplemented(
            "cache hints on the openai_compat renderer",
        ));
    }
    // A failed tool result is a fact about the conversation, not a detail of
    // it: the model reads it as "that call did not work" and behaves
    // differently. Chat completions has no bit for it — a tool message is
    // `{content, role, tool_call_id}` and nothing else — and the only way to
    // convey it would be to edit the content, which puts words in the tool's
    // mouth that its author never wrote.
    //
    // `is_error: false` is the ordinary case and carries no information, so it
    // renders normally; only a result that says it FAILED has nowhere to go.
    if request
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .any(has_failed_result)
    {
        return Err(Error::NotImplemented(
            "a failed tool result on the openai_compat renderer",
        ));
    }
    if let Some(reasoning) = &request.reasoning {
        if reasoning.budget_tokens.is_some() || reasoning.exclude.is_some() {
            return Err(Error::NotImplemented(
                "a reasoning budget or exclusion on the openai_compat renderer",
            ));
        }
        // `effort` is the whole vocabulary, so asking for reasoning without
        // one leaves nothing to emit.
        if reasoning.enabled.is_some() && reasoning.effort.is_none() {
            return Err(Error::NotImplemented(
                "reasoning without an effort on the openai_compat renderer",
            ));
        }
    }
    Ok(())
}

fn has_cache_hint(block: &ContentBlock) -> bool {
    match block {
        ContentBlock::Text { cache, .. }
        | ContentBlock::Image { cache, .. }
        | ContentBlock::Document { cache, .. } => cache.is_some(),
        ContentBlock::ToolResult { content, .. } => content.iter().any(has_cache_hint),
        _ => false,
    }
}

fn has_failed_result(block: &ContentBlock) -> bool {
    match block {
        ContentBlock::ToolResult {
            is_error, content, ..
        } => *is_error || content.iter().any(has_failed_result),
        _ => false,
    }
}

/// One gateway message, rendered as the one OR MORE wire messages this
/// protocol needs to say it.
///
/// The count is not always one because tool results group differently per
/// protocol. Chat completions carries exactly one result per message, keyed by
/// `tool_call_id`; Anthropic merges a run of consecutive results into a single
/// `user` turn holding several `tool_result` blocks. So a message arriving from
/// there splits back into one message per result here, which is the inverse of
/// the merge [`ingest_message`] undoes. Rendering only the first would leave
/// every other call the model made looking unanswered — and a model told its
/// tool never replied does not fail, it guesses.
fn render_message(
    message: &types::Message,
    provider: &str,
) -> Result<Vec<ChatCompletionRequestMessage>> {
    let mut unknown_fields = merged_ext(&message.ext, provider);
    let role = match unknown_fields.remove("role") {
        Some(Value::String(raw)) => raw,
        _ => role_to_wire(message.role).to_string(),
    };
    let retained_content = take_typed(&mut unknown_fields, "content");
    let retained_calls = take_typed(&mut unknown_fields, "tool_calls");
    let retained_id = take_string(&mut unknown_fields, "tool_call_id");

    let results = results_of(&message.content);
    if results.len() > 1 {
        // Only reachable from another protocol — this one's ingest never
        // builds a message with two results, so there is no retained residue
        // to divide and `unknown_fields` is empty. It rides the first message
        // rather than being copied onto every one, which would duplicate any
        // vendor field a future path did put there.
        return Ok(results
            .into_iter()
            .enumerate()
            .map(
                |(position, (call_id, content))| ChatCompletionRequestMessage {
                    role: role.clone(),
                    content: render_content(content).map(Some),
                    tool_calls: None,
                    tool_call_id: Some(call_id.clone()),
                    unknown_fields: if position == 0 {
                        std::mem::take(&mut unknown_fields)
                    } else {
                        UnknownFields::new()
                    },
                },
            )
            .collect());
    }

    Ok(vec![ChatCompletionRequestMessage {
        role,
        content: match retained_content {
            Some(content) => Some(content),
            // A message built on another protocol always has content to say,
            // so there is no explicit null to restore — only a value or
            // nothing.
            None => render_content(&message.content).map(Some),
        },
        tool_calls: match retained_calls {
            Some(calls) => Some(calls),
            None => render_tool_calls(&message.content),
        },
        // Chat completions puts the answered call's id on the message; the
        // gateway form puts it on the block, so it is read back off the one
        // result this message carries.
        tool_call_id: retained_id.or_else(|| results.first().map(|(id, _)| (*id).clone())),
        unknown_fields,
    }])
}

/// Every tool result in a message, as the call it answers and its payload.
///
/// `is_error` is deliberately not returned: [`ensure_renderable`] has already
/// refused any result that sets it, so every result reaching here is a
/// successful one and there is no flag left to carry.
fn results_of(blocks: &[ContentBlock]) -> Vec<(&String, &[ContentBlock])> {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolResult {
                call_id, content, ..
            } => Some((call_id, content.as_slice())),
            _ => None,
        })
        .collect()
}

/// Blocks → a message's `content`. A single text block renders as a bare
/// string, which is what the overwhelming majority of messages are and what
/// every provider's own examples show; anything else renders as parts.
fn render_content(blocks: &[ContentBlock]) -> Option<ChatCompletionRequestMessageContent> {
    // A tool result's payload is its inner blocks, and chat completions has
    // nowhere else to put them: the result IS the message.
    if let Some(ContentBlock::ToolResult { content, .. }) = blocks
        .iter()
        .find(|b| matches!(b, ContentBlock::ToolResult { .. }))
    {
        return render_content(content);
    }
    let parts: Vec<_> = blocks.iter().filter_map(render_part).collect();
    match parts.as_slice() {
        [] => None,
        [ChatCompletionRequestMessageContentPart::Text(part)] if part.unknown_fields.is_empty() => {
            Some(ChatCompletionRequestMessageContent::Text(part.text.clone()))
        }
        _ => Some(ChatCompletionRequestMessageContent::Parts(parts)),
    }
}

/// `None` for a block that is not message content: a tool CALL rides
/// `tool_calls`, and reasoning has no request-side home on this protocol at
/// all — a provider that returned thinking cannot be told about it here.
fn render_part(block: &ContentBlock) -> Option<ChatCompletionRequestMessageContentPart> {
    match block {
        ContentBlock::Text { text, .. } => Some(ChatCompletionRequestMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartText {
                r#type: "text".into(),
                text: text.clone(),
                unknown_fields: UnknownFields::new(),
            },
        )),
        ContentBlock::Image { source, detail, .. } => {
            Some(ChatCompletionRequestMessageContentPart::Image(
                ChatCompletionRequestMessageContentPartImage {
                    r#type: "image_url".into(),
                    image_url: ImageUrl {
                        url: media_url(source),
                        detail: detail.clone(),
                        unknown_fields: UnknownFields::new(),
                    },
                    unknown_fields: UnknownFields::new(),
                },
            ))
        }
        ContentBlock::Audio { source, format } => {
            let (media_type, data) = match source {
                MediaSource::Base64 { media_type, data } => (media_type.as_str(), data.clone()),
                // A URL or a provider file cannot become inline audio bytes,
                // which is the only audio form this protocol takes.
                _ => return Some(unknown_part(block)),
            };
            Some(ChatCompletionRequestMessageContentPart::Audio(
                ChatCompletionRequestMessageContentPartAudio {
                    r#type: "input_audio".into(),
                    input_audio: InputAudio {
                        data,
                        format: format
                            .clone()
                            .unwrap_or_else(|| subtype(media_type).to_string()),
                        unknown_fields: UnknownFields::new(),
                    },
                    unknown_fields: UnknownFields::new(),
                },
            ))
        }
        ContentBlock::Document { source, title, .. } => {
            let file = match source {
                MediaSource::ProviderFile { id } => FileContent {
                    file_data: None,
                    file_id: Some(id.clone()),
                    filename: title.clone(),
                    unknown_fields: UnknownFields::new(),
                },
                // Raw base64 back out, matching what the field takes. The
                // media type has nowhere to go — the file part declares none —
                // so a document that carried one loses it here.
                MediaSource::Base64 { data, .. } => FileContent {
                    file_data: Some(data.clone()),
                    file_id: None,
                    filename: title.clone(),
                    unknown_fields: UnknownFields::new(),
                },
                // A file part takes bytes or an uploaded id, never a URL.
                MediaSource::Url { .. } => return Some(unknown_part(block)),
            };
            Some(ChatCompletionRequestMessageContentPart::File(
                ChatCompletionRequestMessageContentPartFile {
                    r#type: "file".into(),
                    file,
                    unknown_fields: UnknownFields::new(),
                },
            ))
        }
        ContentBlock::Unknown(value) => Some(ChatCompletionRequestMessageContentPart::Unknown(
            value.clone(),
        )),
        // Handled elsewhere, or not representable in a request on this
        // protocol. See this function's doc.
        ContentBlock::ToolCall { .. }
        | ContentBlock::ToolResult { .. }
        | ContentBlock::Reasoning { .. } => None,
    }
}

fn unknown_part(block: &ContentBlock) -> ChatCompletionRequestMessageContentPart {
    ChatCompletionRequestMessageContentPart::Unknown(plain(block))
}

/// The inverse of [`media_source`]: inline bytes become a `data:` URL, which
/// is the single string field this protocol has for them.
fn media_url(source: &MediaSource) -> String {
    match source {
        MediaSource::Url { url } => url.clone(),
        MediaSource::Base64 { media_type, data } => format!("data:{media_type};base64,{data}"),
        MediaSource::ProviderFile { id } => id.clone(),
    }
}

fn subtype(media_type: &str) -> &str {
    media_type
        .split_once('/')
        .map_or(media_type, |(_, sub)| sub)
}

fn render_tool_calls(blocks: &[ContentBlock]) -> Option<Vec<ChatCompletionMessageToolCallUnion>> {
    let calls: Vec<_> = blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => Some(ChatCompletionMessageToolCallUnion::Function(
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
            _ => None,
        })
        .collect();
    (!calls.is_empty()).then_some(calls)
}

fn render_tool(tool: &types::ToolDef) -> Result<ChatCompletionTool> {
    // A tool with no arguments schema is a hosted or custom one, and this
    // protocol has no shape for one it did not itself ingest. Refusing names
    // it; sending a function tool with an empty schema would have the model
    // call something that does not exist.
    if tool.input_schema.is_null() {
        return Err(Error::NotImplemented(
            "a tool with no arguments schema on the openai_compat renderer",
        ));
    }
    Ok(ChatCompletionTool {
        r#type: "function".into(),
        function: Some(FunctionObject {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: Some(tool.input_schema.clone()),
            strict: tool.strict.map(Some),
            unknown_fields: UnknownFields::new(),
        }),
        unknown_fields: UnknownFields::new(),
    })
}

fn render_tool_choice(choice: &types::ToolChoice) -> ChatCompletionToolChoiceOption {
    match choice {
        types::ToolChoice::Auto => ChatCompletionToolChoiceOption::Mode("auto".into()),
        types::ToolChoice::None => ChatCompletionToolChoiceOption::Mode("none".into()),
        types::ToolChoice::Required => ChatCompletionToolChoiceOption::Mode("required".into()),
        types::ToolChoice::Tool { name } => {
            ChatCompletionToolChoiceOption::Named(ChatCompletionNamedToolChoice {
                r#type: "function".into(),
                function: ToolChoiceFunction {
                    name: name.clone(),
                    unknown_fields: UnknownFields::new(),
                },
                unknown_fields: UnknownFields::new(),
            })
        }
    }
}

fn render_response_format(format: &types::ResponseFormat) -> ChatCompletionResponseFormat {
    match format {
        types::ResponseFormat::Text => ChatCompletionResponseFormat::Simple(ResponseFormatSimple {
            r#type: "text".into(),
            unknown_fields: UnknownFields::new(),
        }),
        types::ResponseFormat::JsonObject => {
            ChatCompletionResponseFormat::Simple(ResponseFormatSimple {
                r#type: "json_object".into(),
                unknown_fields: UnknownFields::new(),
            })
        }
        types::ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => ChatCompletionResponseFormat::JsonSchema(ResponseFormatJsonSchema {
            r#type: "json_schema".into(),
            json_schema: JsonSchemaDefinition {
                name: name.clone(),
                description: None,
                schema: Some(schema.clone()),
                strict: strict.map(Some),
                unknown_fields: UnknownFields::new(),
            },
            unknown_fields: UnknownFields::new(),
        }),
    }
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
    use crate::gateway::types::{ProviderExt, Role};

    fn wire(value: Value) -> CreateChatCompletionRequest {
        serde_json::from_value(value).unwrap()
    }

    fn full_wire_request() -> CreateChatCompletionRequest {
        let mut request = CreateChatCompletionRequest {
            model: "openai/gpt-5.6".into(),
            messages: vec![
                ChatCompletionRequestMessage {
                    role: "developer".into(),
                    content: Some(Some("be brief".into())),
                    ..Default::default()
                },
                ChatCompletionRequestMessage {
                    role: "user".into(),
                    content: Some(Some("hi".into())),
                    ..Default::default()
                },
            ],
            temperature: Some(Some(0.7)),
            max_tokens: Some(Some(256)),
            top_p: Some(Some(0.9)),
            stop: Some(Some(ChatCompletionStop::Many(vec![
                "END".into(),
                "STOP".into(),
            ]))),
            stream: Some(false),
            ..Default::default()
        };
        request.unknown_fields.insert("top_k".into(), json!(40));
        request.unknown_fields.insert("seed".into(), json!(1234));
        request
            .unknown_fields
            .insert("vendor_beta".into(), json!(true));
        request.messages[1]
            .unknown_fields
            .insert("name".into(), json!("alice"));
        request.messages[1]
            .unknown_fields
            .insert("vendor_tag".into(), json!("vt"));
        request
    }

    /// An OpenAI-compatible request must survive normalization and come
    /// back out with zero data loss. The only permitted transformations:
    /// the provider prefix is stripped from `model`, `stream` is dropped,
    /// and an empty `stop` list is omitted.
    #[test]
    fn openai_compat_round_trip_is_lossless() {
        let original = full_wire_request();
        let normalized = ingest_request(original.clone(), "gpt-5.6");
        let wire = render_request(&normalized, "openai").unwrap();

        assert_eq!(wire.model, "gpt-5.6");
        assert_eq!(wire.messages.len(), 2);
        // The unrecognized "developer" role survives verbatim.
        assert_eq!(wire.messages[0].role, "developer");
        assert!(wire.messages[0].unknown_fields.is_empty());
        assert_eq!(wire.messages[1].role, "user");
        // Message-level passthrough fields ride ext and come back verbatim.
        assert_eq!(wire.messages[1].unknown_fields["name"], json!("alice"));
        assert_eq!(wire.messages[1].unknown_fields["vendor_tag"], json!("vt"));
        assert_eq!(wire.temperature, Some(Some(0.7)));
        assert_eq!(wire.max_tokens, Some(Some(256)));
        assert_eq!(wire.top_p, Some(Some(0.9)));
        assert_eq!(plain(&wire.stop), json!(["END", "STOP"]));
        assert_eq!(wire.stream, None);
        // Request-level passthrough fields ride ext and come back verbatim.
        assert_eq!(wire.unknown_fields["top_k"], json!(40));
        assert_eq!(wire.unknown_fields["seed"], json!(1234));
        assert_eq!(wire.unknown_fields["vendor_beta"], json!(true));

        // Whole-value check so a future typed field can't slip past the
        // per-field assertions above.
        let mut expected = serde_json::to_value(&original).unwrap();
        expected["model"] = json!("gpt-5.6");
        expected.as_object_mut().unwrap().remove("stream");
        assert_eq!(serde_json::to_value(&wire).unwrap(), expected);
    }

    /// THE claim the retain half of this module exists for. Tools, a tool
    /// choice, a multimodal parts list, an assistant turn that called a tool
    /// and the tool message answering it all carry residue the gateway shapes
    /// have no field for — a part's `type` spelling, a vendor extension on a
    /// function, the difference between a bare string and a one-part list.
    /// All of it has to come back byte for byte.
    #[test]
    fn a_tool_calling_request_round_trips_byte_for_byte() {
        let original = wire(json!({
            "model": "openai/gpt-5.6",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "what is the weather"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png", "detail": "high"}},
                    {"type": "input_audio", "input_audio": {"data": "aGk=", "format": "wav"}}
                ]},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\": \"Seoul\"}"},
                     "vendor_call_field": 1}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "18C"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up the weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                    "strict": true,
                    "vendor_function_field": "x"
                }
            }],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "weather", "schema": {"type": "object"}, "strict": true
            }},
            "stream_options": {"include_usage": true},
            "reasoning_effort": "high"
        }));

        let normalized = ingest_request(original.clone(), "gpt-5.6");
        let rendered = render_request(&normalized, "openai").unwrap();

        let mut expected = serde_json::to_value(&original).unwrap();
        expected["model"] = json!("gpt-5.6");
        // Nothing else is permitted to differ — including the `content: null`
        // on the tool-calling assistant turn, and the `strict` nested inside
        // the tool, both of which are explicit values rather than absences.
        assert_eq!(serde_json::to_value(&rendered).unwrap(), expected);
    }

    /// ...and the same request must ALSO be readable as gateway types, which
    /// is the half that makes another protocol reachable. The retained
    /// residue above is invisible here.
    #[test]
    fn ingest_promotes_tools_and_content_into_the_gateway_form() {
        let normalized = ingest_request(
            wire(json!({
                "model": "openai/gpt-5.6",
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "what is the weather"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                    ]},
                    {"role": "assistant", "content": null, "tool_calls": [
                        {"id": "call_1", "type": "function",
                         "function": {"name": "get_weather", "arguments": "{\"city\":\"Seoul\"}"}}
                    ]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "18C"}
                ],
                "tools": [{"type": "function", "function": {
                    "name": "get_weather", "parameters": {"type": "object"}, "strict": true
                }}],
                "tool_choice": "required",
                "response_format": {"type": "json_object"},
                "stream_options": {"include_usage": true},
                "reasoning_effort": "high"
            })),
            "gpt-5.6",
        );

        assert_eq!(normalized.tools.len(), 1);
        assert_eq!(normalized.tools[0].name, "get_weather");
        assert_eq!(normalized.tools[0].input_schema, json!({"type": "object"}));
        assert_eq!(normalized.tools[0].strict, Some(true));
        assert!(matches!(
            normalized.tool_choice,
            Some(types::ToolChoice::Required)
        ));
        assert!(matches!(
            normalized.response_format,
            Some(types::ResponseFormat::JsonObject)
        ));
        assert_eq!(normalized.stream_include_usage, Some(true));
        assert_eq!(
            normalized.reasoning.as_ref().unwrap().effort.as_deref(),
            Some("high")
        );

        // A parts list becomes real blocks, and a `data:` URL is split so a
        // protocol that wants the media type apart does not re-parse it.
        match &normalized.messages[0].content[..] {
            [
                ContentBlock::Text { text, .. },
                ContentBlock::Image { source, .. },
            ] => {
                assert_eq!(text, "what is the weather");
                assert!(matches!(
                    source,
                    MediaSource::Base64 { media_type, data }
                        if media_type == "image/png" && data == "AAAA"
                ));
            }
            other => panic!("expected a text and an image block, got {other:?}"),
        }

        // The assistant's call, and the tool message answering it, both land
        // as blocks carrying the id that correlates them.
        match &normalized.messages[1].content[..] {
            [
                ContentBlock::ToolCall {
                    id,
                    name,
                    arguments,
                },
            ] => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "get_weather");
                assert_eq!(arguments.as_str(), r#"{"city":"Seoul"}"#);
            }
            other => panic!("expected one tool call, got {other:?}"),
        }
        assert_eq!(normalized.messages[2].role, Role::Tool);
        match &normalized.messages[2].content[..] {
            [
                ContentBlock::ToolResult {
                    call_id,
                    content,
                    is_error,
                },
            ] => {
                assert_eq!(call_id, "call_1");
                assert!(!is_error);
                assert!(
                    matches!(&content[..], [ContentBlock::Text { text, .. }] if text == "18C"),
                    "{content:?}"
                );
            }
            other => panic!("expected one tool result, got {other:?}"),
        }
    }

    /// The cross-protocol direction: a gateway request built somewhere else
    /// carries no `openai_compat` residue, so every field has to be rendered
    /// from the gateway form alone. This is the path a second protocol's
    /// ingest feeds, and nothing but this test exercises it today.
    #[test]
    fn a_request_with_no_residue_renders_from_the_gateway_form() {
        let request = types::ChatRequest {
            model: "gpt-5.6".into(),
            messages: vec![
                types::Message {
                    role: Role::User,
                    content: vec![
                        ContentBlock::Text {
                            text: "what is the weather".into(),
                            cache: None,
                        },
                        ContentBlock::Image {
                            source: MediaSource::Base64 {
                                media_type: "image/png".into(),
                                data: "AAAA".into(),
                            },
                            detail: Some("high".into()),
                            cache: None,
                        },
                    ],
                    ext: ProviderExt::new(),
                },
                types::Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolCall {
                        id: "toolu_01".into(),
                        name: "get_weather".into(),
                        arguments: RawJson::new(r#"{"city":"Seoul"}"#),
                    }],
                    ext: ProviderExt::new(),
                },
                types::Message {
                    role: Role::Tool,
                    content: vec![ContentBlock::ToolResult {
                        call_id: "toolu_01".into(),
                        content: vec![ContentBlock::Text {
                            text: "18C".into(),
                            cache: None,
                        }],
                        is_error: false,
                    }],
                    ext: ProviderExt::new(),
                },
            ],
            tools: vec![types::ToolDef {
                name: "get_weather".into(),
                description: Some("Look up the weather".into()),
                input_schema: json!({"type": "object"}),
                strict: None,
                cache: None,
                ext: ProviderExt::new(),
            }],
            tool_choice: Some(types::ToolChoice::Tool {
                name: "get_weather".into(),
            }),
            stream_include_usage: Some(true),
            ..Default::default()
        };

        assert_eq!(
            serde_json::to_value(render_request(&request, "openai").unwrap()).unwrap(),
            json!({
                "model": "gpt-5.6",
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "what is the weather"},
                        {"type": "image_url",
                         "image_url": {"url": "data:image/png;base64,AAAA", "detail": "high"}}
                    ]},
                    {"role": "assistant", "tool_calls": [
                        {"id": "toolu_01", "type": "function",
                         "function": {"arguments": "{\"city\":\"Seoul\"}", "name": "get_weather"}}
                    ]},
                    // The result's payload IS the message on this protocol,
                    // and the call id moves from the block onto the message.
                    {"role": "tool", "content": "18C", "tool_call_id": "toolu_01"}
                ],
                "stream_options": {"include_usage": true},
                "tools": [{"type": "function", "function": {
                    "name": "get_weather",
                    "description": "Look up the weather",
                    "parameters": {"type": "object"}
                }}],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
            })
        );
    }

    /// A hosted tool has no arguments schema, so there is nothing to render a
    /// function tool FROM. Refusing names it; emitting one with an empty
    /// schema would have the model call something that does not exist. A
    /// neutral representation for these is #47's to settle.
    #[test]
    fn a_tool_with_no_schema_is_refused_cross_protocol() {
        let request = types::ChatRequest {
            model: "gpt-5.6".into(),
            tools: vec![types::ToolDef {
                name: "web_search".into(),
                description: None,
                input_schema: Value::Null,
                strict: None,
                cache: None,
                ext: ProviderExt::new(),
            }],
            ..Default::default()
        };
        let err = render_request(&request, "openai").unwrap_err();
        assert!(matches!(err, Error::NotImplemented(_)), "{err:?}");

        // ...but the same tool arriving ON this protocol still reaches the
        // provider, because the residue renders it without this path.
        let same_protocol = ingest_request(
            wire(json!({
                "model": "openai/gpt-5.6",
                "messages": [],
                "tools": [{"type": "web_search"}]
            })),
            "gpt-5.6",
        );
        assert_eq!(
            plain(&render_request(&same_protocol, "openai").unwrap().tools),
            json!([{"type": "web_search"}])
        );
    }

    #[test]
    fn ingest_maps_typed_params_and_strips_prefix() {
        let normalized = ingest_request(full_wire_request(), "gpt-5.6");
        assert_eq!(normalized.model, "gpt-5.6");
        assert_eq!(normalized.params.temperature, Some(0.7));
        assert_eq!(normalized.params.max_tokens, Some(256));
        assert_eq!(normalized.params.top_p, Some(0.9));
        assert_eq!(normalized.params.stop, vec!["END", "STOP"]);
        assert!(!normalized.stream);
    }

    #[test]
    fn ingest_keeps_passthrough_fields_in_the_protocol_namespace() {
        let normalized = ingest_request(full_wire_request(), "gpt-5.6");
        // Every field without a typed wire home rides ext verbatim.
        assert_eq!(normalized.ext["openai_compat"]["top_k"], json!(40));
        assert_eq!(normalized.ext["openai_compat"]["seed"], json!(1234));
        assert_eq!(normalized.ext["openai_compat"]["vendor_beta"], json!(true));
        assert_eq!(
            normalized.messages[1].ext["openai_compat"]["name"],
            json!("alice")
        );
        assert_eq!(
            normalized.messages[1].ext["openai_compat"]["vendor_tag"],
            json!("vt")
        );
        // The unrecognized role is stashed for exact restoration.
        assert_eq!(
            normalized.messages[0].ext["openai_compat"]["role"],
            json!("developer")
        );
    }

    #[test]
    fn ingest_populates_source() {
        let original = full_wire_request();
        let normalized = ingest_request(original.clone(), "gpt-5.6");
        let source = normalized.source.as_ref().unwrap();
        assert_eq!(source.protocol, Protocol::OpenAiCompat);
        assert_eq!(source.body, serde_json::to_value(&original).unwrap());
    }

    #[test]
    fn render_merges_provider_ext_namespace_over_source_protocol() {
        let mut normalized = ingest_request(full_wire_request(), "deepseek-v4-flash");
        normalized.ext.insert(
            "deepseek".into(),
            json!({"vendor_beta": "ds-version", "ds_only": true}),
        );

        let for_deepseek = render_request(&normalized, "deepseek").unwrap();
        // The provider namespace overlays the protocol namespace...
        assert_eq!(
            for_deepseek.unknown_fields["vendor_beta"],
            json!("ds-version")
        );
        assert_eq!(for_deepseek.unknown_fields["ds_only"], json!(true));

        // ...and rendering for openai ignores it entirely.
        let for_openai = render_request(&normalized, "openai").unwrap();
        assert_eq!(for_openai.unknown_fields["vendor_beta"], json!(true));
        assert!(for_openai.unknown_fields.get("ds_only").is_none());
    }

    /// Rendering never filters params against what a target documents:
    /// the provider is the authority and rejects unsupported params
    /// itself.
    #[test]
    fn render_forwards_params_regardless_of_provider() {
        let normalized = ingest_request(full_wire_request(), "deepseek-v4-flash");
        let wire = render_request(&normalized, "deepseek").unwrap();
        // DeepSeek documents neither seed nor top_k; both still render.
        assert_eq!(wire.unknown_fields["seed"], json!(1234));
        assert_eq!(wire.unknown_fields["top_k"], json!(40));
    }

    #[test]
    fn empty_stop_is_omitted_and_stream_never_rendered() {
        let mut wire_request = full_wire_request();
        wire_request.stop = None;
        wire_request.stream = Some(false);
        let normalized = ingest_request(wire_request, "gpt-5.6");
        let wire = render_request(&normalized, "openai").unwrap();
        assert!(wire.stop.is_none());
        assert_eq!(wire.stream, None);
    }

    /// `file_data` is raw base64, flatly — the vendor documents it as "the
    /// base64 encoded file data", where `image_url.url` is "either a URL of
    /// the image OR the base64 encoded image data". Reading a file's bytes as
    /// a URL would hand the next protocol a `MediaSource::Url` pointing at
    /// `"aGk="`, and no residue saves that once the request came from
    /// somewhere else.
    #[test]
    fn a_files_bytes_are_raw_base64_in_both_directions() {
        let normalized = ingest_request(
            wire(json!({
                "model": "openai/gpt-5.6",
                "messages": [{"role": "user", "content": [
                    {"type": "file", "file": {"file_data": "aGk=", "filename": "a.pdf"}},
                    {"type": "file", "file": {"file_id": "file-1"}},
                    // ...while the IMAGE part's data URL is still split, since
                    // that field really does take both.
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                ]}]
            })),
            "gpt-5.6",
        );
        match &normalized.messages[0].content[..] {
            [
                ContentBlock::Document {
                    source: bytes,
                    title,
                    ..
                },
                ContentBlock::Document {
                    source: uploaded, ..
                },
                ContentBlock::Image { source: image, .. },
            ] => {
                assert!(
                    matches!(bytes, MediaSource::Base64 { media_type, data }
                        if media_type.is_empty() && data == "aGk="),
                    "{bytes:?}"
                );
                assert_eq!(title.as_deref(), Some("a.pdf"));
                assert!(matches!(uploaded, MediaSource::ProviderFile { id } if id == "file-1"));
                assert!(
                    matches!(image, MediaSource::Base64 { media_type, data }
                        if media_type == "image/png" && data == "AAAA"),
                    "{image:?}"
                );
            }
            other => panic!("expected two documents and an image, got {other:?}"),
        }

        // ...and back out, from a request with no residue to restore from.
        let cross_protocol = types::ChatRequest {
            model: "gpt-5.6".into(),
            messages: vec![types::Message {
                role: Role::User,
                content: vec![ContentBlock::Document {
                    source: MediaSource::Base64 {
                        media_type: "application/pdf".into(),
                        data: "aGk=".into(),
                    },
                    title: Some("a.pdf".into()),
                    cache: None,
                }],
                ext: ProviderExt::new(),
            }],
            ..Default::default()
        };
        assert_eq!(
            plain(&render_request(&cross_protocol, "openai").unwrap().messages),
            json!([{"role": "user", "content": [
                // Raw, not `data:application/pdf;base64,aGk=`. The media type
                // is dropped because the file part declares no field for one.
                {"type": "file", "file": {"file_data": "aGk=", "filename": "a.pdf"}}
            ]}])
        );
    }

    /// Anthropic merges a run of consecutive tool results into ONE `user`
    /// turn; this protocol carries exactly one result per message. So a
    /// grouped turn has to split back into one message each, and the count of
    /// wire messages stops matching the count of gateway ones.
    ///
    /// Rendering only the first result would leave the model's other calls
    /// looking unanswered — and a model told its tool never replied does not
    /// fail, it guesses.
    #[test]
    fn a_grouped_turn_splits_into_one_message_per_tool_result() {
        let result = |call_id: &str, text: &str| ContentBlock::ToolResult {
            call_id: call_id.into(),
            content: vec![ContentBlock::Text {
                text: text.into(),
                cache: None,
            }],
            is_error: false,
        };
        let request = types::ChatRequest {
            model: "gpt-5.6".into(),
            messages: vec![
                types::Message {
                    role: Role::Assistant,
                    content: vec![
                        ContentBlock::ToolCall {
                            id: "toolu_01".into(),
                            name: "weather".into(),
                            arguments: RawJson::new(r#"{"city":"Seoul"}"#),
                        },
                        ContentBlock::ToolCall {
                            id: "toolu_02".into(),
                            name: "weather".into(),
                            arguments: RawJson::new(r#"{"city":"Tokyo"}"#),
                        },
                    ],
                    ext: ProviderExt::new(),
                },
                // The merged turn: two results, one message.
                types::Message {
                    role: Role::Tool,
                    content: vec![result("toolu_01", "18C"), result("toolu_02", "24C")],
                    ext: ProviderExt::new(),
                },
            ],
            ..Default::default()
        };

        let rendered = render_request(&request, "openai").unwrap();
        assert_eq!(
            plain(&rendered.messages),
            json!([
                {"role": "assistant", "tool_calls": [
                    {"id": "toolu_01", "type": "function",
                     "function": {"arguments": "{\"city\":\"Seoul\"}", "name": "weather"}},
                    {"id": "toolu_02", "type": "function",
                     "function": {"arguments": "{\"city\":\"Tokyo\"}", "name": "weather"}}
                ]},
                {"role": "tool", "content": "18C", "tool_call_id": "toolu_01"},
                {"role": "tool", "content": "24C", "tool_call_id": "toolu_02"}
            ]),
            "every call the assistant made must have an answer"
        );

        // Two gateway messages became three wire ones, which is the point.
        assert_eq!(rendered.messages.len(), 3);
    }

    /// A tool that FAILED is a different fact than a tool that returned an
    /// unhelpful answer, and the model acts on the difference. Chat
    /// completions has no bit for it, and the only way to convey it would be
    /// to edit the content — putting words in the tool's mouth its author
    /// never wrote.
    ///
    /// The ordinary case has to keep working, which is the other half of this:
    /// `is_error: false` carries no information and must not be refused, or
    /// every tool result crossing protocols would be.
    #[test]
    fn a_failed_tool_result_is_refused_rather_than_flattened() {
        let answered = |is_error| types::ChatRequest {
            model: "gpt-5.6".into(),
            messages: vec![types::Message {
                role: Role::Tool,
                content: vec![ContentBlock::ToolResult {
                    call_id: "toolu_01".into(),
                    content: vec![ContentBlock::Text {
                        text: "the city was not found".into(),
                        cache: None,
                    }],
                    is_error,
                }],
                ext: ProviderExt::new(),
            }],
            ..Default::default()
        };

        let err = render_request(&answered(true), "openai").unwrap_err();
        assert!(matches!(err, Error::NotImplemented(_)), "{err:?}");
        assert!(err.to_string().contains("failed tool result"), "{err}");

        // The same text, not marked as a failure, is an ordinary result.
        assert_eq!(
            plain(&render_request(&answered(false), "openai").unwrap().messages),
            json!([{
                "role": "tool",
                "content": "the city was not found",
                "tool_call_id": "toolu_01"
            }])
        );
    }

    /// A cache hint has no spelling on this protocol, so rendering it as
    /// silence would change the request rather than translate it — the caller
    /// asked for a breakpoint and would get an ordinary completion with
    /// nothing saying otherwise. Every level that can carry one is checked,
    /// because widening the renderer removed the gate that used to cover them.
    #[test]
    fn a_cache_hint_is_refused_rather_than_dropped() {
        let hint = || {
            Some(types::CacheHint {
                ttl: Some("5m".into()),
            })
        };
        let text = |cache| types::Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "hi".into(),
                cache,
            }],
            ext: ProviderExt::new(),
        };
        let tool = |cache| types::ToolDef {
            name: "f".into(),
            description: None,
            input_schema: json!({"type": "object"}),
            strict: None,
            cache,
            ext: ProviderExt::new(),
        };

        for request in [
            // On the request...
            types::ChatRequest {
                cache: hint(),
                ..Default::default()
            },
            // ...on a block...
            types::ChatRequest {
                messages: vec![text(hint())],
                ..Default::default()
            },
            // ...on a block nested inside a tool result...
            types::ChatRequest {
                messages: vec![types::Message {
                    role: Role::Tool,
                    content: vec![ContentBlock::ToolResult {
                        call_id: "c".into(),
                        content: vec![ContentBlock::Text {
                            text: "18C".into(),
                            cache: hint(),
                        }],
                        is_error: false,
                    }],
                    ext: ProviderExt::new(),
                }],
                ..Default::default()
            },
            // ...and on a tool.
            types::ChatRequest {
                tools: vec![tool(hint())],
                ..Default::default()
            },
        ] {
            let err = render_request(&request, "openai").unwrap_err();
            assert!(matches!(err, Error::NotImplemented(_)), "{err:?}");
            assert!(err.to_string().contains("cache"), "{err}");
        }

        // The same shapes without hints still render, so the check is not one
        // that refuses everything.
        render_request(
            &types::ChatRequest {
                messages: vec![text(None)],
                tools: vec![tool(None)],
                ..Default::default()
            },
            "openai",
        )
        .unwrap();
    }

    /// `reasoning_effort` is the whole of this protocol's reasoning
    /// vocabulary. A budget, an exclusion, or a bare "reason" with no effort
    /// to render has nowhere to go, and dropping them would answer a request
    /// nobody made.
    #[test]
    fn an_unrenderable_reasoning_field_is_refused() {
        let with = |reasoning| types::ChatRequest {
            reasoning: Some(reasoning),
            ..Default::default()
        };
        for reasoning in [
            types::ReasoningConfig {
                budget_tokens: Some(4096),
                ..Default::default()
            },
            types::ReasoningConfig {
                exclude: Some(true),
                ..Default::default()
            },
            types::ReasoningConfig {
                enabled: Some(true),
                ..Default::default()
            },
            // Even alongside an effort, a budget is a number this protocol
            // cannot state.
            types::ReasoningConfig {
                enabled: Some(true),
                effort: Some("high".into()),
                budget_tokens: Some(4096),
                ..Default::default()
            },
        ] {
            let err = render_request(&with(reasoning), "openai").unwrap_err();
            assert!(matches!(err, Error::NotImplemented(_)), "{err:?}");
            assert!(err.to_string().contains("reasoning"), "{err}");
        }

        // An effort alone is exactly what this protocol says, with or without
        // the enablement that effort already implies.
        for reasoning in [
            types::ReasoningConfig {
                effort: Some("high".into()),
                ..Default::default()
            },
            types::ReasoningConfig {
                enabled: Some(true),
                effort: Some("high".into()),
                ..Default::default()
            },
        ] {
            assert_eq!(
                render_request(&with(reasoning), "openai")
                    .unwrap()
                    .reasoning_effort,
                Some(Some("high".into()))
            );
        }
    }

    /// The legacy `role: "function"` message is where absent and `null`
    /// genuinely differ: its schema is `content: string | null`, so the key is
    /// REQUIRED and may hold null. A form that read the null as an absence
    /// would forward `{"role": "function", "name": "f"}` with the key gone,
    /// and a provider holding OpenAI to its own schema may reject that.
    #[test]
    fn a_null_content_on_a_function_message_keeps_its_key() {
        let normalized = ingest_request(
            wire(json!({
                "model": "openai/gpt-5.6",
                "messages": [
                    {"role": "function", "name": "f", "content": null},
                    // ...while an assistant turn may legitimately omit it, and
                    // must not gain one it never had.
                    {"role": "assistant", "tool_calls": [
                        {"id": "call_1", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}
                    ]}
                ]
            })),
            "gpt-5.6",
        );
        let messages = plain(&render_request(&normalized, "openai").unwrap().messages);
        assert_eq!(messages[0]["content"], Value::Null, "{messages}");
        assert!(
            messages[0].get("content").is_some(),
            "the key a function message requires was dropped: {messages}"
        );
        assert!(
            messages[1].get("content").is_none(),
            "an absent content came back as a null: {messages}"
        );
    }

    /// Typing a field is what makes it translatable, and it must not cost the
    /// passthrough that field had while it was untyped. Every one of these
    /// reached a provider verbatim as an `unknown_fields` entry before it had
    /// a home; an explicit `null` still has to arrive.
    ///
    /// `strict` is the case that says the fix had to be in the TYPES: `retain`
    /// stores the parsed value re-serialized, so a null the type collapsed on
    /// the way in is gone before retention ever runs.
    #[test]
    fn an_explicit_null_survives_on_every_nullable_field() {
        let body = json!({
            "model": "openai/gpt-5.6",
            "messages": [],
            "temperature": null,
            "max_tokens": null,
            "top_p": null,
            "stop": null,
            "stream_options": null,
            "reasoning_effort": null,
            "tools": [{"type": "function", "function": {
                "name": "f", "parameters": {"type": "object"}, "strict": null
            }}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "r", "schema": {"type": "object"}, "strict": null
            }}
        });
        let normalized = ingest_request(wire(body.clone()), "gpt-5.6");
        let rendered = plain(&render_request(&normalized, "openai").unwrap());

        let mut expected = body;
        expected["model"] = json!("gpt-5.6");
        assert_eq!(rendered, expected);
    }

    /// `stop` is one sequence or several, and OpenAI's own examples show the
    /// bare string. Both have to be read, both have to reach the gateway form
    /// as the same list, and each has to render back as the form it arrived
    /// in — a request that said `"END"` is not one that said `["END"]`.
    #[test]
    fn a_bare_stop_string_is_read_and_rendered_back_as_one() {
        for (wire_stop, sequences) in [
            (json!("END"), vec!["END".to_string()]),
            (json!(["END", "STOP"]), vec!["END".into(), "STOP".into()]),
        ] {
            let original = wire(json!({
                "model": "openai/gpt-5.6",
                "messages": [],
                "stop": wire_stop,
            }));
            let normalized = ingest_request(original, "gpt-5.6");
            assert_eq!(normalized.params.stop, sequences);
            assert_eq!(
                plain(&render_request(&normalized, "openai").unwrap().stop),
                wire_stop
            );
        }
    }
}
