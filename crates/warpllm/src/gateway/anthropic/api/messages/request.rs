//! Request conversions: Anthropic's wire request → gateway (ingest) and
//! gateway → wire (render).
//!
//! # Promote but retain
//!
//! Every typed field lifted into the gateway form is ALSO kept in
//! `ext["anthropic"]` exactly as it arrived, and [`render_request`] prefers the
//! retained value over re-rendering. The reason is the same as its
//! openai_compat counterpart's and bites harder here, because the gateway
//! blocks have no `ext` of their own: a block's `cache_control` type, a
//! source's unknown fields, the difference between `"hi"` and
//! `[{"type":"text",…}]` — none has anywhere to go except the retained
//! `content` of the message that held it.
//!
//! So a request that arrives on this protocol and leaves on it is byte-exact,
//! and one that arrives on ANOTHER protocol has no residue and is rendered from
//! the gateway form. The second path is the one that matters today: warpllm is
//! called in OpenAI's protocol and speaks this one to the provider.
//!
//! # What this protocol arranges differently, and where each is handled
//!
//! | | chat completions | Anthropic | handled by |
//! |---|---|---|---|
//! | system prompt | a message with `role: "system"` | a top-level `system` | [`render_messages`] hoists it |
//! | tool result | one message per result | `tool_result` blocks in a user turn | [`render_messages`] merges a run |
//! | tool arguments | a string of JSON | a decoded object | [`render_block`], which can fail |
//! | reasoning depth | `reasoning_effort` | `output_config.effort` | [`render_output_config`] |
//! | structured output | `response_format` | `output_config.format` | [`render_output_format`] |
//! | `max_tokens` | optional | REQUIRED | [`resolve_max_tokens`] |

use serde_json::Value;

use crate::error::{Error, Result};
use crate::gateway::anthropic::{
    cache_control, merged_ext, namespaced, role_from_wire, role_to_wire,
};
use crate::gateway::types::{self, ContentBlock, IngestSource, MediaSource, RawJson, Role};
use crate::protocol::UnknownFields;
use crate::protocol::anthropic::messages::types::{
    Base64Source, ContentBlock as WireBlock, CreateMessageRequest, CustomTool, DocumentBlock,
    ImageBlock, InputMessage, MessageContent, OutputConfig, OutputFormat, Source, SystemPrompt,
    TextBlock, ThinkingAdaptive, ThinkingConfig, ThinkingDisabled, ThinkingEnabled, Tool,
    ToolChoice, ToolChoiceMode, ToolChoiceTool, ToolResultBlock, ToolResultContent, ToolUseBlock,
    UrlSource,
};
use crate::types::Protocol;

// `render_source` and `render_result_content` are deliberately NOT imported.
// The reply has its own infallible pair and this module has a fallible one; the
// two are not interchangeable, and borrowing the reply's is what let a foreign
// file id and a nested unknown block past this module's refusals. See
// [`render_block`].
use super::response::{control_of, ingest_block, object, plain, render_reasoning, take_typed};

/// Permissive and infallible: capture, don't validate. `model` is the
/// prefix-stripped name.
///
/// The destructuring is exhaustive (no `..`) so that adding a typed field to
/// the wire request without mapping it here is a compile error. Everything
/// without a typed wire field — `top_k`, `metadata`, `service_tier` — rides
/// `ext["anthropic"]` verbatim.
pub(crate) fn ingest_request(request: CreateMessageRequest, model: &str) -> types::ChatRequest {
    // Wire structs are plain serde data; serialization cannot fail.
    let body = plain(&request);
    let CreateMessageRequest {
        model: _,
        messages,
        max_tokens,
        system,
        temperature,
        top_p,
        stop_sequences,
        stream,
        tools,
        tool_choice,
        thinking,
        output_config,
        unknown_fields,
    } = request;

    let mut anthropic = unknown_fields;
    retain(&mut anthropic, "system", system.as_ref());
    retain(&mut anthropic, "stop_sequences", stop_sequences.as_ref());
    retain(&mut anthropic, "tools", tools.as_ref());
    retain(&mut anthropic, "tool_choice", tool_choice.as_ref());
    retain(&mut anthropic, "thinking", thinking.as_ref());
    retain(&mut anthropic, "output_config", output_config.as_ref());

    let effort = output_config
        .as_ref()
        .and_then(|config| config.effort.as_deref());
    types::ChatRequest {
        model: model.to_string(),
        // The system prompt becomes a LEADING message, which is the gateway's
        // arrangement — `ChatRequest::messages` documents system messages as
        // included and hoisted per target. Rendering hoists it back out.
        messages: system
            .iter()
            .map(ingest_system)
            .chain(messages.into_iter().map(ingest_message))
            .collect(),
        tools: tools
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(ingest_tool)
            .collect(),
        tool_choice: tool_choice.as_ref().and_then(ingest_tool_choice),
        params: types::GenerationParams {
            max_tokens: Some(max_tokens),
            temperature,
            top_p,
            stop: stop_sequences.unwrap_or_default(),
        },
        response_format: output_config
            .as_ref()
            .and_then(|config| config.format.as_ref())
            .and_then(ingest_output_format),
        reasoning: ingest_reasoning(thinking.as_ref(), effort),
        // Anthropic's cache breakpoints are POSITIONAL — each sits on the block
        // it marks — so there is no request-level hint to lift. The hints ride
        // the blocks, where `ingest_block` puts them.
        cache: None,
        stream: stream.unwrap_or(false),
        // No wire counterpart, and `None` would be the wrong reading of that
        // for a STREAMED request: Anthropic reports a stream's totals
        // unconditionally, so a caller who asked for a stream asked for them
        // by construction. Saying so is what keeps the counts in the gateway
        // form on this protocol's own path, and what makes an OpenAI backend
        // send the trailing usage chunk this protocol requires when a request
        // crosses the other way.
        //
        // Gated on `stream` because the field is meaningless without one, and
        // not merely untidy: chat completions defines `stream_options` only
        // alongside `stream: true` — "Only set this when you set stream:
        // true", per the vendor's own SDK — and a stricter compatible API
        // refuses the pair. A non-streamed request has no totals to ask for.
        stream_include_usage: stream.unwrap_or(false).then_some(true),
        ext: namespaced(anthropic),
        source: Some(IngestSource {
            protocol: Protocol::Anthropic,
            body,
        }),
    }
}

/// Keeps a typed field's wire value under this protocol's namespace, so
/// [`render_request`] can hand back exactly what arrived.
fn retain<T: serde::Serialize>(fields: &mut UnknownFields, key: &str, value: Option<&T>) {
    if let Some(value) = value {
        fields.insert(key.to_string(), plain(value));
    }
}

/// The top-level `system` as the gateway's leading system message. Which of the
/// two wire forms it arrived in is retained at request level, so nothing is
/// decided here that render cannot undo.
fn ingest_system(prompt: &SystemPrompt) -> types::Message {
    types::Message {
        role: Role::System,
        content: match prompt {
            SystemPrompt::Text(text) => vec![ContentBlock::Text {
                text: text.clone(),
                cache: None,
            }],
            SystemPrompt::Blocks(blocks) => blocks.iter().map(ingest_block).collect(),
        },
        ext: types::ProviderExt::new(),
    }
}

/// A wire turn as a gateway message.
///
/// The one judgement here is [`Role::Tool`]. Anthropic has no tool role — a
/// result is a `tool_result` block inside a USER turn — so a turn carrying
/// nothing but results is exactly what the gateway calls a tool message, and
/// reading it as an ordinary user turn would leave a renderer for a protocol
/// with a tool role unable to find one. A MIXED turn stays `User`: its results
/// are blocks among others, and `Role::Tool`'s contract is results only.
fn ingest_message(message: InputMessage) -> types::Message {
    let InputMessage {
        role,
        content,
        unknown_fields,
    } = message;
    let (role, raw_role) = role_from_wire(role);
    let mut anthropic = unknown_fields;
    if let Some(raw) = raw_role {
        anthropic.insert("role".into(), Value::String(raw));
    }
    anthropic.insert("content".into(), plain(&content));

    let blocks: Vec<ContentBlock> = match &content {
        MessageContent::Text(text) => vec![ContentBlock::Text {
            text: text.clone(),
            cache: None,
        }],
        MessageContent::Blocks(blocks) => blocks.iter().map(ingest_block).collect(),
    };
    let only_results = !blocks.is_empty()
        && blocks
            .iter()
            .all(|block| matches!(block, ContentBlock::ToolResult { .. }));
    types::Message {
        role: if role == Role::User && only_results {
            Role::Tool
        } else {
            role
        },
        content: blocks,
        ext: namespaced(anthropic),
    }
}

fn ingest_tool(tool: &Tool) -> types::ToolDef {
    match tool {
        Tool::Custom(custom) => types::ToolDef {
            name: custom.name.clone(),
            description: custom.description.clone(),
            // Verbatim: Anthropic spells this `input_schema` where chat
            // completions spells it `parameters`, and the name is the only
            // difference.
            input_schema: custom.input_schema.clone(),
            strict: custom.strict,
            cache: custom
                .cache_control
                .as_ref()
                .and_then(Option::as_ref)
                .map(crate::gateway::anthropic::cache_hint),
            ext: namespaced(custom.unknown_fields.clone()),
        },
        // A server tool: no arguments schema exists to translate, so it keeps
        // its name and carries the whole wire object in ext. That is what lets
        // a renderer for another protocol refuse it BY NAME rather than send
        // something invented. #54 owns a neutral representation.
        Tool::Other(value) => types::ToolDef {
            name: value["name"]
                .as_str()
                .or_else(|| value["type"].as_str())
                .unwrap_or_default()
                .to_string(),
            description: None,
            input_schema: Value::Null,
            strict: None,
            cache: None,
            ext: namespaced(object(value.clone())),
        },
    }
}

fn ingest_tool_choice(choice: &ToolChoice) -> Option<types::ToolChoice> {
    match choice {
        ToolChoice::Auto(_) => Some(types::ToolChoice::Auto),
        ToolChoice::None(_) => Some(types::ToolChoice::None),
        // "any tool" is Anthropic's spelling of "required".
        ToolChoice::Any(_) => Some(types::ToolChoice::Required),
        ToolChoice::Tool(tool) => Some(types::ToolChoice::Tool {
            name: tool.name.clone(),
        }),
        // A shape this gateway has no word for. It still reaches an Anthropic
        // target through the retained residue; there is just nothing to tell
        // another protocol.
        ToolChoice::Unknown(_) => None,
    }
}

fn ingest_output_format(format: &OutputFormat) -> Option<types::ResponseFormat> {
    // `name` and `strict` are chat-completions labels with no Anthropic
    // counterpart; a schema is the whole of what this protocol says.
    (format.r#type == "json_schema").then(|| types::ResponseFormat::JsonSchema {
        name: String::new(),
        schema: format.schema.clone().unwrap_or(Value::Null),
        strict: None,
    })
}

/// `thinking` and `output_config.effort` are ONE gateway concept split across
/// two Anthropic parameters, which is why this reads both.
fn ingest_reasoning(
    thinking: Option<&ThinkingConfig>,
    effort: Option<&str>,
) -> Option<types::ReasoningConfig> {
    let mut config = types::ReasoningConfig {
        effort: effort.map(str::to_string),
        ..Default::default()
    };
    match thinking {
        Some(ThinkingConfig::Enabled(enabled)) => {
            config.enabled = Some(true);
            config.budget_tokens = Some(enabled.budget_tokens);
            config.exclude = exclusion(enabled.display.as_deref());
        }
        Some(ThinkingConfig::Adaptive(adaptive)) => {
            config.enabled = Some(true);
            config.exclude = exclusion(adaptive.display.as_deref());
        }
        Some(ThinkingConfig::Disabled(_)) => config.enabled = Some(false),
        // A thinking mode warpllm does not model reaches Anthropic through the
        // retained residue; there is nothing to promote.
        Some(ThinkingConfig::Unknown(_)) | None => {}
    }
    (config.enabled.is_some() || config.effort.is_some()).then_some(config)
}

/// `display` as the gateway's "reason but don't return it".
///
/// Anthropic's field says what the CALLER sees, not whether the model thinks:
/// `"omitted"` still bills the thinking and still requires the blocks back
/// unchanged next turn. That is exactly `ReasoningConfig::exclude`.
fn exclusion(display: Option<&str>) -> Option<bool> {
    match display {
        Some("omitted") => Some(true),
        Some("summarized") => Some(false),
        _ => None,
    }
}

/// Renders the gateway request onto this protocol's wire.
///
/// Takes a third argument its openai_compat counterpart does not, and this is
/// the reason: `max_tokens` is REQUIRED here and optional in the gateway form,
/// so a ceiling has to come from somewhere — see [`resolve_max_tokens`]. The
/// invariant every protocol shares is gateway types on both ends, not an
/// identical argument list.
///
/// Mechanical otherwise: nothing is filtered against what the target documents.
/// The provider is the authority on its own parameters and rejects what it does
/// not accept — including the two thinking arms, which are mutually exclusive
/// PER MODEL rather than per protocol. Guessing which one the routed model
/// takes would mean either inventing a token budget for a model that wants none
/// or discarding one the caller wrote; Anthropic's own 400 says which, by name.
///
/// `stream` is emitted only when true, matching the transport's contract: it
/// CHECKS the flag rather than setting it, so a streamed exchange must render
/// `Some(true)` and a whole-reply exchange must not.
pub(crate) fn render_request(
    request: &types::ChatRequest,
    provider: &'static str,
    max_output_tokens: Option<u32>,
) -> Result<CreateMessageRequest> {
    ensure_renderable(request)?;
    let mut unknown_fields = merged_ext(&request.ext, provider);
    let params = &request.params;
    let (system, messages) = render_messages(&request.messages, provider)?;
    // Hoisted rather than inlined below: rendering a tool can fail, and `?` has
    // nowhere to go inside an `or_else` closure returning `Option`.
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
    Ok(CreateMessageRequest {
        model: request.model.clone(),
        max_tokens: resolve_max_tokens(request, max_output_tokens)?,
        messages,
        // Each of these prefers what arrived over what the gateway form can
        // reconstruct; see this module's docs.
        system: take_typed(&mut unknown_fields, "system").or(system),
        temperature: params.temperature,
        top_p: params.top_p,
        stop_sequences: take_typed(&mut unknown_fields, "stop_sequences")
            .or_else(|| (!params.stop.is_empty()).then(|| params.stop.clone())),
        stream: request.stream.then_some(true),
        tools,
        tool_choice: take_typed(&mut unknown_fields, "tool_choice")
            .or_else(|| request.tool_choice.as_ref().map(render_tool_choice)),
        thinking: take_typed(&mut unknown_fields, "thinking")
            .or_else(|| request.reasoning.as_ref().and_then(render_thinking)),
        output_config: render_output_config(request, &mut unknown_fields)?,
        unknown_fields,
    })
}

/// The ceiling Anthropic requires, from the caller or from the roster.
///
/// Never an invented default. A number picked here would silently truncate
/// replies at a length nobody asked for, and the failure would look like the
/// model stopping early rather than like a missing parameter — so an
/// unresolvable ceiling names the model and refuses.
fn resolve_max_tokens(request: &types::ChatRequest, max_output_tokens: Option<u32>) -> Result<u32> {
    request
        .params
        .max_tokens
        .or(max_output_tokens)
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "'{}' needs a max_tokens: Anthropic requires one and the roster \
                 documents no output ceiling for this model",
                request.model
            ))
        })
}

/// What the gateway form can hold and this protocol cannot say.
///
/// Short, because Anthropic says most of it: tools, tool choices, structured
/// output, cache breakpoints and a FAILED tool result all render. That last one
/// is the mirror image of the openai_compat gate, which refuses `is_error`
/// because chat completions has no bit for it.
///
/// Rendering any of these as silence would CHANGE the request rather than
/// translate it, which is the one thing a renderer must not do.
///
/// Per-value refusals are not here — a tool with no schema, an unparseable
/// argument string, a JSON mode with no schema — because each is decided where
/// the value is converted and would have to be found twice to be checked here.
fn ensure_renderable(request: &types::ChatRequest) -> Result<()> {
    // No audio block exists on this protocol, in any shape. Dropping one would
    // send a request that reads as if the caller never attached the audio, and
    // the model would answer confidently about nothing.
    if blocks(request).any(|block| matches!(block, ContentBlock::Audio { .. })) {
        return Err(Error::NotImplemented(
            "an audio block on the anthropic renderer",
        ));
    }
    // Anthropic's breakpoints are positional and BLOCK-level, which the gateway
    // models directly; a request-level hint means "put one on the last
    // cacheable block", and choosing that block is a policy nothing has asked
    // for yet. Block-level hints render normally.
    if request.cache.is_some() {
        return Err(Error::NotImplemented(
            "a request-level cache hint on the anthropic renderer",
        ));
    }
    // Anthropic's base64 source REQUIRES a media type — `application/pdf`,
    // `image/png` — and the gateway form can arrive without one: chat
    // completions' `file` part carries raw bytes and a filename and declares no
    // MIME type at all, so its ingest fills an empty string rather than a
    // guess. `openai_compat`'s own comment says a target that requires one "has
    // to say so"; this is it saying so. Sending the empty string builds a
    // request Anthropic is certain to reject, and the caller would read that
    // 400 as being about their document rather than about the conversion.
    if blocks(request).any(|block| matches!(source_of(block), Some(MediaSource::Base64 { media_type, .. }) if media_type.is_empty()))
    {
        return Err(Error::NotImplemented(
            "inline bytes with no media type on the anthropic renderer",
        ));
    }
    if let Some(reasoning) = &request.reasoning {
        // `display` is the only way to say "reason but do not return it", and
        // it is invalid without a mode to attach to — Anthropic rejects it on
        // `disabled`, and nothing is emitted at all when neither a budget nor
        // `enabled` was asked for.
        if reasoning.exclude.is_some()
            && !matches!(
                thinking_arm(reasoning),
                Some(Arm::Enabled(_) | Arm::Adaptive)
            )
        {
            return Err(Error::NotImplemented(
                "a reasoning exclusion with no thinking mode on the anthropic renderer",
            ));
        }
    }
    Ok(())
}

/// Where a block's bytes come from, for the blocks that have bytes.
fn source_of(block: &ContentBlock) -> Option<&MediaSource> {
    match block {
        ContentBlock::Image { source, .. }
        | ContentBlock::Document { source, .. }
        | ContentBlock::Audio { source, .. } => Some(source),
        _ => None,
    }
}

/// Every block in a request, tool-result payloads included — a nested block is
/// as unrenderable as a top-level one.
fn blocks(request: &types::ChatRequest) -> impl Iterator<Item = &ContentBlock> {
    request
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .flat_map(|block| {
            let nested = match block {
                ContentBlock::ToolResult { content, .. } => content.as_slice(),
                _ => &[],
            };
            std::iter::once(block).chain(nested)
        })
}

/// Splits the gateway's one message list into this protocol's two: the
/// top-level `system` parameter and the turns.
///
/// A system message AFTER the conversation has started is refused rather than
/// moved. Anthropic has one system slot and it is read before everything else,
/// so hoisting a later instruction to the front would silently reorder the
/// conversation — the model would see mid-conversation guidance as if it had
/// been there from the start — and rendering it as a user turn would fabricate
/// a turn the caller never sent. Neither is a translation.
fn render_messages(
    messages: &[types::Message],
    provider: &str,
) -> Result<(Option<SystemPrompt>, Vec<InputMessage>)> {
    let leading = messages
        .iter()
        .take_while(|message| message.role == Role::System)
        .count();
    if messages[leading..]
        .iter()
        .any(|message| message.role == Role::System)
    {
        return Err(Error::NotImplemented(
            "a system message after the conversation has started on the anthropic renderer",
        ));
    }

    let mut turns = Vec::with_capacity(messages.len() - leading);
    let mut rest = &messages[leading..];
    while let Some(first) = rest.first() {
        // Consecutive tool results become ONE user turn holding all of them —
        // the merge that `Role::Tool`'s own doc describes. Rendering them as
        // separate turns would leave a protocol that sends one result per
        // message unable to answer more than one call, and rendering only the
        // first would leave every other call the model made looking unanswered.
        //
        // ...but only for messages that do NOT already describe a turn of their
        // own. A tool message ingested from THIS protocol was a whole wire turn
        // and retained it; merging two of those would rewrite boundaries the
        // caller drew and drop the residue of every turn after the first. What
        // arrived as two turns goes back as two turns, and whether Anthropic
        // likes that arrangement is Anthropic's to say — warpllm's job is not
        // to quietly redraw it.
        //
        // The retained content IS the provenance test, and exactly the right
        // one: `merged_ext` reads only this protocol's namespace, so a tool
        // message from chat completions has no `content` here however much
        // residue it carries under its own name.
        let run = match first.role {
            Role::Tool if !describes_a_whole_turn(first, provider) => rest
                .iter()
                .take_while(|message| {
                    message.role == Role::Tool && !describes_a_whole_turn(message, provider)
                })
                .count(),
            _ => 1,
        };
        turns.push(render_turn(&rest[..run], provider)?);
        rest = &rest[run..];
    }
    Ok((render_system(&messages[..leading])?, turns))
}

/// Whether this message already carries the wire turn it came from, which is
/// what makes it ineligible to be merged into a neighbour's.
fn describes_a_whole_turn(message: &types::Message, provider: &str) -> bool {
    merged_ext(&message.ext, provider).contains_key("content")
}

/// The leading system messages as the top-level parameter. A single plain text
/// block renders as the bare string form, which is what a system prompt almost
/// always is; anything else renders as blocks.
fn render_system(messages: &[types::Message]) -> Result<Option<SystemPrompt>> {
    let blocks: Vec<&ContentBlock> = messages
        .iter()
        .flat_map(|message| &message.content)
        .collect();
    Ok(match blocks.as_slice() {
        [] => None,
        [ContentBlock::Text { text, cache: None }] => Some(SystemPrompt::Text(text.clone())),
        _ => {
            let mut rendered = Vec::with_capacity(blocks.len());
            for block in blocks {
                rendered.extend(render_block(block)?);
            }
            Some(SystemPrompt::Blocks(rendered))
        }
    })
}

/// One or more gateway messages as the single wire turn they make.
///
/// A run of length one replays its retained wire content when it has one. A
/// longer run is a merge, and [`render_messages`] only ever builds one out of
/// messages that carry NO retained content — so nothing is being overridden
/// here, and the first message's other residue rides the merged turn rather
/// than being copied onto turns that no longer exist.
///
/// That split is load-bearing rather than incidental: without it, two native
/// tool-result turns would merge into one and every turn after the first would
/// lose its residue, while still passing any test that only checked a single
/// turn.
fn render_turn(run: &[types::Message], provider: &str) -> Result<InputMessage> {
    let mut unknown_fields = merged_ext(&run[0].ext, provider);
    let role = match unknown_fields.remove("role") {
        Some(Value::String(raw)) => raw,
        _ => role_to_wire(run[0].role).to_string(),
    };
    let retained = take_typed(&mut unknown_fields, "content");
    if let ([_], Some(content)) = (run, retained) {
        return Ok(InputMessage {
            role,
            content,
            unknown_fields,
        });
    }
    let mut blocks = Vec::new();
    for message in run {
        for block in &message.content {
            blocks.extend(render_block(block)?);
        }
    }
    Ok(InputMessage {
        role,
        content: render_content(blocks),
        unknown_fields,
    })
}

/// A lone plain text block renders as the bare string form; anything else
/// renders as blocks. The two must stay distinguishable — collapsing the string
/// into a one-element list is lossless in meaning and still fails a
/// byte-for-byte round trip.
fn render_content(blocks: Vec<WireBlock>) -> MessageContent {
    match blocks.as_slice() {
        [WireBlock::Text(text)]
            if text.cache_control.is_none() && text.unknown_fields.is_empty() =>
        {
            MessageContent::Text(text.text.clone())
        }
        _ => MessageContent::Blocks(blocks),
    }
}

/// A gateway block as this protocol's, where one exists.
///
/// `None` means the block has no shape here and is dropped: reasoning another
/// provider produced, which Anthropic could not verify. Audio is the other
/// unrenderable block and never reaches this function — [`ensure_renderable`]
/// refuses the whole request first, because dropping content the caller
/// attached is not the same kind of loss as dropping a signature that would
/// have been rejected anyway.
fn render_block(block: &ContentBlock) -> Result<Option<WireBlock>> {
    Ok(match block {
        ContentBlock::Text { text, cache } => Some(WireBlock::Text(TextBlock {
            text: text.clone(),
            cache_control: control_of(cache),
            unknown_fields: UnknownFields::new(),
        })),
        ContentBlock::Image { source, cache, .. } => Some(WireBlock::Image(ImageBlock {
            source: render_source(source)?,
            // `detail` is a resolution hint chat completions has and Anthropic
            // does not. Dropping it changes nothing the model is shown — the
            // image is sent either way — which is why it is not a refusal.
            cache_control: control_of(cache),
            unknown_fields: UnknownFields::new(),
        })),
        ContentBlock::Document {
            source,
            title,
            cache,
        } => Some(WireBlock::Document(DocumentBlock {
            source: render_source(source)?,
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
            // THE lossy seam in the tool path, and the one place it can fail.
            // Chat completions carries arguments as text that models do not
            // always make valid JSON; this field is an object and cannot hold
            // text that is not. There is nothing honest to send, so it names
            // the call and the text.
            //
            // Parsing is not enough: `null`, `"a"`, `7` and `[1]` are all valid
            // JSON and none is an OBJECT, which is what `input` is documented to
            // be and what Anthropic requires. Accepting them would move the
            // failure to a 400 that names neither the call nor the text.
            input: render_tool_input(arguments, id, name)?,
            cache_control: None,
            unknown_fields: UnknownFields::new(),
        })),
        ContentBlock::ToolResult {
            call_id,
            content,
            is_error,
        } => Some(WireBlock::ToolResult(ToolResultBlock {
            tool_use_id: call_id.clone(),
            content: Some(render_result_content(content)?),
            // The bit chat completions has no way to say. `false` carries no
            // information and stays absent; only a result that FAILED emits it.
            is_error: is_error.then_some(true),
            cache_control: None,
            unknown_fields: UnknownFields::new(),
        })),
        ContentBlock::Reasoning {
            detail, provenance, ..
        } => render_reasoning(detail, provenance.as_deref()),
        // A block this gateway has no shape for, which is NOT the same as one
        // Anthropic would accept. Every producer of it is another protocol
        // saying something this one cannot: chat completions turns a `refusal`
        // part, a `file` part carrying neither reference nor bytes, a custom
        // tool call, and any part type it does not know into exactly this. Sent
        // verbatim they become `{"type": "refusal"}` and `{"type": "custom"}`
        // blocks that Anthropic rejects — the catch-all rendering as speech.
        //
        // A block Anthropic itself sent does not come through here: ingest
        // retains the whole wire `content` array per message and `render_turn`
        // replays it, so a same-protocol round trip never reaches this arm.
        // That is what makes refusing free rather than a losslessness break.
        ContentBlock::Unknown(value) => {
            return Err(Error::InvalidInput(format!(
                "a '{}' content block has no Anthropic form: it came from \
                 another protocol and Anthropic would reject it",
                block_type_of(value)
            )));
        }
        // Refused by `ensure_renderable` before anything gets here.
        ContentBlock::Audio { .. } => None,
    })
}

/// A tool result's payload, for a REQUEST.
///
/// Mirrors the reply's renderer including its bare-string special case, and
/// differs in the one way that matters: it goes through THIS module's fallible
/// [`render_block`]. Blocks nested in a tool result are no more this protocol's
/// than the blocks around them, so every refusal that holds at the top level
/// has to hold one level down — otherwise an OpenAI `refusal` part rejected as
/// a message block sails through as a `tool_result` block, and a tool call
/// whose arguments will not parse becomes `null` instead of an error.
fn render_result_content(blocks: &[ContentBlock]) -> Result<ToolResultContent> {
    if let [ContentBlock::Text { text, cache: None }] = blocks {
        return Ok(ToolResultContent::Text(text.clone()));
    }
    let mut rendered = Vec::new();
    for block in blocks {
        rendered.extend(render_block(block)?);
    }
    Ok(ToolResultContent::Blocks(rendered))
}

/// A gateway media source as this protocol's, for a REQUEST.
///
/// Separate from the reply's for the same reason, and it refuses the one source
/// that cannot cross a protocol boundary: a provider file id is a handle issued
/// by ONE provider's files API and means nothing to another, while
/// [`MediaSource`] carries no provenance to tell an Anthropic id from an OpenAI
/// one. Forwarding it produced a request that could only ever 404.
///
/// An Anthropic file the caller referenced itself never arrives here — a
/// same-protocol turn replays its retained content — so this refuses exactly
/// the foreign case, the same way the `Unknown` block arm does.
fn render_source(source: &MediaSource) -> Result<Source> {
    match source {
        MediaSource::ProviderFile { id } => Err(Error::InvalidInput(format!(
            "file reference '{id}' was issued by another provider's files API, \
             which Anthropic cannot resolve"
        ))),
        MediaSource::Base64 { media_type, data } => Ok(Source::Base64(Base64Source {
            media_type: media_type.clone(),
            data: data.clone(),
            unknown_fields: UnknownFields::new(),
        })),
        MediaSource::Url { url } => Ok(Source::Url(UrlSource {
            url: url.clone(),
            unknown_fields: UnknownFields::new(),
        })),
    }
}

/// A tool call's arguments as Anthropic's `input`.
///
/// Separate from [`render_block`] because it has two distinct refusals and both
/// name the call: text that is not JSON at all, and JSON that is not an object.
fn render_tool_input(arguments: &RawJson, id: &str, name: &str) -> Result<Value> {
    let refuse = |what: &str| {
        Error::InvalidInput(format!(
            "tool call '{id}' to '{name}' has arguments that are {what}, \
             which Anthropic requires: {}",
            arguments.as_str()
        ))
    };
    let input = arguments.parse().map_err(|_| refuse("not JSON"))?;
    if !input.is_object() {
        return Err(refuse("valid JSON but not an object"));
    }
    Ok(input)
}

/// The `type` of an unknown block, for a refusal that names it. Falls back to
/// the whole value when there is no string `type` to quote — a block with none
/// is exactly as unrenderable, and saying so beats saying nothing.
fn block_type_of(value: &Value) -> String {
    match value.get("type") {
        Some(Value::String(tag)) => tag.clone(),
        _ => value.to_string(),
    }
}

fn render_tool(tool: &types::ToolDef) -> Result<Tool> {
    // A tool with no arguments schema is a hosted or server one, and this
    // protocol has no shape for one it did not itself ingest. Refusing names
    // it; sending a custom tool with an empty schema would have the model call
    // something that does not exist.
    if tool.input_schema.is_null() {
        return Err(Error::NotImplemented(
            "a tool with no arguments schema on the anthropic renderer",
        ));
    }
    Ok(Tool::Custom(CustomTool {
        name: tool.name.clone(),
        description: tool.description.clone(),
        input_schema: tool.input_schema.clone(),
        strict: tool.strict,
        cache_control: tool.cache.as_ref().map(|hint| Some(cache_control(hint))),
        unknown_fields: UnknownFields::new(),
    }))
}

fn render_tool_choice(choice: &types::ToolChoice) -> ToolChoice {
    match choice {
        types::ToolChoice::Auto => ToolChoice::Auto(ToolChoiceMode::default()),
        types::ToolChoice::None => ToolChoice::None(ToolChoiceMode::default()),
        types::ToolChoice::Required => ToolChoice::Any(ToolChoiceMode::default()),
        types::ToolChoice::Tool { name } => ToolChoice::Tool(ToolChoiceTool {
            name: name.clone(),
            disable_parallel_tool_use: None,
            unknown_fields: UnknownFields::new(),
        }),
    }
}

/// Which thinking arm a reasoning config asks for, before `display` is attached.
///
/// One function so that [`ensure_renderable`] and [`render_thinking`] cannot
/// disagree about whether a block will be emitted.
enum Arm {
    Enabled(u32),
    Adaptive,
    Disabled,
}

fn thinking_arm(reasoning: &types::ReasoningConfig) -> Option<Arm> {
    // An explicit off wins over a budget: a budget says how MUCH to think, and
    // this says not to.
    if reasoning.enabled == Some(false) {
        return Some(Arm::Disabled);
    }
    match (reasoning.budget_tokens, reasoning.enabled) {
        // A budget is unambiguous — `enabled` is the only arm that takes one.
        (Some(budget), _) => Some(Arm::Enabled(budget)),
        // ...and so is its absence: `adaptive` is the only arm that works
        // without a budget, and inventing one would be a number nobody wrote.
        (None, Some(true)) => Some(Arm::Adaptive),
        (None, _) => None,
    }
}

fn render_thinking(reasoning: &types::ReasoningConfig) -> Option<ThinkingConfig> {
    let display = reasoning
        .exclude
        .map(|exclude| if exclude { "omitted" } else { "summarized" }.to_string());
    Some(match thinking_arm(reasoning)? {
        Arm::Enabled(budget_tokens) => ThinkingConfig::Enabled(ThinkingEnabled {
            budget_tokens,
            display,
            unknown_fields: UnknownFields::new(),
        }),
        Arm::Adaptive => ThinkingConfig::Adaptive(ThinkingAdaptive {
            display,
            unknown_fields: UnknownFields::new(),
        }),
        // `display` is invalid here — there is nothing to display — and
        // `ensure_renderable` has already refused a request that asked for one.
        Arm::Disabled => ThinkingConfig::Disabled(ThinkingDisabled::default()),
    })
}

/// The two unrelated controls Anthropic files under one key.
///
/// Emitted only when one of them has something to say, so a request that asked
/// for neither does not grow an empty object.
fn render_output_config(
    request: &types::ChatRequest,
    unknown_fields: &mut UnknownFields,
) -> Result<Option<OutputConfig>> {
    if let Some(retained) = take_typed(unknown_fields, "output_config") {
        return Ok(Some(retained));
    }
    let config = OutputConfig {
        format: request
            .response_format
            .as_ref()
            .map(render_output_format)
            .transpose()?
            .flatten(),
        // Passed through verbatim rather than checked against Anthropic's set.
        // The two vocabularies overlap but are not equal — chat completions has
        // `minimal`, Anthropic has `max` — and the provider is the authority on
        // its own values, by name, where a table here would go stale silently.
        effort: request
            .reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.clone()),
        unknown_fields: UnknownFields::new(),
    };
    Ok((config.format.is_some() || config.effort.is_some()).then_some(config))
}

/// The gateway's `response_format` as Anthropic's structured output.
///
/// Anthropic gained this after the plan for this work was written, which had it
/// down as untranslatable. It is not: a schema is a schema. What does NOT cross
/// is a schema-less JSON mode — Anthropic requires the schema — so that one is
/// named and refused rather than sent as a request for free-form text, which is
/// what dropping it would silently produce.
fn render_output_format(format: &types::ResponseFormat) -> Result<Option<OutputFormat>> {
    match format {
        // Anthropic's default and the meaning of an absent `format`. Emitting
        // nothing asks for exactly what this asks for.
        types::ResponseFormat::Text => Ok(None),
        types::ResponseFormat::JsonObject => Err(Error::NotImplemented(
            "a schema-less json_object response format on the anthropic renderer",
        )),
        // `name` and `strict` are dropped: Anthropic's format object has
        // neither, its structured output is strict by construction, and the
        // name is a caller-side label with no effect on the reply.
        types::ResponseFormat::JsonSchema { schema, .. } => Ok(Some(OutputFormat {
            r#type: "json_schema".into(),
            schema: Some(schema.clone()),
            unknown_fields: UnknownFields::new(),
        })),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::gateway::openai_compat::api::chat_completions::{
        ingest_request as ingest_openai, render_response as render_openai,
    };
    use crate::gateway::types::{MediaSource, RawJson, ReasoningDetail};

    fn wire(body: Value) -> CreateMessageRequest {
        serde_json::from_value(body).unwrap()
    }

    fn rendered(request: &types::ChatRequest) -> Value {
        plain(&render_request(request, "anthropic", None).unwrap())
    }

    /// Every documented parameter, both `system` forms' worth of shapes, tools
    /// of both kinds, and unknown fields at three nesting levels.
    fn maximal_body() -> Value {
        json!({
            "model": "claude-opus-5",
            "max_tokens": 1024,
            "system": [{
                "type": "text",
                "text": "Be brief.",
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            }],
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "stop_sequences": ["END"],
            "stream": true,
            "metadata": {"user_id": "u1"},
            "service_tier": "auto",
            "tools": [
                {
                    "name": "counter",
                    "description": "counts",
                    "input_schema": {"type": "object", "properties": {"z": {"type": "number"}}},
                    "cache_control": {"type": "ephemeral"},
                    "vendor_hint": true
                },
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
            ],
            "tool_choice": {"type": "any", "disable_parallel_tool_use": true},
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {
                "effort": "xhigh",
                "format": {"type": "json_schema", "schema": {"type": "object"}}
            },
            "messages": [
                {"role": "user", "content": "count them"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "counting", "signature": "sig"},
                    {"type": "tool_use", "id": "toolu_01", "name": "counter", "input": {"z": 1}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01", "content": "3"}
                ]},
                {"role": "user", "content": [
                    {"type": "image", "source": {"type": "url", "url": "https://e.com/a.png"}},
                    {"type": "text", "text": "and this?"}
                ], "vendor_tag": "vt"}
            ]
        })
    }

    /// An Anthropic request must survive normalization and come back out with
    /// zero data loss — including the block-level residue the gateway blocks
    /// have no field for, which only the retained `content` preserves.
    #[test]
    fn an_anthropic_request_round_trips_byte_for_byte() {
        let body = maximal_body();
        let normalized = ingest_request(wire(body.clone()), "claude-opus-5");
        assert_eq!(rendered(&normalized), body);
    }

    /// A caller who asked this protocol for a STREAM asked for its totals by
    /// construction — Anthropic reports them unconditionally, so there is no
    /// state in which one of its streaming callers does not want them. `None`
    /// would read as "no opinion" and cost them the counts on their own
    /// protocol's path.
    ///
    /// It is a request field with no wire spelling here, so nothing about the
    /// round trip above can catch it going wrong.
    #[test]
    fn a_streaming_anthropic_caller_has_asked_for_the_totals() {
        let request = ingest_request(wire(maximal_body()), "claude-opus-5");
        assert!(request.stream, "the fixture stopped asking for a stream");
        assert_eq!(request.stream_include_usage, Some(true));
        // And it stays a gateway-form fact: Anthropic has no `stream_options`,
        // so saying so must not put anything on the wire.
        assert_eq!(rendered(&request), maximal_body());
    }

    /// And a caller who asked for a whole reply asked for no such thing.
    ///
    /// `stream_options` is defined only alongside `stream: true`, so a
    /// normalized request carrying one without the other renders to a body a
    /// stricter chat-completions backend refuses outright. The state is
    /// unreachable from `Client` today — nothing ingests an Anthropic-shaped
    /// request — which is exactly why it needs a test rather than a caller to
    /// find it.
    ///
    /// Both of the two `stream` states, because the fixture above pins only
    /// one and the bug this replaces was invisible from that one.
    #[test]
    fn a_non_streaming_anthropic_caller_has_asked_for_nothing_of_the_kind() {
        for body in [
            json!({"model": "claude-opus-5", "max_tokens": 16,
                   "messages": [{"role": "user", "content": "hi"}]}),
            json!({"model": "claude-opus-5", "max_tokens": 16, "stream": false,
                   "messages": [{"role": "user", "content": "hi"}]}),
        ] {
            let request = ingest_request(wire(body.clone()), "claude-opus-5");
            assert!(!request.stream);
            assert_eq!(request.stream_include_usage, None, "{body}");

            // The whole point of the gate: what the OTHER protocol's renderer
            // then puts on the wire. Asserted on the serialized body, because
            // a struct-level check cannot see a field that should be absent.
            let openai = plain(
                &crate::gateway::openai_compat::api::chat_completions::render_request(
                    &request, "openai",
                )
                .unwrap(),
            );
            assert!(
                openai.get("stream_options").is_none(),
                "a streaming-only option reached a non-streamed request: {openai}"
            );
        }
    }

    /// The gateway view, so a change to the mapping fails on the mapping rather
    /// than only inside a round trip that both halves could change together.
    #[test]
    fn the_gateway_view_of_a_maximal_request() {
        let request = ingest_request(wire(maximal_body()), "claude-opus-5");
        assert_eq!(request.model, "claude-opus-5");
        assert_eq!(request.params.max_tokens, Some(1024));
        assert_eq!(request.params.stop, vec!["END".to_string()]);
        assert!(request.stream);

        // The system prompt becomes a LEADING message, and the tool-result turn
        // becomes the gateway's tool role.
        let roles: Vec<_> = request.messages.iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            vec![
                Role::System,
                Role::User,
                Role::Assistant,
                Role::Tool,
                Role::User
            ]
        );

        // A server tool keeps its name and carries a null schema, which is what
        // lets another protocol's renderer refuse it BY NAME.
        assert_eq!(request.tools.len(), 2);
        assert_eq!(request.tools[0].name, "counter");
        assert_eq!(request.tools[1].name, "web_search");
        assert!(request.tools[1].input_schema.is_null());
        assert!(matches!(
            request.tool_choice,
            Some(types::ToolChoice::Required)
        ));

        // `thinking` and `output_config.effort` are one gateway concept.
        let reasoning = request.reasoning.as_ref().unwrap();
        assert_eq!(reasoning.enabled, Some(true));
        assert_eq!(reasoning.budget_tokens, None, "adaptive carries no budget");
        assert_eq!(reasoning.effort.as_deref(), Some("xhigh"));
        assert_eq!(reasoning.exclude, Some(false), "summarized means returned");
        assert!(matches!(
            request.response_format,
            Some(types::ResponseFormat::JsonSchema { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // The cross-protocol path: an OpenAI-shaped conversation reaching Claude.
    // -----------------------------------------------------------------------

    fn openai_tool_conversation() -> types::ChatRequest {
        ingest_openai(
            serde_json::from_value(json!({
                "model": "anthropic/claude-opus-5",
                "max_tokens": 512,
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "weather in SF and NYC?"},
                    {"role": "assistant", "content": null, "tool_calls": [
                        {"id": "call_a", "type": "function",
                         "function": {"name": "weather", "arguments": "{\"city\":\"SF\"}"}},
                        {"id": "call_b", "type": "function",
                         "function": {"name": "weather", "arguments": "{\"city\":\"NYC\"}"}}
                    ]},
                    {"role": "tool", "tool_call_id": "call_a", "content": "18C"},
                    {"role": "tool", "tool_call_id": "call_b", "content": "3C"}
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "current weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                    }
                }],
                "tool_choice": "auto",
                "reasoning_effort": "medium"
            }))
            .unwrap(),
            "claude-opus-5",
        )
    }

    /// THE test the neutral tool path exists for, and the one Responses will
    /// copy: an OpenAI-shaped request with tools, a two-call assistant turn and
    /// two consecutive tool results becomes a valid Anthropic body — tools
    /// translated, system hoisted, results merged into ONE user turn.
    ///
    /// Rendering the results as two turns would have Anthropic reject the
    /// conversation; rendering only the first would leave the model's second
    /// call looking unanswered, and a model told its tool never replied does
    /// not fail, it guesses.
    #[test]
    fn an_openai_tool_conversation_renders_as_a_valid_anthropic_body() {
        assert_eq!(
            rendered(&openai_tool_conversation()),
            json!({
                "model": "claude-opus-5",
                "max_tokens": 512,
                "system": "Be brief.",
                "messages": [
                    {"role": "user", "content": "weather in SF and NYC?"},
                    {"role": "assistant", "content": [
                        {"type": "tool_use", "id": "call_a", "name": "weather",
                         "input": {"city": "SF"}},
                        {"type": "tool_use", "id": "call_b", "name": "weather",
                         "input": {"city": "NYC"}}
                    ]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "call_a", "content": "18C"},
                        {"type": "tool_result", "tool_use_id": "call_b", "content": "3C"}
                    ]}
                ],
                "tools": [{
                    "name": "weather",
                    "description": "current weather",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
                }],
                "tool_choice": {"type": "auto"},
                "output_config": {"effort": "medium"}
            })
        );
    }

    /// The other direction, closing the loop: Anthropic's reply renders back as
    /// OpenAI-shaped `tool_calls`. Together with the test above this is the
    /// whole round trip a caller actually makes.
    #[test]
    fn an_anthropic_reply_renders_back_as_openai_tool_calls() {
        let reply = super::super::ingest_response(
            serde_json::from_value(json!({
                "id": "msg_1", "type": "message", "role": "assistant",
                "model": "claude-opus-5",
                "content": [{"type": "tool_use", "id": "toolu_9", "name": "weather",
                             "input": {"city": "SF"}}],
                "stop_reason": "tool_use", "stop_sequence": null,
                "usage": {"input_tokens": 10, "output_tokens": 5}
            }))
            .unwrap(),
        );
        let openai = plain(&render_openai(&reply, "anthropic"));
        assert_eq!(openai["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            openai["choices"][0]["message"]["tool_calls"][0],
            json!({
                "id": "toolu_9",
                "type": "function",
                "function": {"name": "weather", "arguments": "{\"city\":\"SF\"}"}
            })
        );
    }

    /// Two NATIVE tool-result turns must come back as two turns.
    ///
    /// Each already was a whole wire turn, so merging them would redraw
    /// boundaries the caller drew and — because a merged turn replays only the
    /// first message's residue — silently drop every later turn's. The merge
    /// exists to serve protocols that must split results across messages, not
    /// to normalize a conversation that already arrived in this one's shape.
    #[test]
    fn consecutive_native_tool_result_turns_are_not_merged() {
        let body = json!({
            "model": "claude-opus-5",
            "max_tokens": 512,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "a", "content": "18C"}],
                    "vendor_tag": "first"
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "b", "content": "3C"}],
                    "vendor_tag": "second"
                }
            ]
        });
        let normalized = ingest_request(wire(body.clone()), "claude-opus-5");
        // Both are tool messages, which is what makes them merge candidates...
        assert_eq!(
            normalized
                .messages
                .iter()
                .map(|m| m.role)
                .collect::<Vec<_>>(),
            vec![Role::Tool, Role::Tool]
        );
        // ...and both come back whole, residue included.
        assert_eq!(rendered(&normalized), body);
    }

    /// The merge still happens where it is the ONLY way to answer more than one
    /// call: messages from a protocol that sends one result each, which carry no
    /// turn of their own under this protocol's namespace. Pinned beside the test
    /// above because the fix for one could trivially break the other.
    #[test]
    fn tool_results_from_another_protocol_still_merge_into_one_turn() {
        let turns = rendered(&openai_tool_conversation())["messages"]
            .as_array()
            .unwrap()
            .len();
        assert_eq!(turns, 3, "the two tool results must share one user turn");
    }

    /// A run that mixes the two: the native turn stands alone and the foreign
    /// ones merge around it, rather than one rule winning for the whole run.
    #[test]
    fn a_native_turn_does_not_absorb_its_foreign_neighbours() {
        let mut request = bare_request();
        request.params.max_tokens = Some(64);
        let foreign = |call_id: &str| types::Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult {
                call_id: call_id.into(),
                content: vec![ContentBlock::Text {
                    text: "ok".into(),
                    cache: None,
                }],
                is_error: false,
            }],
            ext: types::ProviderExt::new(),
        };
        let mut native = foreign("native");
        native.ext.insert(
            "anthropic".into(),
            json!({"content": [{"type": "tool_result", "tool_use_id": "native", "content": "kept"}]}),
        );
        request.messages = vec![native, foreign("a"), foreign("b")];

        let turns = rendered(&request);
        let turns = turns["messages"].as_array().unwrap();
        assert_eq!(turns.len(), 2, "{turns:?}");
        assert_eq!(turns[0]["content"][0]["content"], json!("kept"));
        assert_eq!(turns[1]["content"].as_array().unwrap().len(), 2);
    }

    /// A field addressed to one protocol must not reach another's body. The
    /// merge is unit-tested next door; this pins it at the layer a leak would
    /// actually be observed — a request rendered onto the wire.
    #[test]
    fn another_protocols_ext_never_reaches_an_anthropic_body() {
        let mut request = openai_tool_conversation();
        request
            .ext
            .insert("openai_compat".into(), json!({"seed": 7, "logit_bias": {}}));
        request.ext.insert("anthropic".into(), json!({"top_k": 40}));

        let body = rendered(&request);
        assert_eq!(body["top_k"], json!(40));
        assert!(body.get("seed").is_none(), "{body}");
        assert!(body.get("logit_bias").is_none(), "{body}");
    }

    // -----------------------------------------------------------------------
    // max_tokens: required upstream, optional in the gateway form.
    // -----------------------------------------------------------------------

    fn bare_request() -> types::ChatRequest {
        types::ChatRequest {
            model: "claude-opus-5".into(),
            messages: vec![types::Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "hi".into(),
                    cache: None,
                }],
                ext: types::ProviderExt::new(),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn max_tokens_prefers_the_caller_over_the_roster() {
        let mut request = bare_request();
        request.params.max_tokens = Some(64);
        let body = plain(&render_request(&request, "anthropic", Some(8192)).unwrap());
        assert_eq!(body["max_tokens"], json!(64));
    }

    #[test]
    fn max_tokens_falls_back_to_the_models_documented_ceiling() {
        let body = plain(&render_request(&bare_request(), "anthropic", Some(8192)).unwrap());
        assert_eq!(body["max_tokens"], json!(8192));
    }

    /// Never an invented default: a number picked here would truncate replies
    /// at a length nobody asked for, and would look like the model stopping
    /// early rather than like a missing parameter.
    #[test]
    fn an_unresolvable_max_tokens_refuses_and_names_the_model() {
        let error = render_request(&bare_request(), "anthropic", None).unwrap_err();
        assert!(matches!(error, Error::InvalidInput(_)), "{error}");
        assert!(error.to_string().contains("claude-opus-5"), "{error}");
        assert!(error.to_string().contains("max_tokens"), "{error}");
    }

    // -----------------------------------------------------------------------
    // What this protocol cannot say.
    // -----------------------------------------------------------------------

    fn refusal_of(mutate: impl FnOnce(&mut types::ChatRequest)) -> Error {
        let mut request = bare_request();
        request.params.max_tokens = Some(64);
        mutate(&mut request);
        render_request(&request, "anthropic", None).unwrap_err()
    }

    /// Model-generated arguments are not always valid JSON, and this protocol's
    /// field is an OBJECT — there is nothing honest to send, so the failure
    /// names the call and the text rather than shipping an empty object the
    /// model never wrote.
    #[test]
    fn unparseable_tool_arguments_refuse_and_name_the_call() {
        let error = refusal_of(|request| {
            request.messages[0].role = Role::Assistant;
            request.messages[0].content = vec![ContentBlock::ToolCall {
                id: "call_a".into(),
                name: "weather".into(),
                arguments: RawJson::new(r#"{"city": "S"#),
            }];
        });
        assert!(matches!(error, Error::InvalidInput(_)), "{error}");
        for expected in ["call_a", "weather", r#"{"city": "S"#] {
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    /// Parsing is not the whole check. Anthropic's `input` is an OBJECT, and
    /// every one of these is valid JSON that is not one — so a parse-only gate
    /// would forward them and let the provider answer with a 400 that names
    /// neither the call nor the text.
    #[test]
    fn tool_arguments_that_are_json_but_not_an_object_refuse() {
        for arguments in ["null", "[1]", "7", r#""a string""#, "true"] {
            let error = refusal_of(|request| {
                request.messages[0].role = Role::Assistant;
                request.messages[0].content = vec![ContentBlock::ToolCall {
                    id: "call_a".into(),
                    name: "weather".into(),
                    arguments: RawJson::new(arguments),
                }];
            });
            assert!(
                matches!(error, Error::InvalidInput(_)),
                "{arguments}: {error}"
            );
            for expected in ["call_a", "weather", arguments, "not an object"] {
                assert!(error.to_string().contains(expected), "{arguments}: {error}");
            }
        }
    }

    /// An empty argument string is the one non-object that is NOT a refusal:
    /// `RawJson::parse` reads it as `{}`, which is what a no-argument call
    /// means and what every protocol here spells differently.
    #[test]
    fn empty_tool_arguments_stay_an_empty_object() {
        let mut request = bare_request();
        request.params.max_tokens = Some(64);
        request.messages[0].role = Role::Assistant;
        request.messages[0].content = vec![ContentBlock::ToolCall {
            id: "call_a".into(),
            name: "ping".into(),
            arguments: RawJson::new(""),
        }];
        let body = plain(&render_request(&request, "anthropic", None).unwrap());
        assert_eq!(body["messages"][0]["content"][0]["input"], json!({}));
    }

    /// The catch-all that would otherwise render as speech. Chat completions
    /// turns a `refusal` part, a bytes-less `file` part, a custom tool call and
    /// any unrecognized part into `ContentBlock::Unknown`; forwarding those
    /// verbatim sends Anthropic block types it does not define.
    #[test]
    fn an_unknown_block_from_another_protocol_refuses_and_names_its_type() {
        let error = refusal_of(|request| {
            request.messages[0].content = vec![ContentBlock::Unknown(
                json!({"type": "refusal", "refusal": "I cannot help with that"}),
            )];
        });
        assert!(matches!(error, Error::InvalidInput(_)), "{error}");
        assert!(error.to_string().contains("refusal"), "{error}");
    }

    /// A block with no string `type` is exactly as unrenderable, and the
    /// refusal still has to say WHICH block — so it quotes the value.
    #[test]
    fn an_unknown_block_without_a_type_still_names_itself() {
        let error = refusal_of(|request| {
            request.messages[0].content = vec![ContentBlock::Unknown(json!({"odd": 1}))];
        });
        assert!(error.to_string().contains(r#"{"odd":1}"#), "{error}");
    }

    /// The refusal above is free ONLY because a block Anthropic itself sent
    /// never reaches that arm: ingest retains the whole wire `content` array
    /// and `render_turn` replays it. Without this test the refusal would look
    /// like it broke same-protocol losslessness for any block type Anthropic
    /// adds tomorrow — the case the `Unknown` arm was built for.
    #[test]
    fn an_anthropic_native_unknown_block_still_round_trips() {
        let body = json!({
            "model": "claude-opus-5",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": [
                {"type": "web_search_tool_result_20991231", "payload": {"q": "rust"}}
            ]}]
        });
        let normalized = ingest_request(wire(body.clone()), "claude-opus-5");
        assert!(
            matches!(
                normalized.messages[0].content.as_slice(),
                [ContentBlock::Unknown(_)]
            ),
            "the block must land in the arm the refusal guards, or this proves nothing"
        );
        assert_eq!(rendered(&normalized), body);
    }

    /// The refusals above have to survive one level of nesting. A tool result
    /// carries blocks, and borrowing the REPLY's infallible renderer for them
    /// let exactly the values rejected at the top level through underneath —
    /// so this is the same input as
    /// `an_unknown_block_from_another_protocol_refuses_and_names_its_type`,
    /// moved inside a `tool_result`.
    #[test]
    fn an_unknown_block_nested_in_a_tool_result_refuses() {
        let error = refusal_of(|request| {
            request.messages[0].role = Role::Tool;
            request.messages[0].content = vec![ContentBlock::ToolResult {
                call_id: "call_a".into(),
                content: vec![ContentBlock::Unknown(json!({"type": "refusal"}))],
                is_error: false,
            }];
        });
        assert!(matches!(error, Error::InvalidInput(_)), "{error}");
        assert!(error.to_string().contains("refusal"), "{error}");
    }

    /// The other half of the same hole: nested arguments went through the
    /// reply's `unwrap_or(Value::Null)` rather than this module's refusal, so a
    /// malformed tool call became a silent `null` input.
    #[test]
    fn unparseable_arguments_nested_in_a_tool_result_refuse() {
        let error = refusal_of(|request| {
            request.messages[0].role = Role::Tool;
            request.messages[0].content = vec![ContentBlock::ToolResult {
                call_id: "call_a".into(),
                content: vec![ContentBlock::ToolCall {
                    id: "call_b".into(),
                    name: "weather".into(),
                    arguments: RawJson::new("{oops"),
                }],
                is_error: false,
            }];
        });
        assert!(matches!(error, Error::InvalidInput(_)), "{error}");
        assert!(error.to_string().contains("call_b"), "{error}");
    }

    /// A file id is a handle into ONE provider's files API. `MediaSource` has
    /// no provenance, so an OpenAI `file-…` and an Anthropic one are the same
    /// shape here — and forwarding the foreign one builds a request that can
    /// only 404.
    #[test]
    fn a_foreign_provider_file_reference_refuses_and_names_the_id() {
        for block in [
            ContentBlock::Document {
                source: MediaSource::ProviderFile {
                    id: "file-abc123".into(),
                },
                title: None,
                cache: None,
            },
            ContentBlock::Image {
                source: MediaSource::ProviderFile {
                    id: "file-abc123".into(),
                },
                detail: None,
                cache: None,
            },
        ] {
            let error = refusal_of(|request| request.messages[0].content = vec![block]);
            assert!(matches!(error, Error::InvalidInput(_)), "{error}");
            assert!(error.to_string().contains("file-abc123"), "{error}");
        }
    }

    /// And the refusal above stays free: Anthropic's own file reference comes
    /// back through the retained content, never through the renderer that
    /// refuses. Without this the fix would have broken a documented native
    /// request shape.
    #[test]
    fn an_anthropic_native_file_reference_still_round_trips() {
        let body = json!({
            "model": "claude-opus-5",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": [
                {"type": "document", "source": {"type": "file", "file_id": "file_011CNha8iCJcU1wXNR6q4V8w"}}
            ]}]
        });
        let normalized = ingest_request(wire(body.clone()), "claude-opus-5");
        assert!(
            matches!(
                normalized.messages[0].content.as_slice(),
                [ContentBlock::Document {
                    source: MediaSource::ProviderFile { .. },
                    ..
                }]
            ),
            "the source must land in the arm the refusal guards, or this proves nothing"
        );
        assert_eq!(rendered(&normalized), body);
    }

    /// A system message the caller placed mid-conversation cannot be hoisted
    /// without reordering what the model sees, and cannot be rendered as a user
    /// turn without fabricating one. Neither is a translation.
    #[test]
    fn a_system_message_after_the_first_turn_refuses() {
        let error = refusal_of(|request| {
            request.messages.push(types::Message {
                role: Role::System,
                content: vec![ContentBlock::Text {
                    text: "now be terse".into(),
                    cache: None,
                }],
                ext: types::ProviderExt::new(),
            });
        });
        assert!(matches!(error, Error::NotImplemented(_)), "{error}");
        assert!(error.to_string().contains("system message"), "{error}");
    }

    /// Dropping audio would send a request that reads as if the caller never
    /// attached it, and the model would answer confidently about nothing.
    #[test]
    fn an_audio_block_refuses() {
        let error = refusal_of(|request| {
            request.messages[0].content.push(ContentBlock::Audio {
                source: MediaSource::Base64 {
                    media_type: "audio/wav".into(),
                    data: "aGk=".into(),
                },
                format: Some("wav".into()),
            });
        });
        assert!(error.to_string().contains("audio"), "{error}");
    }

    /// Nested as well as top-level: a block inside a tool result is as
    /// unrenderable as one beside it.
    #[test]
    fn an_audio_block_inside_a_tool_result_refuses() {
        let error = refusal_of(|request| {
            request.messages[0].content = vec![ContentBlock::ToolResult {
                call_id: "call_a".into(),
                content: vec![ContentBlock::Audio {
                    source: MediaSource::Url {
                        url: "https://e.com/a.wav".into(),
                    },
                    format: None,
                }],
                is_error: false,
            }];
        });
        assert!(error.to_string().contains("audio"), "{error}");
    }

    /// Chat completions' `file` part carries bytes and a filename and declares
    /// no MIME type, so its ingest fills an empty string rather than guessing
    /// from the extension. Anthropic REQUIRES one, so forwarding the empty
    /// string would build a request certain to be rejected — and the caller
    /// would read that 400 as being about their document rather than about the
    /// conversion. Refusing here says which.
    #[test]
    fn inline_bytes_with_no_media_type_refuse() {
        let error = refusal_of(|request| {
            request.messages[0].content.push(ContentBlock::Document {
                source: MediaSource::Base64 {
                    media_type: String::new(),
                    data: "JVBERi0=".into(),
                },
                title: Some("report.pdf".into()),
                cache: None,
            });
        });
        assert!(matches!(error, Error::NotImplemented(_)), "{error}");
        assert!(error.to_string().contains("media type"), "{error}");
    }

    /// The end-to-end shape of that refusal: an OpenAI request carrying a
    /// `file` part is what actually produces it, and it must be named rather
    /// than sent.
    #[test]
    fn an_openai_file_part_refuses_rather_than_being_sent_invalid() {
        let request = ingest_openai(
            serde_json::from_value(json!({
                "model": "anthropic/claude-opus-5",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "summarize this"},
                    {"type": "file", "file": {"file_data": "JVBERi0=", "filename": "r.pdf"}}
                ]}]
            }))
            .unwrap(),
            "claude-opus-5",
        );
        let error = render_request(&request, "anthropic", None).unwrap_err();
        assert!(error.to_string().contains("media type"), "{error}");
    }

    /// ...while a document that DOES name its type renders normally. The gate
    /// must refuse the missing media type, not inline bytes in general.
    #[test]
    fn inline_bytes_with_a_media_type_render() {
        let mut request = bare_request();
        request.params.max_tokens = Some(64);
        request.messages[0].content = vec![ContentBlock::Document {
            source: MediaSource::Base64 {
                media_type: "application/pdf".into(),
                data: "JVBERi0=".into(),
            },
            title: None,
            cache: None,
        }];
        assert_eq!(
            rendered(&request)["messages"][0]["content"],
            json!([{
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBERi0="
                }
            }])
        );
    }

    #[test]
    fn a_tool_with_no_schema_refuses_by_name() {
        let error = refusal_of(|request| {
            request.tools.push(types::ToolDef {
                name: "web_search".into(),
                description: None,
                input_schema: Value::Null,
                strict: None,
                cache: None,
                ext: types::ProviderExt::new(),
            });
        });
        assert!(error.to_string().contains("arguments schema"), "{error}");
    }

    /// Anthropic requires a schema, so a schema-less JSON mode has no wire
    /// form. Dropping it would silently ask for free-form text instead.
    #[test]
    fn a_schema_less_json_mode_refuses() {
        let error = refusal_of(|request| {
            request.response_format = Some(types::ResponseFormat::JsonObject);
        });
        assert!(error.to_string().contains("json_object"), "{error}");
    }

    #[test]
    fn a_request_level_cache_hint_refuses() {
        let error = refusal_of(|request| {
            request.cache = Some(types::CacheHint { ttl: None });
        });
        assert!(error.to_string().contains("cache hint"), "{error}");
    }

    // -----------------------------------------------------------------------
    // Thinking and effort, the two parameters that are one gateway concept.
    // -----------------------------------------------------------------------

    fn with_reasoning(reasoning: types::ReasoningConfig) -> Value {
        let mut request = bare_request();
        request.params.max_tokens = Some(64);
        request.reasoning = Some(reasoning);
        rendered(&request)
    }

    /// A budget is unambiguous — `enabled` is the only arm that takes one — and
    /// its absence is equally so, since `adaptive` is the only arm that works
    /// without. Neither needs to know which model is routed to: a model that
    /// rejects the arm says so by name, where warpllm guessing would have to
    /// invent a budget or discard the caller's.
    #[test]
    fn the_thinking_arm_follows_the_budget_not_the_model() {
        assert_eq!(
            with_reasoning(types::ReasoningConfig {
                enabled: Some(true),
                budget_tokens: Some(4096),
                ..Default::default()
            })["thinking"],
            json!({"type": "enabled", "budget_tokens": 4096})
        );
        assert_eq!(
            with_reasoning(types::ReasoningConfig {
                enabled: Some(true),
                ..Default::default()
            })["thinking"],
            json!({"type": "adaptive"})
        );
        assert_eq!(
            with_reasoning(types::ReasoningConfig {
                enabled: Some(false),
                ..Default::default()
            })["thinking"],
            json!({"type": "disabled"})
        );
    }

    /// An explicit off beats a budget: a budget says how MUCH to think, and
    /// this says not to.
    #[test]
    fn an_explicit_off_outranks_a_budget() {
        assert_eq!(
            with_reasoning(types::ReasoningConfig {
                enabled: Some(false),
                budget_tokens: Some(4096),
                ..Default::default()
            })["thinking"],
            json!({"type": "disabled"})
        );
    }

    /// "Reason but do not return it" is `display`, which rides whichever arm is
    /// being emitted rather than being a mode of its own.
    #[test]
    fn an_exclusion_becomes_the_display_field() {
        assert_eq!(
            with_reasoning(types::ReasoningConfig {
                enabled: Some(true),
                budget_tokens: Some(4096),
                exclude: Some(true),
                ..Default::default()
            })["thinking"],
            json!({"type": "enabled", "budget_tokens": 4096, "display": "omitted"})
        );
    }

    /// ...and has nowhere to go without one, so it is named rather than
    /// dropped. A caller who asked not to be shown reasoning would otherwise
    /// get a request that never mentioned it.
    #[test]
    fn an_exclusion_with_no_thinking_mode_refuses() {
        let error = refusal_of(|request| {
            request.reasoning = Some(types::ReasoningConfig {
                exclude: Some(true),
                ..Default::default()
            });
        });
        assert!(error.to_string().contains("exclusion"), "{error}");
    }

    /// The effort word crosses verbatim into the parameter Anthropic now has
    /// for it — this is the mapping the plan for this work had down as
    /// impossible, before `output_config` existed.
    #[test]
    fn an_effort_renders_into_output_config_and_not_into_thinking() {
        let body = with_reasoning(types::ReasoningConfig {
            effort: Some("low".into()),
            ..Default::default()
        });
        assert_eq!(body["output_config"], json!({"effort": "low"}));
        assert!(body.get("thinking").is_none(), "{body}");
    }

    /// The two vocabularies overlap without being equal — chat completions has
    /// `minimal`, Anthropic has `max` — and the provider is the authority on
    /// its own values. A table here would go stale silently.
    #[test]
    fn an_effort_anthropic_does_not_define_is_passed_through_not_filtered() {
        assert_eq!(
            with_reasoning(types::ReasoningConfig {
                effort: Some("minimal".into()),
                ..Default::default()
            })["output_config"],
            json!({"effort": "minimal"})
        );
    }

    #[test]
    fn a_json_schema_becomes_the_output_format() {
        let mut request = bare_request();
        request.params.max_tokens = Some(64);
        request.response_format = Some(types::ResponseFormat::JsonSchema {
            name: "answer".into(),
            schema: json!({"type": "object"}),
            strict: Some(true),
        });
        // `name` and `strict` have no Anthropic counterpart and are dropped;
        // the schema is the whole of what this protocol says.
        assert_eq!(
            rendered(&request)["output_config"],
            json!({"format": {"type": "json_schema", "schema": {"type": "object"}}})
        );
    }

    /// `text` IS this protocol's default, so emitting nothing asks for exactly
    /// what it asks for — and a request that wanted neither control must not
    /// grow an empty `output_config`.
    #[test]
    fn a_text_response_format_emits_nothing_at_all() {
        let mut request = bare_request();
        request.params.max_tokens = Some(64);
        request.response_format = Some(types::ResponseFormat::Text);
        let body = rendered(&request);
        assert!(body.get("output_config").is_none(), "{body}");
    }

    // -----------------------------------------------------------------------
    // Reasoning blocks, which only replay when Anthropic produced them.
    // -----------------------------------------------------------------------

    /// Anthropic requires a run of its own thinking blocks back untouched, and
    /// this sends exactly those. Another provider's reasoning carries no
    /// signature Anthropic would accept, so forwarding it would reject the turn
    /// rather than preserve anything.
    #[test]
    fn only_anthropics_own_reasoning_is_sent_back() {
        let mut request = bare_request();
        request.params.max_tokens = Some(64);
        request.messages[0].role = Role::Assistant;
        request.messages[0].content = vec![
            ContentBlock::Reasoning {
                detail: ReasoningDetail::Text {
                    text: "mine".into(),
                    signature: Some("sig".into()),
                },
                provenance: Some("anthropic".into()),
                id: None,
            },
            ContentBlock::Reasoning {
                detail: ReasoningDetail::Text {
                    text: "theirs".into(),
                    signature: None,
                },
                provenance: Some("openai_compat".into()),
                id: None,
            },
        ];
        assert_eq!(
            rendered(&request)["messages"][0]["content"],
            json!([{"type": "thinking", "thinking": "mine", "signature": "sig"}])
        );
    }

    /// The transport CHECKS this flag rather than setting it, so a whole-reply
    /// exchange must not carry one and a streamed one must.
    #[test]
    fn stream_is_emitted_only_when_it_is_true() {
        let mut request = bare_request();
        request.params.max_tokens = Some(64);
        assert!(rendered(&request).get("stream").is_none());
        request.stream = true;
        assert_eq!(rendered(&request)["stream"], json!(true));
    }
}
