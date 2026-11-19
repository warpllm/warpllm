//! Request conversion: gateway → Google's `generateContent` wire request.
//!
//! EGRESS ONLY (see [`crate::gateway::google`]), so there is one direction and
//! nothing is retained: every request is rendered from the gateway form.
//!
//! # What this protocol arranges differently, and where each is handled
//!
//! | | gateway | Google | handled by |
//! |---|---|---|---|
//! | system prompt | leading `System` messages | top-level `systemInstruction` | [`render_messages`] hoists it |
//! | roles | system / user / assistant / tool | `user` / `model` only | [`role_to_wire`] |
//! | tool result | a block holding `call_id` | a `functionResponse` holding the NAME | [`call_names`] looks it up |
//! | tool arguments | a string of JSON | a decoded object | [`render_tool_args`], which can fail |
//! | sampling | `params` | `generationConfig` | [`render_generation_config`] |
//! | reasoning | `reasoning` | `generationConfig.thinkingConfig` | [`render_thinking`] |
//! | structured output | `response_format` | `responseMimeType` + `responseJsonSchema` | [`render_generation_config`] |
//!
//! # Refusal policy
//!
//! A field with no Google slot is refused rather than dropped, because
//! dropping it would CHANGE the request instead of translating it. Where
//! Google can say no itself (a bad thinking level, thinking with a tool loop,
//! a budget on a model that takes a level), nothing is checked here: the
//! provider is the authority on its own values, and refusal logic here would
//! be logic to maintain against a moving target.

use std::collections::HashMap;

use serde_json::{Value, json};

use crate::error::{Error, Result};
use crate::gateway::google::{merged_ext, role_to_wire};
use crate::gateway::types::{self, ContentBlock, MediaSource, RawJson, Role};
use crate::protocol::UnknownFields;
use crate::protocol::google::generate_content::types::{
    Blob, Content, FunctionCall, FunctionCallingConfig, FunctionDeclaration, FunctionResponse,
    GenerateContentRequest, GenerationConfig, Part, ThinkingConfig, Tool, ToolConfig,
};

/// Fields this module sets itself. A copy riding the ext bag would serialize a
/// second key beside the typed one, so the typed field wins and the copy goes.
const TYPED_KEYS: [&str; 5] = [
    "contents",
    "systemInstruction",
    "generationConfig",
    "tools",
    "toolConfig",
];

/// The model is not a field: on AI Studio and on Vertex it is in the URL.
// Dead until the transport (subtask 6 of #35) calls it.
#[allow(dead_code)]
pub(crate) fn render_request(
    request: &types::ChatRequest,
    provider: &str,
) -> Result<GenerateContentRequest> {
    ensure_renderable(request)?;
    let mut unknown_fields = merged_ext(&request.ext, provider);
    for key in TYPED_KEYS {
        unknown_fields.remove(key);
    }
    let names = call_names(request);
    let (system_instruction, contents) = render_messages(&request.messages, &names)?;
    Ok(GenerateContentRequest {
        contents,
        system_instruction,
        generation_config: render_generation_config(request)?,
        tools: render_tools(request)?,
        tool_config: request.tool_choice.as_ref().map(render_tool_choice),
        unknown_fields,
    })
}

/// What the gateway form can hold and this protocol cannot say.
fn ensure_renderable(request: &types::ChatRequest) -> Result<()> {
    // Google has no prompt-cache breakpoint on a block or a request, so a hint
    // would be dropped. Cost-only, but still the caller's instruction.
    // TODO(#35): revisit whether block-level hints should be dropped instead.
    if request.cache.is_some()
        || request.tools.iter().any(|tool| tool.cache.is_some())
        || blocks(request).any(|block| {
            matches!(
                block,
                ContentBlock::Text { cache: Some(_), .. }
                    | ContentBlock::Image { cache: Some(_), .. }
                    | ContentBlock::Document { cache: Some(_), .. }
            )
        })
    {
        return Err(Error::NotImplemented(
            "a prompt-cache hint on the google renderer",
        ));
    }
    // Google has no switch for this, and sending the request without it would
    // allow what the caller forbade.
    if request.parallel_tool_calls == Some(false) {
        return Err(Error::NotImplemented(
            "parallel_tool_calls: false on the google renderer",
        ));
    }
    Ok(())
}

/// Every block in a request, tool-result payloads included.
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

/// Tool-call id → function name, from every call in the conversation.
///
/// A `functionResponse` is matched to its call by NAME, and the gateway's
/// `ToolResult` carries only the call's id.
fn call_names(request: &types::ChatRequest) -> HashMap<&str, &str> {
    blocks(request)
        .filter_map(|block| match block {
            ContentBlock::ToolCall { id, name, .. } => Some((id.as_str(), name.as_str())),
            _ => None,
        })
        .collect()
}

/// Splits the gateway's one message list into this protocol's two: the
/// top-level `systemInstruction` and the turns.
///
/// A system message AFTER the conversation has started is refused rather than
/// moved, for the reason the Anthropic renderer gives: hoisting it would
/// reorder the conversation, and rendering it as a user turn would fabricate
/// one.
///
/// Consecutive tool results become ONE user turn holding every
/// `functionResponse`, because the responses to a turn's parallel calls must
/// arrive together. A turn left with no parts (an assistant turn that held only
/// another provider's reasoning) is skipped: Google rejects an empty turn.
fn render_messages(
    messages: &[types::Message],
    names: &HashMap<&str, &str>,
) -> Result<(Option<Content>, Vec<Content>)> {
    let leading = messages
        .iter()
        .take_while(|message| message.role == Role::System)
        .count();
    if messages[leading..]
        .iter()
        .any(|message| message.role == Role::System)
    {
        return Err(Error::NotImplemented(
            "a system message after the conversation has started on the google renderer",
        ));
    }

    let mut system_parts = Vec::new();
    for message in &messages[..leading] {
        for block in &message.content {
            system_parts.extend(render_block(block, names)?);
        }
    }
    let system = (!system_parts.is_empty()).then(|| Content {
        role: None,
        parts: system_parts,
        ..Default::default()
    });

    let mut turns = Vec::new();
    let mut rest = &messages[leading..];
    while let Some(first) = rest.first() {
        let run = match first.role {
            Role::Tool => rest
                .iter()
                .take_while(|message| message.role == Role::Tool)
                .count(),
            _ => 1,
        };
        let mut parts = Vec::new();
        for message in &rest[..run] {
            for block in &message.content {
                parts.extend(render_block(block, names)?);
            }
        }
        if !parts.is_empty() {
            turns.push(Content {
                role: Some(role_to_wire(first.role).to_string()),
                parts,
                ..Default::default()
            });
        }
        rest = &rest[run..];
    }
    Ok((system, turns))
}

/// A gateway block as a Google part, where one exists.
///
/// `None` means the block has no shape here and is dropped: reasoning another
/// provider produced, which Google could not verify.
fn render_block(block: &ContentBlock, names: &HashMap<&str, &str>) -> Result<Option<Part>> {
    Ok(match block {
        ContentBlock::Text { text, .. } => Some(Part {
            text: Some(text.clone()),
            ..Default::default()
        }),
        // `detail` (images) and `title` (documents) are hints with no Google
        // field. Dropping them changes nothing the model is shown.
        ContentBlock::Image { source, .. }
        | ContentBlock::Document { source, .. }
        | ContentBlock::Audio { source, .. } => Some(Part {
            inline_data: Some(render_blob(source)?),
            ..Default::default()
        }),
        ContentBlock::ToolCall {
            id,
            name,
            arguments,
        } => Some(Part {
            function_call: Some(FunctionCall {
                // The id goes on the wire so a call and its response can be
                // paired and logged on Google's side. Whether every model
                // accepts one is untested: a live check is a follow-up.
                id: Some(id.clone()),
                name: name.clone(),
                args: Some(render_tool_args(arguments, id, name)?),
                unknown_fields: UnknownFields::new(),
            }),
            ..Default::default()
        }),
        ContentBlock::ToolResult {
            call_id,
            content,
            is_error,
        } => {
            let name = names.get(call_id.as_str()).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "tool result for call '{call_id}' has no matching tool call in the \
                     conversation, and Google needs the function's name"
                ))
            })?;
            Some(Part {
                function_response: Some(FunctionResponse {
                    id: Some(call_id.clone()),
                    name: (*name).to_string(),
                    response: render_result(content, *is_error)?,
                    unknown_fields: UnknownFields::new(),
                }),
                ..Default::default()
            })
        }
        ContentBlock::Reasoning { .. } => None,
        ContentBlock::Unknown(value) => {
            return Err(Error::InvalidInput(format!(
                "a '{}' content block has no Google form: it came from another \
                 protocol and Google would reject it",
                block_type_of(value)
            )));
        }
    })
}

/// A media source as inline bytes.
///
/// Refused when there is no MIME type to send, which is every URL source and an
/// inline one that declared none. Guessing a type from an extension would send
/// a request the caller did not write.
/// TODO(#35): revisit once there is a Gemini or cloud library to validate MIME
/// types against; until then refusal is the deferred answer.
fn render_blob(source: &MediaSource) -> Result<Blob> {
    match source {
        MediaSource::Base64 { media_type, data } if !media_type.is_empty() => Ok(Blob {
            mime_type: media_type.clone(),
            data: data.clone(),
            unknown_fields: UnknownFields::new(),
        }),
        MediaSource::Base64 { .. } => Err(Error::NotImplemented(
            "inline bytes with no media type on the google renderer",
        )),
        MediaSource::Url { .. } => Err(Error::NotImplemented(
            "a URL media source on the google renderer: Google's file reference \
             needs a media type the gateway form does not carry",
        )),
        MediaSource::ProviderFile { id } => Err(Error::InvalidInput(format!(
            "file reference '{id}' was issued by another provider's files API, \
             which Google cannot resolve"
        ))),
    }
}

/// A tool call's arguments as Google's `args`: an object.
///
/// Two refusals, both naming the call: text that is not JSON, and JSON that is
/// not an object.
fn render_tool_args(arguments: &RawJson, id: &str, name: &str) -> Result<Value> {
    let refuse = |what: &str| {
        Error::InvalidInput(format!(
            "tool call '{id}' to '{name}' has arguments that are {what}, \
             which Google requires: {}",
            arguments.as_str()
        ))
    };
    let args = arguments.parse().map_err(|_| refuse("not JSON"))?;
    if !args.is_object() {
        return Err(refuse("valid JSON but not an object"));
    }
    Ok(args)
}

/// A tool result as `functionResponse.response`, which must be an object.
///
/// The text goes under `result`, or under `error` for a failed call: Google's
/// documented convention for a response with no structure of its own.
/// A non-text payload is refused rather than flattened.
fn render_result(content: &[ContentBlock], is_error: bool) -> Result<Value> {
    let mut texts = Vec::with_capacity(content.len());
    for block in content {
        match block {
            ContentBlock::Text { text, .. } => texts.push(text.as_str()),
            _ => {
                return Err(Error::NotImplemented(
                    "a non-text tool result on the google renderer",
                ));
            }
        }
    }
    let key = if is_error { "error" } else { "result" };
    Ok(json!({ key: texts.join("\n") }))
}

/// The `type` of an unknown block, for a refusal that names it.
fn block_type_of(value: &Value) -> String {
    match value.get("type") {
        Some(Value::String(tag)) => tag.clone(),
        _ => value.to_string(),
    }
}

fn render_tools(request: &types::ChatRequest) -> Result<Option<Vec<Tool>>> {
    if request.tools.is_empty() {
        return Ok(None);
    }
    let mut declarations = Vec::with_capacity(request.tools.len());
    for tool in &request.tools {
        // A tool with no schema is a hosted one, which this renderer has no
        // shape for. `strict` is dropped: Google has no such flag and the schema
        // still constrains the call.
        if tool.input_schema.is_null() {
            return Err(Error::NotImplemented(
                "a tool with no arguments schema on the google renderer",
            ));
        }
        declarations.push(FunctionDeclaration {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters_json_schema: Some(tool.input_schema.clone()),
            unknown_fields: UnknownFields::new(),
        });
    }
    Ok(Some(vec![Tool {
        function_declarations: Some(declarations),
        ..Default::default()
    }]))
}

fn render_tool_choice(choice: &types::ToolChoice) -> ToolConfig {
    let (mode, allowed) = match choice {
        types::ToolChoice::Auto => ("AUTO", None),
        types::ToolChoice::None => ("NONE", None),
        types::ToolChoice::Required => ("ANY", None),
        types::ToolChoice::Tool { name } => ("ANY", Some(vec![name.clone()])),
    };
    ToolConfig {
        function_calling_config: Some(FunctionCallingConfig {
            mode: Some(mode.to_string()),
            allowed_function_names: allowed,
            unknown_fields: UnknownFields::new(),
        }),
        ..Default::default()
    }
}

/// Sampling, output format and thinking, emitted only when one has something to
/// say so a request that asked for none does not grow an empty object.
///
/// `name` and `strict` on a JSON schema are dropped: Google's schema field has
/// neither, and the name is a caller-side label.
fn render_generation_config(request: &types::ChatRequest) -> Result<Option<GenerationConfig>> {
    let params = &request.params;
    let (mime, schema) = match &request.response_format {
        None | Some(types::ResponseFormat::Text) => (None, None),
        Some(types::ResponseFormat::JsonObject) => (Some("application/json".to_string()), None),
        Some(types::ResponseFormat::JsonSchema { schema, .. }) => {
            (Some("application/json".to_string()), Some(schema.clone()))
        }
    };
    let config = GenerationConfig {
        stop_sequences: (!params.stop.is_empty()).then(|| params.stop.clone()),
        max_output_tokens: params.max_tokens,
        temperature: params.temperature,
        top_p: params.top_p,
        top_k: params.top_k,
        response_mime_type: mime,
        response_json_schema: schema,
        thinking_config: render_thinking(request.reasoning.as_ref())?,
        ..Default::default()
    };
    // Compared as JSON so a field added to the type later cannot be forgotten
    // by a hand-written emptiness check.
    let empty = serde_json::to_value(&config)
        .ok()
        .and_then(|value| value.as_object().map(|object| object.is_empty()))
        .unwrap_or(false);
    Ok((!empty).then_some(config))
}

/// The gateway's reasoning config as `thinkingConfig`.
///
/// Budget and level are exclusive PER MODEL, so both are passed through when
/// both are set and Google's own error says which the routed model takes. The
/// level is uppercased and otherwise unchecked, for the same reason. An
/// explicit off wins over a budget: one says how much to think, the other says
/// not to.
fn render_thinking(reasoning: Option<&types::ReasoningConfig>) -> Result<Option<ThinkingConfig>> {
    let Some(reasoning) = reasoning else {
        return Ok(None);
    };
    let mut config = ThinkingConfig::default();
    if reasoning.enabled == Some(false) {
        config.thinking_budget = Some(0);
    } else {
        config.thinking_budget = reasoning
            .budget_tokens
            .map(|budget| {
                i32::try_from(budget).map_err(|_| {
                    Error::InvalidInput(format!(
                        "a thinking budget of {budget} tokens is larger than Google's int32 field"
                    ))
                })
            })
            .transpose()?;
        config.thinking_level = reasoning
            .effort
            .as_ref()
            .map(|effort| effort.to_uppercase());
    }
    config.include_thoughts = reasoning.exclude.map(|exclude| !exclude);
    let says_something = config.thinking_budget.is_some()
        || config.thinking_level.is_some()
        || config.include_thoughts.is_some();
    Ok(says_something.then_some(config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::types::{CacheHint, ProviderExt};

    fn message(role: Role, content: Vec<ContentBlock>) -> types::Message {
        types::Message {
            role,
            content,
            ext: ProviderExt::new(),
        }
    }

    fn text(role: Role, text: &str) -> types::Message {
        message(
            role,
            vec![ContentBlock::Text {
                text: text.into(),
                cache: None,
            }],
        )
    }

    fn call(id: &str, name: &str, arguments: &str) -> ContentBlock {
        ContentBlock::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: RawJson::new(arguments),
        }
    }

    fn result(call_id: &str, body: &str, is_error: bool) -> ContentBlock {
        ContentBlock::ToolResult {
            call_id: call_id.into(),
            content: vec![ContentBlock::Text {
                text: body.into(),
                cache: None,
            }],
            is_error,
        }
    }

    fn request(messages: Vec<types::Message>) -> types::ChatRequest {
        types::ChatRequest {
            model: "gemini-2.5-flash".into(),
            messages,
            ..Default::default()
        }
    }

    fn render(request: &types::ChatRequest) -> Value {
        serde_json::to_value(render_request(request, "google").unwrap()).unwrap()
    }

    fn refused(request: &types::ChatRequest) -> Error {
        render_request(request, "google").unwrap_err()
    }

    #[test]
    fn a_leading_system_message_is_hoisted_and_roles_are_renamed() {
        let wire = render(&request(vec![
            text(Role::System, "be brief"),
            text(Role::User, "hi"),
            text(Role::Assistant, "hello"),
        ]));
        assert_eq!(
            wire["systemInstruction"],
            json!({"parts": [{"text": "be brief"}]})
        );
        assert_eq!(
            wire["contents"],
            json!([
                {"role": "user", "parts": [{"text": "hi"}]},
                {"role": "model", "parts": [{"text": "hello"}]}
            ])
        );
        assert!(wire.get("generationConfig").is_none());
    }

    #[test]
    fn a_system_message_after_the_conversation_starts_is_refused() {
        let err = refused(&request(vec![
            text(Role::User, "hi"),
            text(Role::System, "late"),
        ]));
        assert!(matches!(err, Error::NotImplemented(_)), "{err}");
    }

    #[test]
    fn a_tool_loop_pairs_calls_with_results_by_name_and_merges_results() {
        let wire = render(&request(vec![
            text(Role::User, "weather in Paris and Rome?"),
            message(
                Role::Assistant,
                vec![
                    call("c1", "get_weather", r#"{"city":"Paris"}"#),
                    call("c2", "get_weather", r#"{"city":"Rome"}"#),
                ],
            ),
            message(Role::Tool, vec![result("c1", "20C", false)]),
            message(Role::Tool, vec![result("c2", "boom", true)]),
        ]));
        let contents = wire["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 3, "the two results are ONE user turn");
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(
            contents[1]["parts"][0]["functionCall"],
            json!({"id": "c1", "name": "get_weather", "args": {"city": "Paris"}})
        );
        assert_eq!(contents[2]["role"], "user");
        assert_eq!(
            contents[2]["parts"],
            json!([
                {"functionResponse": {"id": "c1", "name": "get_weather", "response": {"result": "20C"}}},
                {"functionResponse": {"id": "c2", "name": "get_weather", "response": {"error": "boom"}}}
            ])
        );
    }

    #[test]
    fn a_tool_result_with_no_matching_call_is_refused_by_id() {
        let err = refused(&request(vec![message(
            Role::Tool,
            vec![result("ghost", "x", false)],
        )]));
        assert!(matches!(err, Error::InvalidInput(_)), "{err}");
        assert!(err.to_string().contains("ghost"), "{err}");
    }

    #[test]
    fn tool_arguments_must_be_a_json_object() {
        for bad in ["not json", "[1]", "7", "null"] {
            let err = refused(&request(vec![message(
                Role::Assistant,
                vec![call("c1", "f", bad)],
            )]));
            assert!(matches!(err, Error::InvalidInput(_)), "{bad}: {err}");
            assert!(err.to_string().contains("c1"), "{err}");
        }
        // No arguments at all is an empty object, not an error.
        let wire = render(&request(vec![message(
            Role::Assistant,
            vec![call("c1", "f", "")],
        )]));
        assert_eq!(
            wire["contents"][0]["parts"][0]["functionCall"]["args"],
            json!({})
        );
    }

    #[test]
    fn thinking_with_a_tool_result_is_not_refused() {
        let mut req = request(vec![
            message(Role::Assistant, vec![call("c1", "f", "{}")]),
            message(Role::Tool, vec![result("c1", "ok", false)]),
        ]);
        req.reasoning = Some(types::ReasoningConfig {
            enabled: Some(true),
            ..Default::default()
        });
        render_request(&req, "google").expect("Google's own error explains this case");
    }

    #[test]
    fn inline_media_renders_and_unsendable_media_is_refused() {
        let b64 = |media_type: &str| MediaSource::Base64 {
            media_type: media_type.into(),
            data: "aGk=".into(),
        };
        let wire = render(&request(vec![message(
            Role::User,
            vec![
                ContentBlock::Image {
                    source: b64("image/png"),
                    detail: None,
                    cache: None,
                },
                ContentBlock::Audio {
                    source: b64("audio/wav"),
                    format: Some("wav".into()),
                },
            ],
        )]));
        assert_eq!(
            wire["contents"][0]["parts"],
            json!([
                {"inlineData": {"mimeType": "image/png", "data": "aGk="}},
                {"inlineData": {"mimeType": "audio/wav", "data": "aGk="}}
            ])
        );
        for source in [
            b64(""),
            MediaSource::Url {
                url: "https://example.com/a.png".into(),
            },
            MediaSource::ProviderFile {
                id: "file-1".into(),
            },
        ] {
            let err = refused(&request(vec![message(
                Role::User,
                vec![ContentBlock::Image {
                    source,
                    detail: None,
                    cache: None,
                }],
            )]));
            assert!(
                matches!(err, Error::NotImplemented(_) | Error::InvalidInput(_)),
                "{err}"
            );
        }
    }

    #[test]
    fn a_block_from_another_protocol_is_refused_by_type() {
        let err = refused(&request(vec![message(
            Role::User,
            vec![ContentBlock::Unknown(
                json!({"type": "refusal", "refusal": "no"}),
            )],
        )]));
        assert!(err.to_string().contains("refusal"), "{err}");
    }

    #[test]
    fn another_providers_reasoning_is_dropped_and_an_emptied_turn_is_skipped() {
        let wire = render(&request(vec![
            text(Role::User, "hi"),
            message(
                Role::Assistant,
                vec![ContentBlock::Reasoning {
                    detail: types::ReasoningDetail::Summary {
                        summary: "s".into(),
                    },
                    provenance: Some("openai-responses-v1".into()),
                    id: None,
                }],
            ),
        ]));
        assert_eq!(wire["contents"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn what_has_no_google_slot_is_refused() {
        let mut cached = request(vec![text(Role::User, "hi")]);
        cached.cache = Some(CacheHint { ttl: None });
        assert!(matches!(refused(&cached), Error::NotImplemented(_)));

        let block_cached = request(vec![message(
            Role::User,
            vec![ContentBlock::Text {
                text: "hi".into(),
                cache: Some(CacheHint { ttl: None }),
            }],
        )]);
        assert!(matches!(refused(&block_cached), Error::NotImplemented(_)));

        let mut serial = request(vec![text(Role::User, "hi")]);
        serial.parallel_tool_calls = Some(false);
        assert!(matches!(refused(&serial), Error::NotImplemented(_)));
    }

    #[test]
    fn sampling_moves_under_generation_config() {
        let mut req = request(vec![text(Role::User, "hi")]);
        req.params = types::GenerationParams {
            max_tokens: Some(256),
            temperature: Some(0.5),
            top_p: Some(0.9),
            top_k: Some(40),
            stop: vec!["END".into()],
        };
        assert_eq!(
            render(&req)["generationConfig"],
            json!({
                "stopSequences": ["END"],
                "maxOutputTokens": 256,
                "temperature": 0.5,
                "topP": 0.9,
                "topK": 40
            })
        );
    }

    #[test]
    fn response_formats_render_as_a_mime_type_and_optional_schema() {
        let schema = json!({"type": "object"});
        let mut req = request(vec![text(Role::User, "hi")]);
        req.response_format = Some(types::ResponseFormat::JsonSchema {
            name: "n".into(),
            schema: schema.clone(),
            strict: Some(true),
        });
        assert_eq!(
            render(&req)["generationConfig"],
            json!({"responseMimeType": "application/json", "responseJsonSchema": schema})
        );
        // Unlike Anthropic, a schema-less JSON mode has a Google spelling.
        req.response_format = Some(types::ResponseFormat::JsonObject);
        assert_eq!(
            render(&req)["generationConfig"],
            json!({"responseMimeType": "application/json"})
        );
        req.response_format = Some(types::ResponseFormat::Text);
        assert!(render(&req).get("generationConfig").is_none());
    }

    #[test]
    fn reasoning_maps_onto_thinking_config() {
        let thinking = |reasoning: types::ReasoningConfig| {
            let mut req = request(vec![text(Role::User, "hi")]);
            req.reasoning = Some(reasoning);
            render(&req)["generationConfig"]["thinkingConfig"].clone()
        };
        assert_eq!(
            thinking(types::ReasoningConfig {
                budget_tokens: Some(1024),
                exclude: Some(false),
                ..Default::default()
            }),
            json!({"includeThoughts": true, "thinkingBudget": 1024})
        );
        assert_eq!(
            thinking(types::ReasoningConfig {
                effort: Some("low".into()),
                exclude: Some(true),
                ..Default::default()
            }),
            json!({"includeThoughts": false, "thinkingLevel": "LOW"})
        );
        // An explicit off wins over a budget.
        assert_eq!(
            thinking(types::ReasoningConfig {
                enabled: Some(false),
                budget_tokens: Some(1024),
                ..Default::default()
            }),
            json!({"thinkingBudget": 0})
        );
        // Enabled with nothing else says nothing: Google's default is on.
        let mut req = request(vec![text(Role::User, "hi")]);
        req.reasoning = Some(types::ReasoningConfig {
            enabled: Some(true),
            ..Default::default()
        });
        assert!(render(&req).get("generationConfig").is_none());
    }

    #[test]
    fn a_budget_too_large_for_int32_is_refused() {
        let mut req = request(vec![text(Role::User, "hi")]);
        req.reasoning = Some(types::ReasoningConfig {
            budget_tokens: Some(u32::MAX),
            ..Default::default()
        });
        assert!(matches!(refused(&req), Error::InvalidInput(_)));
    }

    #[test]
    fn tools_render_as_one_declaration_list_with_json_schema() {
        let schema = json!({"type": "object", "properties": {"city": {"type": "string"}}});
        let mut req = request(vec![text(Role::User, "hi")]);
        req.tools = vec![types::ToolDef {
            name: "get_weather".into(),
            description: Some("d".into()),
            input_schema: schema.clone(),
            strict: Some(true),
            cache: None,
            ext: ProviderExt::new(),
        }];
        req.tool_choice = Some(types::ToolChoice::Tool {
            name: "get_weather".into(),
        });
        let wire = render(&req);
        assert_eq!(
            wire["tools"],
            json!([{"functionDeclarations": [
                {"name": "get_weather", "description": "d", "parametersJsonSchema": schema}
            ]}])
        );
        assert_eq!(
            wire["toolConfig"],
            json!({"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}})
        );
    }

    #[test]
    fn tool_choices_map_onto_calling_modes() {
        for (choice, mode) in [
            (types::ToolChoice::Auto, "AUTO"),
            (types::ToolChoice::None, "NONE"),
            (types::ToolChoice::Required, "ANY"),
        ] {
            let mut req = request(vec![text(Role::User, "hi")]);
            req.tool_choice = Some(choice);
            assert_eq!(
                render(&req)["toolConfig"]["functionCallingConfig"]["mode"],
                mode
            );
        }
    }

    #[test]
    fn a_tool_with_no_schema_is_refused() {
        let mut req = request(vec![text(Role::User, "hi")]);
        req.tools = vec![types::ToolDef {
            name: "hosted".into(),
            description: None,
            input_schema: Value::Null,
            strict: None,
            cache: None,
            ext: ProviderExt::new(),
        }];
        assert!(matches!(refused(&req), Error::NotImplemented(_)));
    }

    #[test]
    fn ext_fields_ride_through_but_never_shadow_a_typed_field() {
        let mut req = request(vec![text(Role::User, "hi")]);
        req.ext.insert(
            "google".into(),
            json!({
                "safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT"}],
                "contents": "would duplicate the typed key"
            }),
        );
        let wire = render(&req);
        assert_eq!(
            wire["safetySettings"],
            json!([{"category": "HARM_CATEGORY_HARASSMENT"}])
        );
        assert_eq!(
            wire["contents"],
            json!([{"role": "user", "parts": [{"text": "hi"}]}])
        );
    }
}
