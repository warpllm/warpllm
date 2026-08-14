//! What a caller can reach, using only what the crate root hands them.
//!
//! An integration test compiles against the PUBLIC surface, which is the only
//! place the facade's claim can be checked: `lib.rs` says it re-exports
//! everything a caller must name to make a call, and unit tests cannot notice
//! when that stops being true — they see the whole crate.
//!
//! So the check is the import list below. Every type a fully-loaded request
//! reaches is named through `warpllm::`, and a field whose type is only
//! reachable at `protocol::openai_compat::chat_completions::types` fails to
//! compile here rather than in someone's editor.

use warpllm::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCallUnion,
    ChatCompletionNamedToolChoice, ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContent, ChatCompletionRequestMessageContentPart,
    ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionResponseFormat, ChatCompletionStop, ChatCompletionStreamOptions,
    ChatCompletionTool, ChatCompletionToolChoiceOption, CreateChatCompletionRequest, Function,
    FunctionObject, ImageUrl, JsonSchemaDefinition, ResponseFormatJsonSchema, ToolChoiceFunction,
};

/// Every typed request feature at once, built the way a caller would.
///
/// Deliberately exhaustive rather than minimal: the point is not that this
/// request is useful, it is that none of it needed an import from a module
/// path that reads as internal.
#[test]
fn a_fully_loaded_request_is_buildable_from_the_crate_root() {
    let request = CreateChatCompletionRequest {
        model: "openai/gpt-5.6".into(),
        messages: vec![
            // A multimodal user turn.
            ChatCompletionRequestMessage {
                role: "user".into(),
                content: Some(Some(ChatCompletionRequestMessageContent::Parts(vec![
                    ChatCompletionRequestMessageContentPart::Text(
                        ChatCompletionRequestMessageContentPartText {
                            r#type: "text".into(),
                            text: "what is in this image, and what is the weather there".into(),
                            unknown_fields: Default::default(),
                        },
                    ),
                    ChatCompletionRequestMessageContentPart::Image(
                        ChatCompletionRequestMessageContentPartImage {
                            r#type: "image_url".into(),
                            image_url: ImageUrl {
                                url: "https://example.com/a.png".into(),
                                detail: Some("high".into()),
                                unknown_fields: Default::default(),
                            },
                            unknown_fields: Default::default(),
                        },
                    ),
                ]))),
                ..Default::default()
            },
            // The assistant turn that called a tool, replayed back in.
            ChatCompletionRequestMessage {
                role: "assistant".into(),
                content: Some(None),
                tool_calls: Some(vec![ChatCompletionMessageToolCallUnion::Function(
                    ChatCompletionMessageToolCall {
                        id: "call_1".into(),
                        r#type: "function".into(),
                        function: Function {
                            arguments: r#"{"city":"Seoul"}"#.into(),
                            name: "get_weather".into(),
                            unknown_fields: Default::default(),
                        },
                        unknown_fields: Default::default(),
                    },
                )]),
                ..Default::default()
            },
            // ...and the answer to it.
            ChatCompletionRequestMessage {
                role: "tool".into(),
                content: Some(Some("18C".into())),
                tool_call_id: Some("call_1".into()),
                ..Default::default()
            },
        ],
        stop: Some(Some(ChatCompletionStop::One("END".into()))),
        stream_options: Some(Some(ChatCompletionStreamOptions {
            include_usage: Some(true),
            unknown_fields: Default::default(),
        })),
        tools: Some(vec![ChatCompletionTool {
            r#type: "function".into(),
            function: Some(FunctionObject {
                name: "get_weather".into(),
                description: Some("Look up the weather".into()),
                parameters: Some(serde_json::json!({"type": "object"})),
                strict: Some(Some(true)),
                unknown_fields: Default::default(),
            }),
            unknown_fields: Default::default(),
        }]),
        tool_choice: Some(ChatCompletionToolChoiceOption::Named(
            ChatCompletionNamedToolChoice {
                r#type: "function".into(),
                function: ToolChoiceFunction {
                    name: "get_weather".into(),
                    unknown_fields: Default::default(),
                },
                unknown_fields: Default::default(),
            },
        )),
        response_format: Some(ChatCompletionResponseFormat::JsonSchema(
            ResponseFormatJsonSchema {
                r#type: "json_schema".into(),
                json_schema: JsonSchemaDefinition {
                    name: "weather".into(),
                    description: None,
                    schema: Some(serde_json::json!({"type": "object"})),
                    strict: Some(Some(true)),
                    unknown_fields: Default::default(),
                },
                unknown_fields: Default::default(),
            },
        )),
        reasoning_effort: Some(Some("high".into())),
        ..Default::default()
    };

    // Serializing is not the claim — building it was. This only pins that the
    // shapes reached above are the ones that go on the wire.
    let wire = serde_json::to_value(&request).unwrap();
    assert_eq!(wire["messages"].as_array().unwrap().len(), 3);
    assert_eq!(wire["tools"][0]["function"]["name"], "get_weather");
    assert_eq!(wire["stop"], "END");
    // The assistant turn's explicit null survives to the wire, where the
    // legacy `function` role needs it and an assistant turn is entitled to it.
    assert!(wire["messages"][1]["content"].is_null());
}
