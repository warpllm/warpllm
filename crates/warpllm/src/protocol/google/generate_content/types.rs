//! Google `generateContent` request and response types, tracking
//! <https://ai.google.dev/api/generate-content>.
//!
//! Wire names are camelCase, which is what Google emits. Every struct captures
//! fields it does not model in an `unknown_fields` catch-all
//! ([`UnknownFields`]), so anything Google can put on the wire survives a round
//! trip through these types.
//!
//! # What is modelled, and what rides the catch-all
//!
//! Only what a conversion is expected to read or set, the same rule as
//! [`anthropic::messages::types`](crate::protocol::anthropic::messages::types).
//! `topK`, `thinkingConfig`, `groundingMetadata` and the rest are untyped until
//! a conversion needs one.
//!
//! # Optionality
//!
//! Plain `Option` with the key skipped when `None`. Google omits absent fields
//! rather than sending `null`; if a provider is found that does, the field
//! moves to the three-state form described in [`crate::protocol`].
//!
//! # A part is whichever field is present
//!
//! Unlike Anthropic's content blocks, a [`Part`] has no `type` tag: Google
//! documents it as a oneof, so exactly one of `text`, `inlineData`,
//! `functionCall` or `functionResponse` is expected. A struct of optionals
//! holds that losslessly, and leaves the "exactly one" check to the
//! conversions, which can say what is wrong.
//!
//! # No codegen derives
//!
//! Same as the Anthropic types, and for the same reason (#55).
//!
//! The model is not a field: on AI Studio and on Vertex it is in the URL.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::protocol::UnknownFields;

/// The body of `POST .../models/{model}:generateContent`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentRequest {
    pub contents: Vec<Content>,
    /// The system prompt. Gemini has no system role, so it is a top-level
    /// [`Content`] whose parts are text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

/// One turn: a role and its parts.
///
/// `role` is `user` or `model`. It is optional on the wire (a system
/// instruction carries none), and tool results come back as a `user` turn.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Content {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default)]
    pub parts: Vec<Part>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

/// One piece of a [`Content`]. See the module docs on why this is not tagged.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inline_data: Option<Blob>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_response: Option<FunctionResponse>,
    /// `true` on a part that carries the model's thinking rather than its answer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought: Option<bool>,
    /// Opaque; must be handed back on the next turn for the model to continue
    /// its reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

/// Inline media: base64 `data` of a given MIME type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Blob {
    pub mime_type: String,
    pub data: String,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

/// A tool call the model made. `args` is a JSON object, not a string.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<Value>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

/// A tool result handed back to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    pub response: Value,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

/// Sampling and output settings. These live here, not at the top level.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<Value>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

/// A tool the model may call. Only function declarations are modelled; the
/// built-in tools (search, code execution) ride the catch-all.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_declarations: Option<Vec<FunctionDeclaration>>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionDeclaration {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// An OpenAPI-subset schema. `parametersJsonSchema`, the full JSON Schema
    /// alternative, rides the catch-all.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_calling_config: Option<FunctionCallingConfig>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

/// `mode` is `AUTO`, `ANY` or `NONE`. Kept a string so a new mode parses.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallingConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

/// The body of a `generateContent` reply.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponse {
    /// Absent when the prompt itself was blocked; see `promptFeedback`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates: Option<Vec<Candidate>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<UsageMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Why a prompt was blocked, if it was. Untyped until a conversion needs it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_feedback: Option<Value>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    /// `STOP`, `MAX_TOKENS`, `SAFETY` and so on. A string so a new reason parses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

/// Token counts. `thoughtsTokenCount` is billed as output.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_token_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates_token_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_token_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_content_token_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thoughts_token_count: Option<u32>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Deserialize, reserialize, and require the same bytes back.
    fn round_trip<T>(body: Value) -> T
    where
        T: Serialize + serde::de::DeserializeOwned,
    {
        let parsed: T = serde_json::from_value(body.clone())
            .unwrap_or_else(|e| panic!("{body} failed to deserialize: {e}"));
        assert_eq!(serde_json::to_value(&parsed).unwrap(), body, "lossy");
        parsed
    }

    #[test]
    fn a_request_round_trips() {
        let req: GenerateContentRequest = round_trip(json!({
            "contents": [
                {"role": "user", "parts": [{"text": "hi"}]},
                {"role": "model", "parts": [
                    {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}
                ]},
                {"role": "user", "parts": [
                    {"functionResponse": {"name": "get_weather", "response": {"temp": 20}}}
                ]}
            ],
            "systemInstruction": {"parts": [{"text": "be brief"}]},
            "generationConfig": {
                "temperature": 0.5,
                "topP": 0.9,
                "maxOutputTokens": 1024,
                "stopSequences": ["END"]
            },
            "tools": [{"functionDeclarations": [
                {"name": "get_weather", "description": "d",
                 "parameters": {"type": "OBJECT", "properties": {}}}
            ]}],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY"}}
        }));
        assert_eq!(req.contents.len(), 3);
        assert!(req.system_instruction.unwrap().role.is_none());
    }

    #[test]
    fn a_response_round_trips() {
        let resp: GenerateContentResponse = round_trip(json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "hello"}]},
                "finishReason": "STOP",
                "index": 0
            }],
            "usageMetadata": {
                "promptTokenCount": 3,
                "candidatesTokenCount": 2,
                "totalTokenCount": 5,
                "thoughtsTokenCount": 7
            },
            "modelVersion": "gemini-x",
            "responseId": "abc"
        }));
        let usage = resp.usage_metadata.unwrap();
        assert_eq!(usage.prompt_token_count, Some(3));
        assert_eq!(usage.thoughts_token_count, Some(7));
    }

    /// Every part kind is read into its own field and nothing else.
    #[test]
    fn each_part_kind_lands_in_its_own_field() {
        let p: Part = round_trip(json!({"text": "a"}));
        assert!(p.text.is_some() && p.inline_data.is_none());
        let p: Part = round_trip(json!({"inlineData": {"mimeType": "image/png", "data": "aGk="}}));
        assert_eq!(p.inline_data.unwrap().mime_type, "image/png");
        let p: Part = round_trip(json!({"functionCall": {"id": "1", "name": "f", "args": {}}}));
        assert_eq!(p.function_call.unwrap().name, "f");
        let p: Part = round_trip(json!({"text": "t", "thought": true, "thoughtSignature": "sig"}));
        assert_eq!(p.thought, Some(true));
    }

    /// What this crate does not model survives, at every level.
    #[test]
    fn unmodelled_fields_survive_at_every_level() {
        let resp: GenerateContentResponse = round_trip(json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "x", "futurePartField": 1}]},
                "finishReason": "STOP",
                "groundingMetadata": {"webSearchQueries": ["q"]}
            }],
            "usageMetadata": {"promptTokenCount": 1, "toolUsePromptTokenCount": 4},
            "someNewTopLevelField": true
        }));
        assert!(resp.unknown_fields.contains_key("someNewTopLevelField"));
        let cand = &resp.candidates.unwrap()[0];
        assert!(cand.unknown_fields.contains_key("groundingMetadata"));
    }

    /// A blocked prompt has no candidates, which must not be an error.
    #[test]
    fn a_blocked_prompt_has_no_candidates() {
        let resp: GenerateContentResponse = round_trip(json!({
            "promptFeedback": {"blockReason": "SAFETY"}
        }));
        assert!(resp.candidates.is_none());
    }

    /// A finish reason or mode warpllm has never heard of still parses.
    #[test]
    fn open_string_enums_accept_new_values() {
        let c: Candidate = round_trip(json!({"finishReason": "SOME_NEW_REASON"}));
        assert_eq!(c.finish_reason.as_deref(), Some("SOME_NEW_REASON"));
        round_trip::<FunctionCallingConfig>(json!({"mode": "VALIDATED"}));
    }
}
