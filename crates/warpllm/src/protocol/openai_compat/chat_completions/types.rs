//! OpenAI-compatible request and response types — warpllm's permissive
//! superset of the OpenAI shapes (the authoritative OpenAI spec lives in
//! <https://github.com/openai/openai-openapi>; changes to it belong
//! upstream). Other providers translate to and from these, so callers see
//! one shape regardless of upstream.
//!
//! The response section is a field-for-field copy of the `chat.completion`
//! object, and the streaming section the same for the `chat.completion.chunk`
//! a `stream: true` request is answered with; both keep upstream object names
//! and field order:
//! - Response object: <https://developers.openai.com/api/reference/resources/chat>
//! - Request parameters: <https://platform.openai.com/docs/api-reference/chat/create>
//!
//! Naming: types matching a named schema in the official OpenAPI spec
//! (<https://github.com/openai/openai-openapi>) use that exact name; types the
//! spec leaves anonymous/inline keep a local descriptive name.
//!
//! Every struct captures fields it doesn't model in an `unknown_fields`
//! catch-all; see [`UnknownFields`].

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

pub use crate::protocol::UnknownFields;
use crate::protocol::present_or_null;

// Codegen policy: Rust and Serde are the contract. Codegen-only attributes may
// enable derives or compensate for representation details a generator cannot
// infer (such as output optionality). Do not add literal unions, schema enums,
// const values, ranges, or other constraints that are absent from the Rust
// field type; OpenAI-compatible providers extend these values independently.

// Optionality is spelled three ways here — see [`present_or_null`] and the
// rule stated with it in `crate::protocol`. The codegen attributes on the
// third kind say that rule in each generator's terms: `ts(optional)` unwraps
// the outer `Option` only, so TypeScript keeps the inner `| null`, and
// `x-nullable-when-present` tells `warpllm-codegen` to leave the null branch a
// serialized schema otherwise drops from an omittable field.

// ---------------------------------------------------------------------------
// Response — the `chat.completion` object
// <https://developers.openai.com/api/reference/resources/chat>
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct CreateChatCompletionResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub created: u64,
    /// Echoes the caller-supplied `provider/model` string.
    pub model: String,
    /// Conventionally `"chat.completion"`; compatible providers may differ.
    pub object: String,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub moderation: Option<Option<ChatCompletionModeration>>,
    /// Provider-defined service tier identifier.
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub service_tier: Option<Option<String>>,
    /// Deprecated upstream but still returned; passed through as-is.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub system_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub usage: Option<CompletionUsage>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct Choice {
    /// Provider-defined reason that generation stopped.
    pub finish_reason: String,
    pub index: u32,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub logprobs: Option<Option<ChoiceLogprobs>>,
    pub message: ChatCompletionResponseMessage,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// OpenAI documents both arrays as required and nullable; OpenAI-compatible
/// backends are looser still — DeepSeek omits `refusal` entirely — so both are
/// optional here, and all three states a caller can receive stay distinct.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChoiceLogprobs {
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub content: Option<Option<Vec<ChatCompletionTokenLogprob>>>,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub refusal: Option<Option<Vec<ChatCompletionTokenLogprob>>>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionTokenLogprob {
    pub token: String,
    pub bytes: Option<Vec<u8>>,
    pub logprob: f64,
    pub top_logprobs: Vec<TopLogprob>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct TopLogprob {
    pub token: String,
    pub bytes: Option<Vec<u8>>,
    pub logprob: f64,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionResponseMessage {
    pub content: Option<String>,
    pub refusal: Option<String>,
    /// Conventionally `"assistant"`; compatible providers may differ.
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub annotations: Option<Vec<Annotation>>,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub audio: Option<Option<ChatCompletionAudio>>,
    /// Deprecated upstream in favor of `tool_calls`.
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub function_call: Option<Option<FunctionCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub tool_calls: Option<Vec<ChatCompletionMessageToolCallUnion>>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct Annotation {
    /// Conventionally `"url_citation"`; compatible providers may differ.
    #[serde(rename = "type")]
    pub r#type: String,
    pub url_citation: AnnotationURLCitation,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

// Exact upstream name; OpenAI-shape fidelity outranks Rust acronym casing.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct AnnotationURLCitation {
    pub end_index: u32,
    pub start_index: u32,
    pub title: String,
    pub url: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// Deprecated upstream in favor of tool calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct FunctionCall {
    /// JSON-encoded arguments; model-generated, so may be invalid JSON.
    pub arguments: String,
    pub name: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// Function- or custom-shaped tool call.
// Untagged so the structs own the `type` field: an internal serde tag would be
// captured by their flattened `unknown_fields` and emitted twice. Each
// variant's required `function`/`custom` field keeps dispatch unambiguous even
// when a provider sends a nonstandard `type` string.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
#[serde(untagged)]
pub enum ChatCompletionMessageToolCallUnion {
    Function(ChatCompletionMessageToolCall),
    Custom(ChatCompletionMessageCustomToolCall),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionMessageToolCall {
    pub id: String,
    /// Conventionally `"function"`; compatible providers may differ.
    #[serde(rename = "type")]
    pub r#type: String,
    pub function: Function,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct Function {
    /// JSON-encoded arguments; model-generated, so may be invalid JSON.
    pub arguments: String,
    pub name: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionMessageCustomToolCall {
    pub id: String,
    /// Conventionally `"custom"`; compatible providers may differ.
    #[serde(rename = "type")]
    pub r#type: String,
    pub custom: Custom,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct Custom {
    pub input: String,
    pub name: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionAudio {
    pub id: String,
    /// Base64-encoded audio bytes.
    pub data: String,
    pub expires_at: u64,
    pub transcript: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct CompletionUsage {
    pub completion_tokens: u32,
    pub prompt_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct CompletionTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub accepted_prediction_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub audio_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub reasoning_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub rejected_prediction_tokens: Option<u32>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct PromptTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub audio_tokens: Option<u32>,
    /// Unadjusted number of prompt tokens written to cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub cache_write_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub cached_tokens: Option<u32>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// Moderation results for the request input and the generated output.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionModeration {
    pub input: ModerationOutcome,
    pub output: ModerationOutcome,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

// The docs define one shared ModerationResults/Error pair used by both
// `input` and `output`.

/// Union of [`ChatCompletionModerationResults`] or
/// [`ChatCompletionModerationError`]. Untagged so the structs own the `type`
/// field exactly as received; their required fields are disjoint, so dispatch
/// is unambiguous. The spec leaves this union unnamed; only this enum's name is
/// Rust-side.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
#[serde(untagged)]
pub enum ModerationOutcome {
    ModerationResults(ChatCompletionModerationResults),
    Error(ChatCompletionModerationError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionModerationResults {
    pub model: String,
    pub results: Vec<ModerationResultBody>,
    /// Conventionally `"moderation_results"`; compatible providers may differ.
    #[serde(rename = "type")]
    pub r#type: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// One verdict in `ChatCompletionModerationResults.results`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ModerationResultBody {
    pub categories: HashMap<String, bool>,
    /// Values are input types, e.g. `"text"` or `"image"`.
    pub category_applied_input_types: HashMap<String, Vec<String>>,
    pub category_scores: HashMap<String, f64>,
    pub flagged: bool,
    pub model: String,
    /// Conventionally `"moderation_result"`; compatible providers may differ.
    #[serde(rename = "type")]
    pub r#type: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// Moderation error.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionModerationError {
    pub code: String,
    pub message: String,
    /// Conventionally `"error"`; compatible providers may differ.
    #[serde(rename = "type")]
    pub r#type: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

// ---------------------------------------------------------------------------
// Streaming response — the `chat.completion.chunk` object
// <https://developers.openai.com/api/reference/resources/chat>
// ---------------------------------------------------------------------------

// A chunk repeats the completion's envelope and replaces the finished message
// with a delta, so the shapes that describe neither — logprobs, usage,
// moderation — are the completion's own types, not copies of them. The request
// is [`CreateChatCompletionRequest`] with `stream` set; only the reply differs.
//
// A stream is where the three spellings of optionality above earn their keep:
// `finish_reason` is null on every chunk until the choice ends, `usage` is
// null on every chunk but the last, and `logprobs` is null on all of them.

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct CreateChatCompletionStreamResponse {
    /// Identifies the completion, not the chunk: every chunk of one stream
    /// carries the same value.
    pub id: String,
    /// Empty on the usage-only chunk `stream_options.include_usage` adds.
    pub choices: Vec<StreamChoice>,
    pub created: u64,
    /// Echoes the caller-supplied `provider/model` string.
    pub model: String,
    /// Conventionally `"chat.completion.chunk"`; compatible providers may
    /// differ.
    pub object: String,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub moderation: Option<Option<ChatCompletionModeration>>,
    /// Provider-defined service tier identifier.
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub service_tier: Option<Option<String>>,
    /// Deprecated upstream but still returned; passed through as-is.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub system_fingerprint: Option<String>,
    /// Totals for the whole request, not the chunk. Present only when the
    /// caller asks for it, and `null` on every chunk before the last.
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub usage: Option<Option<CompletionUsage>>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// One choice of a [`CreateChatCompletionStreamResponse`]. The spec leaves it
/// unnamed; only this name is warpllm's.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct StreamChoice {
    pub delta: ChatCompletionStreamResponseDelta,
    /// Provider-defined reason that generation stopped, and `null` until it
    /// does — which is every chunk of this choice but the last.
    pub finish_reason: Option<String>,
    /// Which choice this chunk continues; chunks of an `n > 1` request
    /// interleave.
    pub index: u32,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub logprobs: Option<Option<ChoiceLogprobs>>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// The streaming counterpart of [`ChatCompletionResponseMessage`]: every field
/// is a fragment, so every field is optional — a chunk carrying only `role`,
/// only a slice of `content`, or nothing at all is ordinary.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionStreamResponseDelta {
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub content: Option<Option<String>>,
    /// Deprecated upstream in favor of `tool_calls`.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub function_call: Option<DeltaFunctionCall>,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub refusal: Option<Option<String>>,
    /// Provider-defined role, sent on the chunk that opens a choice.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub tool_calls: Option<Vec<ChatCompletionMessageToolCallChunk>>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// Deprecated upstream in favor of tool calls. Unlike [`FunctionCall`], both
/// halves arrive in pieces, so neither is required.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct DeltaFunctionCall {
    /// A fragment of the JSON-encoded arguments; concatenating the fragments
    /// of one call yields the whole, which is model-generated and so may still
    /// be invalid JSON.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub arguments: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub name: Option<String>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// A slice of one tool call. `index` — not `id`, which arrives once — is what
/// says which call in the choice the fragment belongs to.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionMessageToolCallChunk {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub function: Option<ToolCallChunkFunction>,
    /// Conventionally `"function"`; compatible providers may differ.
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub r#type: Option<String>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// The function half of a [`ChatCompletionMessageToolCallChunk`]. Unlike
/// [`Function`], both fields are fragments, so neither is required.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ToolCallChunkFunction {
    /// A fragment of the JSON-encoded arguments; concatenating the fragments
    /// of one call yields the whole, which is model-generated and so may still
    /// be invalid JSON.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub arguments: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub name: Option<String>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

// ---------------------------------------------------------------------------
// Request — the `POST /v1/chat/completions` parameters
// <https://platform.openai.com/docs/api-reference/chat/create>
// ---------------------------------------------------------------------------

// Doc comments on these types are REPUBLISHED: ts-rs copies them verbatim into
// the generated `.d.ts` that ships to npm. Rationale meant for this crate goes
// in `//` comments like this one, which ts-rs does not copy — a `///` here
// would put warpllm's internals, and any rustdoc `[link]` syntax that means
// nothing in TypeScript, into a published declaration file.
//
// `unknown_fields` is `ts(skip)` here, same as on responses. The method taking
// this request is generic in TypeScript and Mapping-based in Python, so both
// languages accept extensions without weakening response field checking.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct CreateChatCompletionRequest {
    /// Model string in `provider/model` form, e.g. `"openai/gpt-5.6"`.
    /// `#[serde(default)]` so a body sending only `models` (no `model`)
    /// deserializes; the field stays `String` and the generated
    /// TypeScript/Python types widen from required to optional. (ts-rs
    /// cannot mark a non-`Option` field optional, so warpllm-codegen patches
    /// the TypeScript declaration after generation.)
    #[serde(default)]
    pub model: String,
    /// warpllm extension: ordered candidate models for per-request
    /// failover. When present, the client tries each model in order on
    /// retryable errors; the first successful one serves the request.
    /// Consumed at ingest time; not forwarded upstream.
    ///
    /// The commit point differs by surface. Non-streaming: a whole reply,
    /// so any candidate that completes is the winner. Streaming: only
    /// connection setup up to the FIRST chunk — once a candidate yields its
    /// first chunk the stream is locked in, because chunks already emitted
    /// cannot be unsent (a mid-stream failure surfaces to the caller with
    /// no failover, and cannot be retried without duplicating output).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub models: Option<Vec<String>>,
    pub messages: Vec<ChatCompletionRequestMessage>,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub temperature: Option<Option<f64>>,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub max_tokens: Option<Option<u32>>,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub top_p: Option<Option<f64>>,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub stop: Option<Option<ChatCompletionStop>>,
    // The reply this asks for is CreateChatCompletionStreamResponse, above.
    // The shapes are defined; nothing reads a stream off the socket yet.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub stream: Option<bool>,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub stream_options: Option<Option<ChatCompletionStreamOptions>>,
    /// Tools the model may call.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub tools: Option<Vec<ChatCompletionTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub tool_choice: Option<ChatCompletionToolChoiceOption>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub response_format: Option<ChatCompletionResponseFormat>,
    /// How much reasoning to spend; provider-defined, commonly `"low"`,
    /// `"medium"`, `"high"`.
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub reasoning_effort: Option<Option<String>>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionRequestMessage {
    /// Provider-defined role; common values include `"system"` and `"user"`.
    pub role: String,
    // Optional AND nullable, and the difference is load-bearing here rather
    // than a nicety: `content` is OPTIONAL on an assistant turn that carries
    // only `tool_calls`, and REQUIRED-but-nullable on the legacy
    // `role: "function"` message, whose schema is `content: string | null`. A
    // form that read `null` as absent would forward
    // `{"role": "function", "name": "f"}` with the key gone, and a provider
    // holding OpenAI to its own schema is entitled to reject that.
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub content: Option<Option<ChatCompletionRequestMessageContent>>,
    /// Set on the assistant turn that called tools; each entry is answered by
    /// a later message with `role: "tool"` carrying the matching
    /// `tool_call_id`.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub tool_calls: Option<Vec<ChatCompletionMessageToolCallUnion>>,
    /// Which call this message answers. Set on `role: "tool"` messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub tool_call_id: Option<String>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

impl ChatCompletionRequestMessage {
    /// A message with something to say — the ordinary case.
    ///
    /// `content` is `Option<Option<_>>` because the wire distinguishes an
    /// absent key from an explicit `null`, and both are real: an assistant turn
    /// carrying only `tool_calls` omits it, and the legacy `role: "function"`
    /// message requires the key while permitting the null. A caller writing a
    /// prompt wants neither state, and should not have to spell out the two
    /// they are not choosing.
    ///
    /// The other shapes stay struct literals, because a message with no
    /// content is exactly where those two states start to matter.
    pub fn new(
        role: impl Into<String>,
        content: impl Into<ChatCompletionRequestMessageContent>,
    ) -> Self {
        Self {
            role: role.into(),
            content: Some(Some(content.into())),
            ..Default::default()
        }
    }
}

/// A message's content: one string, or a list of parts.
// Untagged for the same reason as ChatCompletionMessageToolCallUnion: the part
// structs own their `type` field, and an internal serde tag would be captured
// by their flattened `unknown_fields` and emitted twice. The two forms are not
// collapsed into one — a provider is sent back what it was given, and callers
// that pass a bare string see a bare string.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
#[serde(untagged)]
pub enum ChatCompletionRequestMessageContent {
    Text(String),
    Parts(Vec<ChatCompletionRequestMessageContentPart>),
}

impl From<&str> for ChatCompletionRequestMessageContent {
    fn from(text: &str) -> Self {
        Self::Text(text.to_string())
    }
}

impl From<String> for ChatCompletionRequestMessageContent {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

/// One part of a multimodal message.
// Untagged, and each variant's required field (`text`, `image_url`,
// `input_audio`, `file`, `refusal`) is what keeps dispatch unambiguous even
// when a provider sends a nonstandard `type` string. `Unknown` is last so a
// part shape warpllm does not model still reaches the provider verbatim.
//
// `Refusal` earns a variant despite having no gateway meaning — it round trips
// through `Unknown` just as well. What it buys is the PUBLISHED TypeScript:
// `Unknown` generates as `JsonValue`, whose object case is an index signature,
// and TypeScript never assigns an interface to one of those. So a caller
// holding OpenAI's own `ChatCompletionContentPartRefusal` could build a
// request warpllm accepts at runtime and could not typecheck.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
#[serde(untagged)]
pub enum ChatCompletionRequestMessageContentPart {
    Text(ChatCompletionRequestMessageContentPartText),
    Image(ChatCompletionRequestMessageContentPartImage),
    Audio(ChatCompletionRequestMessageContentPartAudio),
    File(ChatCompletionRequestMessageContentPartFile),
    Refusal(ChatCompletionRequestMessageContentPartRefusal),
    Unknown(serde_json::Value),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionRequestMessageContentPartRefusal {
    /// Conventionally `"refusal"`; compatible providers may differ.
    #[serde(rename = "type")]
    pub r#type: String,
    pub refusal: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionRequestMessageContentPartText {
    /// Conventionally `"text"`; compatible providers may differ.
    #[serde(rename = "type")]
    pub r#type: String,
    pub text: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionRequestMessageContentPartImage {
    /// Conventionally `"image_url"`; compatible providers may differ.
    #[serde(rename = "type")]
    pub r#type: String,
    pub image_url: ImageUrl,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ImageUrl {
    /// An `https:` URL, or a `data:` URL carrying base64 image bytes.
    pub url: String,
    /// Provider-defined fidelity hint; commonly `"auto"`, `"low"`, `"high"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub detail: Option<String>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionRequestMessageContentPartAudio {
    /// Conventionally `"input_audio"`; compatible providers may differ.
    #[serde(rename = "type")]
    pub r#type: String,
    pub input_audio: InputAudio,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct InputAudio {
    /// Base64-encoded audio bytes.
    pub data: String,
    /// Provider-defined container; commonly `"wav"`, `"mp3"`.
    pub format: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionRequestMessageContentPartFile {
    /// Conventionally `"file"`; compatible providers may differ.
    #[serde(rename = "type")]
    pub r#type: String,
    pub file: FileContent,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct FileContent {
    /// Base64-encoded file bytes, for a file sent inline.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub file_data: Option<String>,
    /// A file already uploaded to the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub filename: Option<String>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// One tool the model may call.
// `function` is optional rather than required so that a tool shape warpllm
// does not model — a hosted tool like `{"type": "web_search"}`, or a custom
// tool — parses and reaches the provider instead of failing the whole request.
// Its payload rides `unknown_fields`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionTool {
    /// Conventionally `"function"`; compatible providers may differ.
    #[serde(rename = "type")]
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub function: Option<FunctionObject>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct FunctionObject {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub description: Option<String>,
    /// JSON Schema for the arguments, passed through verbatim.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub parameters: Option<serde_json::Value>,
    /// Whether the provider must adhere exactly to the schema.
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub strict: Option<Option<bool>>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// Whether, and which, tool the model must call.
// Untagged, and `Mode` stays a plain string per this module's codegen policy:
// providers extend the vocabulary past OpenAI's `"none"`/`"auto"`/`"required"`
// independently.
//
// `Unknown` is last and is what makes this a superset rather than a reading of
// one vocabulary: OpenAI alone already sends shapes that are neither a mode
// nor a named function — `{"type": "allowed_tools", ...}` and
// `{"type": "custom", "custom": {...}}` — and a form without it would reject a
// request the vendor's own SDK builds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
#[serde(untagged)]
pub enum ChatCompletionToolChoiceOption {
    Named(ChatCompletionNamedToolChoice),
    Mode(String),
    Unknown(serde_json::Value),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionNamedToolChoice {
    /// Conventionally `"function"`; compatible providers may differ.
    #[serde(rename = "type")]
    pub r#type: String,
    pub function: ToolChoiceFunction,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ToolChoiceFunction {
    pub name: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// The shape the reply must take.
// Untagged, with the schema-bearing variant first: `JsonSchema` requires a
// `json_schema` field, and `Simple` covers `{"type": "text"}` and
// `{"type": "json_object"}` alike without closing the vocabulary.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
#[serde(untagged)]
pub enum ChatCompletionResponseFormat {
    JsonSchema(ResponseFormatJsonSchema),
    Simple(ResponseFormatSimple),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ResponseFormatSimple {
    /// Conventionally `"text"` or `"json_object"`; compatible providers may
    /// differ.
    #[serde(rename = "type")]
    pub r#type: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ResponseFormatJsonSchema {
    /// Conventionally `"json_schema"`; compatible providers may differ.
    #[serde(rename = "type")]
    pub r#type: String,
    pub json_schema: JsonSchemaDefinition,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct JsonSchemaDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub description: Option<String>,
    /// JSON Schema for the reply, passed through verbatim.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub schema: Option<serde_json::Value>,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "codegen", ts(optional))]
    #[cfg_attr(feature = "codegen", schemars(extend("x-nullable-when-present" = true)))]
    pub strict: Option<Option<bool>>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// Where generation stops: one sequence, or several.
// Both forms are held rather than normalized to a list, for the same reason a
// message's content keeps its arrived-in form — and because a bare string is
// what OpenAI's own examples show, so a form that took only a list would
// reject an ordinary request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
#[serde(untagged)]
pub enum ChatCompletionStop {
    One(String),
    Many(Vec<String>),
}

impl ChatCompletionStop {
    /// The sequences, however they were spelled.
    pub(crate) fn sequences(&self) -> Vec<String> {
        match self {
            Self::One(stop) => vec![stop.clone()],
            Self::Many(stops) => stops.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionStreamOptions {
    /// Ask for a final chunk carrying the whole request's token totals.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub include_usage: Option<bool>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// OpenAI-compatible proxies often omit the optional response fields;
    /// every one of them must deserialize as absent, not error.
    #[test]
    fn minimal_response_body_deserializes() {
        let completion: CreateChatCompletionResponse = serde_json::from_str(
            r#"{
                "id": "chatcmpl-123",
                "choices": [{
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": "hi", "refusal": null, "role": "assistant"}
                }],
                "created": 1700000000,
                "model": "gpt-5.6",
                "object": "chat.completion"
            }"#,
        )
        .unwrap();
        assert!(completion.moderation.is_none());
        assert!(completion.service_tier.is_none());
        assert!(completion.system_fingerprint.is_none());
        assert!(completion.usage.is_none());
        assert!(completion.choices[0].logprobs.is_none());
        assert!(completion.choices[0].message.tool_calls.is_none());
        // Absent optionals must also stay off the wire when re-serialized.
        let wire = serde_json::to_value(&completion).unwrap();
        assert!(wire.get("usage").is_none());
        assert!(wire.get("moderation").is_none());
        assert!(wire["choices"][0].get("logprobs").is_none());
        assert!(wire["choices"][0]["message"].get("tool_calls").is_none());
    }

    /// Fields we don't model must be captured into `unknown_fields` — at
    /// every nesting level — and re-emitted verbatim, not dropped.
    #[test]
    fn unknown_fields_round_trip() {
        let body = serde_json::json!({
            "id": "chatcmpl-123",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "hi",
                    "refusal": null,
                    "role": "assistant",
                    "reasoning_content": "step by step"
                },
                "new_choice_field": true
            }],
            "created": 1700000000,
            "model": "gpt-5.6",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 1,
                "prompt_tokens": 2,
                "total_tokens": 3,
                "new_usage_field": 7
            },
            "new_top_level_field": "surprise"
        });

        let completion: CreateChatCompletionResponse =
            serde_json::from_value(body.clone()).unwrap();

        assert_eq!(completion.unknown_fields["new_top_level_field"], "surprise");
        assert_eq!(
            completion.choices[0].unknown_fields["new_choice_field"],
            true
        );
        assert_eq!(
            completion.choices[0].message.unknown_fields["reasoning_content"],
            "step by step"
        );
        assert_eq!(
            completion.usage.as_ref().unwrap().unknown_fields["new_usage_field"],
            7
        );
        assert_eq!(serde_json::to_value(&completion).unwrap(), body);
    }

    /// These are strings in the compatibility protocol, not closed enums.
    /// Providers routinely add values before any shared specification does.
    #[test]
    fn provider_defined_string_values_round_trip() {
        let body = serde_json::json!({
            "id": "chatcmpl-123",
            "choices": [{
                "finish_reason": "provider_finished_elsewhere",
                "index": 0,
                "message": {
                    "content": "hi",
                    "refusal": null,
                    "role": "critic",
                    "annotations": [{
                        "type": "provider_citation",
                        "url_citation": {
                            "end_index": 2,
                            "start_index": 0,
                            "title": "Example",
                            "url": "https://example.com"
                        }
                    }],
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "provider_function",
                        "function": {"arguments": "{}", "name": "search"}
                    }]
                }
            }],
            "created": 1,
            "model": "provider/model",
            "object": "provider.chat.result",
            "service_tier": "provider_experimental"
        });

        let completion: CreateChatCompletionResponse =
            serde_json::from_value(body.clone()).unwrap();

        assert_eq!(completion.object, "provider.chat.result");
        assert_eq!(completion.choices[0].message.role, "critic");
        assert_eq!(serde_json::to_value(&completion).unwrap(), body);
    }

    /// A body with every documented field must round-trip byte-for-byte
    /// (as JSON values), proving the deep copy is complete and lossless.
    #[test]
    fn full_response_body_round_trips() {
        let body = serde_json::json!({
            "id": "chatcmpl-123",
            "choices": [{
                "finish_reason": "tool_calls",
                "index": 0,
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
                    "refusal": null,
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
                            "function": {"arguments": "{\"q\":1}", "name": "search"}
                        },
                        {
                            "id": "call-2",
                            "type": "custom",
                            "custom": {"input": "raw text", "name": "my_tool"}
                        }
                    ]
                }
            }],
            "created": 1_700_000_000,
            "model": "gpt-5.6",
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
                "output": {
                    "type": "error",
                    "code": "moderation_unavailable",
                    "message": "try again"
                }
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
                }
            }
        });

        let completion: CreateChatCompletionResponse =
            serde_json::from_value(body.clone()).unwrap();

        let message = &completion.choices[0].message;
        let tool_calls = message.tool_calls.as_ref().unwrap();
        assert!(matches!(
            &tool_calls[0],
            ChatCompletionMessageToolCallUnion::Function(f) if f.function.name == "search"
        ));
        assert!(matches!(
            &tool_calls[1],
            ChatCompletionMessageToolCallUnion::Custom(c) if c.custom.input == "raw text"
        ));
        let moderation = completion.moderation.as_ref().unwrap().as_ref().unwrap();
        assert!(matches!(
            &moderation.input,
            ModerationOutcome::ModerationResults(r) if !r.results[0].flagged
        ));
        assert!(matches!(
            &moderation.output,
            ModerationOutcome::Error(e) if e.code == "moderation_unavailable"
        ));
        let usage = completion.usage.as_ref().unwrap();
        assert_eq!(
            usage
                .prompt_tokens_details
                .as_ref()
                .unwrap()
                .cache_write_tokens,
            Some(2)
        );

        assert_eq!(serde_json::to_value(&completion).unwrap(), body);
    }

    /// Every field OpenAI documents as optional AND nullable, sent as an
    /// explicit null, on the whole-completion side. Each one has to come back
    /// as the null it arrived as — that is what makes these shapes a superset
    /// of OpenAI's own rather than a permissive reader of them, and it is
    /// checked from the other end by `bindings/node/__test__/
    /// openai-oracle.ts`.
    #[test]
    fn an_explicit_null_in_a_completion_is_not_an_absent_field() {
        let body = serde_json::json!({
            "id": "chatcmpl-123",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "logprobs": {"content": null, "refusal": null},
                "message": {
                    "content": "hi",
                    "refusal": null,
                    "role": "assistant",
                    "audio": null,
                    "function_call": null
                }
            }],
            "created": 1_700_000_000,
            "model": "gpt-5.6",
            "object": "chat.completion",
            "moderation": null,
            "service_tier": null
        });

        let completion: CreateChatCompletionResponse =
            serde_json::from_value(body.clone()).unwrap();

        assert!(matches!(completion.moderation, Some(None)));
        assert!(matches!(completion.service_tier, Some(None)));
        assert!(matches!(completion.choices[0].logprobs, Some(Some(_))));
        assert!(matches!(completion.choices[0].message.audio, Some(None)));
        assert_eq!(serde_json::to_value(&completion).unwrap(), body);
    }

    /// The chunk a provider that sends nothing optional would send: a delta
    /// with no fields at all, and a choice that has not finished.
    #[test]
    fn minimal_chunk_deserializes() {
        let chunk: CreateChatCompletionStreamResponse = serde_json::from_str(
            r#"{
                "id": "chatcmpl-123",
                "choices": [{"delta": {}, "finish_reason": null, "index": 0}],
                "created": 1700000000,
                "model": "gpt-5.6",
                "object": "chat.completion.chunk"
            }"#,
        )
        .unwrap();
        assert!(chunk.moderation.is_none());
        assert!(chunk.service_tier.is_none());
        assert!(chunk.system_fingerprint.is_none());
        assert!(chunk.usage.is_none());
        assert!(chunk.choices[0].logprobs.is_none());
        assert!(chunk.choices[0].delta.content.is_none());
        assert!(chunk.choices[0].delta.role.is_none());
        assert!(chunk.choices[0].delta.tool_calls.is_none());
        // Absent optionals stay off the wire; `finish_reason` is required
        // upstream, so it goes back out as the null it arrived as.
        let wire = serde_json::to_value(&chunk).unwrap();
        assert!(wire.get("usage").is_none());
        assert!(wire["choices"][0].get("logprobs").is_none());
        assert!(wire["choices"][0]["delta"].as_object().unwrap().is_empty());
        assert_eq!(wire["choices"][0]["finish_reason"], serde_json::Value::Null);
    }

    /// The distinction the streaming shapes exist to keep: OpenAI sends
    /// `"usage": null` on every chunk but the last and `"logprobs": null` on
    /// all of them, so an explicit null must come back out as one — while a
    /// key the provider never sent must stay absent.
    #[test]
    fn an_explicit_null_is_not_an_absent_field() {
        let nulled: CreateChatCompletionStreamResponse = serde_json::from_str(
            r#"{
                "id": "chatcmpl-123",
                "choices": [{
                    "delta": {"content": null, "refusal": null},
                    "finish_reason": null,
                    "index": 0,
                    "logprobs": null
                }],
                "created": 1700000000,
                "model": "gpt-5.6",
                "object": "chat.completion.chunk",
                "usage": null
            }"#,
        )
        .unwrap();

        assert!(matches!(nulled.usage, Some(None)));
        assert!(matches!(nulled.choices[0].logprobs, Some(None)));
        assert_eq!(nulled.choices[0].delta.content, Some(None));
        let wire = serde_json::to_value(&nulled).unwrap();
        assert_eq!(wire["usage"], serde_json::Value::Null);
        assert_eq!(wire["choices"][0]["logprobs"], serde_json::Value::Null);
        assert_eq!(
            wire["choices"][0]["delta"]["content"],
            serde_json::Value::Null
        );
    }

    /// Fields we don't model must be captured at every nesting level of a
    /// chunk too — `obfuscation`, which OpenAI adds to live streams and no
    /// specification names, is the case that made this concrete.
    #[test]
    fn chunk_unknown_fields_round_trip() {
        let body = serde_json::json!({
            "id": "chatcmpl-123",
            "choices": [{
                "delta": {
                    "content": "hi",
                    "reasoning_content": "step by step"
                },
                "finish_reason": null,
                "index": 0,
                "new_choice_field": true
            }],
            "created": 1700000000,
            "model": "gpt-5.6",
            "object": "chat.completion.chunk",
            "obfuscation": "8Xk2p"
        });

        let chunk: CreateChatCompletionStreamResponse =
            serde_json::from_value(body.clone()).unwrap();

        assert_eq!(chunk.unknown_fields["obfuscation"], "8Xk2p");
        assert_eq!(chunk.choices[0].unknown_fields["new_choice_field"], true);
        assert_eq!(
            chunk.choices[0].delta.unknown_fields["reasoning_content"],
            "step by step"
        );
        assert_eq!(serde_json::to_value(&chunk).unwrap(), body);
    }

    /// A chunk with every documented field must round-trip losslessly, the
    /// same proof the completion object carries — including the tool-call
    /// fragments, whose `index` is what reassembles them.
    #[test]
    fn full_chunk_round_trips() {
        let body = serde_json::json!({
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
                        "function": {"arguments": "{\"q\":", "name": "search"},
                        "type": "function"
                    }]
                },
                "finish_reason": "tool_calls",
                "index": 0,
                "logprobs": {
                    "content": [{
                        "token": "Hello",
                        "bytes": [72, 101, 108, 108, 111],
                        "logprob": -0.1,
                        "top_logprobs": [{"token": "Hello", "bytes": null, "logprob": -0.1}]
                    }],
                    "refusal": null
                }
            }],
            "created": 1_700_000_000,
            "model": "gpt-5.6",
            "object": "chat.completion.chunk",
            "moderation": {
                "input": {
                    "type": "error",
                    "code": "moderation_unavailable",
                    "message": "try again"
                },
                "output": {
                    "type": "error",
                    "code": "moderation_unavailable",
                    "message": "try again"
                }
            },
            "service_tier": "default",
            "system_fingerprint": "fp_44709d6fcb",
            "usage": {
                "completion_tokens": 12,
                "prompt_tokens": 9,
                "total_tokens": 21
            }
        });

        let chunk: CreateChatCompletionStreamResponse =
            serde_json::from_value(body.clone()).unwrap();

        let delta = &chunk.choices[0].delta;
        let tool_call = &delta.tool_calls.as_ref().unwrap()[0];
        assert_eq!(tool_call.index, 0);
        assert_eq!(
            tool_call.function.as_ref().unwrap().arguments.as_deref(),
            Some("{\"q\":")
        );
        assert_eq!(
            chunk.choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
        let logprobs = chunk.choices[0]
            .logprobs
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap();
        let content = logprobs.content.as_ref().unwrap().as_ref().unwrap();
        assert_eq!(content[0].token, "Hello");
        assert_eq!(
            chunk.usage.as_ref().unwrap().as_ref().unwrap().total_tokens,
            21
        );

        assert_eq!(serde_json::to_value(&chunk).unwrap(), body);
    }

    /// Byte-level, not Value-level: some providers' prompt caches hash the
    /// request bytes, so unknown fields must keep the caller's key order
    /// (requires serde_json's `preserve_order` feature).
    #[test]
    fn unknown_field_order_is_preserved() {
        let mut request = CreateChatCompletionRequest {
            model: "openai/gpt-5.6".into(),
            ..Default::default()
        };
        request
            .unknown_fields
            .insert("zeta".into(), serde_json::json!(1));
        request
            .unknown_fields
            .insert("alpha".into(), serde_json::json!(2));

        let wire = serde_json::to_string(&request).unwrap();
        assert!(
            wire.find("zeta").unwrap() < wire.find("alpha").unwrap(),
            "insertion order lost: {wire}"
        );
    }
}
