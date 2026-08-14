//! Anthropic Messages request and response types, tracking
//! <https://platform.claude.com/docs/en/api/messages> and the streaming events
//! at <https://platform.claude.com/docs/en/build-with-claude/streaming>.
//!
//! Naming follows Anthropic's own schema names where it publishes one
//! (`Message`, `Usage`, `ToolChoice`); shapes it leaves anonymous keep a local
//! descriptive name.
//!
//! Every struct captures fields it does not model in an `unknown_fields`
//! catch-all ([`UnknownFields`]), and the optionality of what it DOES model
//! follows the three-state rule stated in [`crate::protocol`]. Between them,
//! anything Anthropic can put on the wire survives a round trip through these
//! types — which is what the fixture and proptest suites in
//! `tests/protocol/anthropic/` assert.
//!
//! # What is modelled, and what rides the catch-all
//!
//! Only what the gateway request and response have a typed home for. This is
//! an EGRESS protocol: nothing but `render_request` constructs a
//! [`CreateMessageRequest`], so a field no renderer sets earns nothing by
//! being typed and still reaches Anthropic through `unknown_fields`. So
//! `top_k`, `metadata`, `service_tier`, `container`, `inference_geo`,
//! `output_config` and a top-level `cache_control` are all deliberately
//! untyped. Add one when a conversion needs to set or read it, not before.
//!
//! # Two structural differences from chat completions
//!
//! Worth stating here because they shape the conversions rather than just the
//! types: a [`Message`] carries **no `created` timestamp** (so
//! `ChatResponse::created` is `None` on this protocol), and Anthropic has **no
//! multi-choice concept** — one request yields one reply, so `completions`
//! always holds exactly one element.
//!
//! # No codegen derives
//!
//! Unlike [`openai_compat`](crate::protocol::openai_compat), these types carry
//! no `ts_rs`/`schemars` attributes. Generated TypeScript is one flat
//! `types.ts`, and both protocols define a `Message`, a `Usage` and a tool-call
//! type that would collide in it. Resolving that needs a per-protocol output
//! directory — #55 — and until then a derive here would be an attribute
//! nothing reads.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::protocol::{UnknownFields, present_or_null};

// Union discriminators: every `type`-tagged enum below is internally tagged on
// that field, with a final untagged `Unknown(Value)` arm. This is the opposite
// of the fully-untagged choice openai_compat makes, and for a reason that does
// not transfer: THAT protocol is a shape many vendors implement independently,
// so its `type` strings are unreliable and dispatch has to key on each
// variant's required payload field instead. Anthropic's protocol has one
// implementer, so `type` is authoritative — and here it must be, because
// `image` and `document` blocks are distinguishable by nothing else.
//
// The `Unknown` arm is what keeps that trust cheap: a block, delta or event
// type Anthropic adds tomorrow parses verbatim and serializes back unchanged
// rather than failing the whole body.

/// Generates a [`Deserialize`] impl for a `type`-tagged union whose `Unknown`
/// arm means exactly one thing: **a string discriminator warpllm does not
/// recognize**. Everything else fails.
///
/// Three payloads could reach that arm, and only the first belongs there:
///
/// | payload | means | verdict |
/// |---|---|---|
/// | `{"type": "message_pause", …}` | a shape from a later API version | `Unknown` |
/// | `{"error": {…}}` | no discriminator — malformed | error |
/// | `{"type": 5}` | a discriminator that is not a name — malformed | error |
/// | `{"type": "error"}` with no `error` | a KNOWN shape that did not parse | error |
///
/// Serde's own untagged fallback collapses all four into `Unknown`: it tries
/// the tagged variants and falls through on ANY failure, and it has no way to
/// distinguish "the tag names something new" from "there is no usable tag".
/// Forward compatibility is a promise about **values of the discriminator**,
/// not about payloads that lack one.
///
/// The collapse is not a harmless relabelling. Every case above is terminal in
/// the streaming reader, and `Unknown` is not: `transport::ends_the_stream`
/// stops recognizing the terminal event, so the reader waits for a
/// `message_stop` that is never coming and reports the socket closing as a
/// TRUNCATION — the provider's stated reason replaced by a wrong one, for a
/// stream that was neither truncated nor silent. For a content block it means
/// a malformed `text` block becomes an opaque value a conversion cannot read.
///
/// Requiring an object with a string `type` costs nothing in permissiveness:
/// every union built with this macro is discriminated that way by definition,
/// so a payload without one is not a member of it in any API version.
///
/// Only the reading side is generated. Serialization stays derived, because
/// writing the tag is unambiguous and has no fallback to get wrong.
///
/// The tag strings appear here AND in each enum's `rename_all`. A mismatch
/// would make a variant unreachable, which the round-trip tests below catch by
/// asserting that no documented value lands in `Unknown`.
macro_rules! unknown_means_an_unknown_tag {
    ($name:ident { $($tag:literal => $variant:ident),+ $(,)? }) => {
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                let mut value = Value::deserialize(deserializer)?;
                let tag = match value.get("type") {
                    Some(Value::String(tag)) => tag.clone(),
                    // Neither of these is a shape from a later API version, so
                    // neither may pass as one.
                    Some(other) => {
                        return Err(serde::de::Error::custom(format!(
                            "`type` must be a string naming the variant, found {other}"
                        )));
                    }
                    None => return Err(serde::de::Error::missing_field("type")),
                };
                match tag.as_str() {
                    $($tag => {
                        // The tag belongs to the enum, not to the variant's
                        // body. Leaving it would land it in that body's
                        // `unknown_fields` and emit it twice on the way out.
                        if let Some(object) = value.as_object_mut() {
                            object.remove("type");
                        }
                        serde_json::from_value(value)
                            .map(Self::$variant)
                            .map_err(serde::de::Error::custom)
                    })+
                    _ => Ok(Self::Unknown(value)),
                }
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CreateMessageRequest {
    pub model: String,
    pub messages: Vec<InputMessage>,
    /// REQUIRED upstream, which is why it is a plain `u32` rather than an
    /// `Option`. Anthropic has no default: a request without it is a 400. The
    /// gateway's `max_tokens` is optional, so resolving one is `render_request`'s
    /// job and it fails loudly rather than inventing a ceiling that would
    /// silently truncate replies.
    pub max_tokens: u32,
    /// The system prompt is a TOP-LEVEL parameter here, not a message with a
    /// `system` role. Hoisting it out of the gateway's message list, and back
    /// in on the way home, is the conversion's job.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Spelled `stop_sequences`, not `stop`, and list-only — there is no bare
    /// string form to accept.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

/// `string | TextBlockParam[]`, and the two forms must round trip as
/// themselves: collapsing the string into a one-element block list would be
/// lossless in meaning and still fail the byte-for-byte fixture check.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMessage {
    /// `"user"` or `"assistant"`. Anthropic has no `system` or `tool` role:
    /// the system prompt is a top-level parameter, and a tool result is a
    /// `tool_result` BLOCK inside a user turn.
    pub role: String,
    pub content: MessageContent,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

/// `string | ContentBlock[]` — see [`SystemPrompt`] for why both forms stay.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

// ---------------------------------------------------------------------------
// Content blocks
//
// One enum for both directions. Anthropic's input and output block shapes
// differ slightly — `cache_control` is an input concern, citations an output
// one — and modelling them separately would double every variant to express a
// difference `unknown_fields` already absorbs.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text(TextBlock),
    Image(ImageBlock),
    Document(DocumentBlock),
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlock),
    Thinking(ThinkingBlock),
    RedactedThinking(RedactedThinkingBlock),
    /// A block type warpllm does not model — a server tool's result, a
    /// `fallback` marker, whatever ships next — kept verbatim.
    #[serde(untagged)]
    Unknown(Value),
}

unknown_means_an_unknown_tag!(ContentBlock {
    "text" => Text,
    "image" => Image,
    "document" => Document,
    "tool_use" => ToolUse,
    "tool_result" => ToolResult,
    "thinking" => Thinking,
    "redacted_thinking" => RedactedThinking,
});

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBlock {
    pub text: String,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_control: Option<Option<CacheControl>>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageBlock {
    pub source: Source,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_control: Option<Option<CacheControl>>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentBlock {
    pub source: Source,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub title: Option<Option<String>>,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_control: Option<Option<CacheControl>>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUseBlock {
    pub id: String,
    pub name: String,
    /// A decoded JSON OBJECT, where chat completions sends the same thing as a
    /// string of JSON. That difference is the one lossy seam in the tool path,
    /// and it belongs to the conversion: a model's arguments are not always
    /// parseable, and this field cannot hold text that is not.
    pub input: Value,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_control: Option<Option<CacheControl>>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultBlock {
    pub tool_use_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ToolResultContent>,
    /// The tool failed. Chat completions has no equivalent bit, which is why
    /// a conversation carrying one cannot cross to that protocol without a
    /// policy — see #57.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_control: Option<Option<CacheControl>>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

/// A tool's output: text, or blocks, because Anthropic permits images in a
/// result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// Reasoning, with a cryptographic `signature` over it.
///
/// INVARIANT: a run of these must go back exactly as it arrived — same order,
/// same text, signature untouched — or Anthropic rejects the turn. Treat the
/// run as opaque.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingBlock {
    pub thinking: String,
    /// Absent while streaming until the `signature_delta` that closes the
    /// block, which is why it is optional on a type that also models the
    /// assembled form.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

/// Reasoning withheld as ciphertext. Never inspect it; replay it verbatim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedactedThinkingBlock {
    pub data: String,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

/// Where a media block's bytes come from.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Source {
    Base64(Base64Source),
    Url(UrlSource),
    /// Document text inline. The gateway's `MediaSource` has no counterpart —
    /// its three cases are a URL, base64 bytes, and a provider file id — so
    /// this one is Anthropic's alone.
    Text(TextSource),
    File(FileSource),
    #[serde(untagged)]
    Unknown(Value),
}

unknown_means_an_unknown_tag!(Source {
    "base64" => Base64,
    "url" => Url,
    "text" => Text,
    "file" => File,
});

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Base64Source {
    pub media_type: String,
    /// Raw base64, with no `data:` URL wrapper.
    pub data: String,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UrlSource {
    pub url: String,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSource {
    pub media_type: String,
    pub data: String,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSource {
    pub file_id: String,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

/// A prompt-cache breakpoint. Positional: it marks the block it sits on.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    /// Conventionally `"ephemeral"`.
    #[serde(rename = "type")]
    pub r#type: String,
    /// `"5m"` or `"1h"`; absent means Anthropic's default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

/// A tool definition.
///
/// Untagged rather than `type`-tagged, because `type` is what a CUSTOM tool
/// omits: `{"name", "description", "input_schema"}` carries no discriminator
/// at all, while a server tool is `{"type": "web_search_20260209", "name"}`
/// with no schema. So the presence of `input_schema` is the discriminator, and
/// it is a required field on the only variant warpllm models.
///
/// `large_enum_variant` is allowed rather than fixed: boxing `CustomTool`
/// would add a heap allocation per tool to shrink the elements of a `Vec` that
/// is already on the heap, built once per request and serialized once. The
/// waste is the size gap times the number of SERVER tools in one request,
/// which is single digits.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
#[allow(clippy::large_enum_variant)]
pub enum Tool {
    Custom(CustomTool),
    /// A server tool, or any other shape with no `input_schema`. Passed
    /// through verbatim: the gateway's `ToolDef` requires a name AND a schema,
    /// so these have no neutral representation to translate into — #54.
    Other(Value),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the arguments, passed through verbatim. Anthropic
    /// spells this `input_schema` where chat completions spells it
    /// `parameters`; the name is the only difference.
    pub input_schema: Value,
    /// Whether the model must match the schema exactly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_control: Option<Option<CacheControl>>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    /// The model decides.
    Auto(ToolChoiceMode),
    /// Some tool must be called — Anthropic's spelling of "required".
    Any(ToolChoiceMode),
    /// No tool may be called.
    None(ToolChoiceMode),
    /// This tool must be called.
    Tool(ToolChoiceTool),
    #[serde(untagged)]
    Unknown(Value),
}

unknown_means_an_unknown_tag!(ToolChoice {
    "auto" => Auto,
    "any" => Any,
    "none" => None,
    "tool" => Tool,
});

/// The body shared by the three tool choices that name no tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolChoiceMode {
    /// Anthropic's inverted spelling of chat completions'
    /// `parallel_tool_calls`. Neither has a gateway home, so both ride `ext`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_parallel_tool_use: Option<bool>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_parallel_tool_use: Option<bool>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

// ---------------------------------------------------------------------------
// Thinking
// ---------------------------------------------------------------------------

/// How much the model may think, in whichever of the two vocabularies the
/// routed model accepts.
///
/// The two are NOT interchangeable and the model decides which: Claude 4.7 and
/// later reject `enabled`, Claude 4.5 and earlier reject `adaptive`. Both are
/// modelled because both are current somewhere in the roster.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThinkingConfig {
    /// Extended thinking with an explicit token budget.
    Enabled(ThinkingEnabled),
    /// Adaptive thinking; the model sets its own budget.
    Adaptive(ThinkingAdaptive),
    Disabled(ThinkingDisabled),
    #[serde(untagged)]
    Unknown(Value),
}

unknown_means_an_unknown_tag!(ThinkingConfig {
    "enabled" => Enabled,
    "adaptive" => Adaptive,
    "disabled" => Disabled,
});

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingEnabled {
    pub budget_tokens: u32,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThinkingAdaptive {
    /// `"summarized"` or `"omitted"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<String>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThinkingDisabled {
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    /// Conventionally `"message"`.
    #[serde(rename = "type")]
    pub r#type: String,
    /// Always `"assistant"` on a reply.
    pub role: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    /// Required and nullable, so it always serializes: `null` is what a
    /// `message_start` event carries, and means the reply is still being
    /// generated rather than that it stopped for no reason.
    pub stop_reason: Option<String>,
    /// Which of the caller's `stop_sequences` was hit, when `stop_reason` is
    /// `stop_sequence`. Required and nullable for the same reason.
    pub stop_sequence: Option<String>,
    pub usage: Usage,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

/// Token counts.
///
/// Note what is NOT here: a total. Anthropic sends none, so anything needing
/// one computes it — and `input_tokens` EXCLUDES both cache counts, so the sum
/// has to add all three.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Optional AND nullable: a `message_start` omits the cache counts
    /// entirely, and a cache-aware reply may send either a number or `null`.
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_creation_input_tokens: Option<Option<u32>>,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_read_input_tokens: Option<Option<u32>>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

// ---------------------------------------------------------------------------
// Streaming
//
// Anthropic streams NAMED events — `event: content_block_delta` — and repeats
// the name in each payload's own `type`. warpllm reads the payload and ignores
// the `event:` line, which is what lets the shared SSE framing stay a plain
// `data:` reader; see `crate::http::SseFrames`.
//
// There is no `[DONE]` sentinel. A stream is over when `message_stop` arrives,
// so recognizing the end means decoding an event rather than matching a
// payload — and a socket that closes before one is a TRUNCATED reply.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageStreamEvent {
    /// The reply's `id`, `model` and input token counts, once, with empty
    /// content. Everything after it is a delta, so a reader that needs any of
    /// those on a later chunk has to remember them.
    MessageStart(MessageStartEvent),
    ContentBlockStart(ContentBlockStartEvent),
    ContentBlockDelta(ContentBlockDeltaEvent),
    ContentBlockStop(ContentBlockStopEvent),
    /// Top-level changes: the stop reason, and cumulative output tokens.
    MessageDelta(MessageDeltaEvent),
    MessageStop(MessageStopEvent),
    /// Keepalive. May arrive anywhere, any number of times.
    Ping(PingEvent),
    /// A failure AFTER a 200 — an overload mid-generation, say. It ends the
    /// stream, and it is the reason a 200 does not guarantee a whole reply.
    Error(ErrorEvent),
    #[serde(untagged)]
    Unknown(Value),
}

unknown_means_an_unknown_tag!(MessageStreamEvent {
    "message_start" => MessageStart,
    "content_block_start" => ContentBlockStart,
    "content_block_delta" => ContentBlockDelta,
    "content_block_stop" => ContentBlockStop,
    "message_delta" => MessageDelta,
    "message_stop" => MessageStop,
    "ping" => Ping,
    "error" => Error,
});

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageStartEvent {
    pub message: Message,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlockStartEvent {
    /// The block's position in the final `content` array — counting ALL
    /// blocks, not just tool calls. Chat completions counts tool calls only,
    /// which makes the two indices different numbers for the same call.
    pub index: u32,
    pub content_block: ContentBlock,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlockDeltaEvent {
    pub index: u32,
    pub delta: ContentBlockDelta,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlockStopEvent {
    pub index: u32,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDeltaEvent {
    pub delta: MessageDelta,
    pub usage: MessageDeltaUsage,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MessageStopEvent {
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PingEvent {
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEvent {
    pub error: super::super::error::ErrorBody,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

/// What a `message_delta` changes about the reply as a whole.
///
/// Both fields are optional AND nullable rather than merely nullable. Anthropic
/// documents the nullability but not the presence, and its own example sends
/// both keys with `stop_sequence` null — so the three-state spelling is what
/// holds whatever actually arrives without asserting more than the spec says.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MessageDelta {
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub stop_reason: Option<Option<String>>,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub stop_sequence: Option<Option<String>>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

/// The usage on a `message_delta`, which is NOT [`Usage`]: it carries
/// `output_tokens` and need carry nothing else, since a stream's input counts
/// arrived once on `message_start`.
///
/// The counts are CUMULATIVE, not per-event — a reader that sums them
/// overcounts the reply.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MessageDeltaUsage {
    pub output_tokens: u32,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub input_tokens: Option<Option<u32>>,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_creation_input_tokens: Option<Option<u32>>,
    #[serde(
        default,
        deserialize_with = "present_or_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_read_input_tokens: Option<Option<u32>>,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockDelta {
    TextDelta(TextDelta),
    /// A slice of a tool call's arguments, as TEXT. The pieces concatenate to
    /// JSON; no single one is valid JSON on its own, so a reader must
    /// accumulate rather than parse per event.
    InputJsonDelta(InputJsonDelta),
    ThinkingDelta(ThinkingDelta),
    /// Arrives once, just before the `content_block_stop` that closes a
    /// thinking block.
    SignatureDelta(SignatureDelta),
    /// Anything else, verbatim — `citations_delta` among them, which has no
    /// gateway home to be lifted into.
    #[serde(untagged)]
    Unknown(Value),
}

unknown_means_an_unknown_tag!(ContentBlockDelta {
    "text_delta" => TextDelta,
    "input_json_delta" => InputJsonDelta,
    "thinking_delta" => ThinkingDelta,
    "signature_delta" => SignatureDelta,
});

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDelta {
    pub text: String,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputJsonDelta {
    pub partial_json: String,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingDelta {
    pub thinking: String,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureDelta {
    pub signature: String,
    #[serde(flatten)]
    pub unknown_fields: UnknownFields,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Deserialize → reserialize is the identity, which every assertion below
    /// leans on.
    fn round_trip<T>(body: Value) -> T
    where
        T: Serialize + serde::de::DeserializeOwned,
    {
        let parsed: T = serde_json::from_value(body.clone())
            .unwrap_or_else(|e| panic!("{body} failed to deserialize: {e}"));
        assert_eq!(serde_json::to_value(&parsed).unwrap(), body, "lossy");
        parsed
    }

    /// The discriminator that untagged dispatch could not have made: these two
    /// blocks differ in `type` and in NOTHING else.
    #[test]
    fn an_image_and_a_document_are_told_apart_by_their_type() {
        let source = json!({"type": "base64", "media_type": "image/png", "data": "aGk="});
        assert!(matches!(
            round_trip::<ContentBlock>(json!({"type": "image", "source": source})),
            ContentBlock::Image(_)
        ));
        let source = json!({"type": "base64", "media_type": "application/pdf", "data": "aGk="});
        assert!(matches!(
            round_trip::<ContentBlock>(json!({"type": "document", "source": source})),
            ContentBlock::Document(_)
        ));
    }

    /// Every block variant is reachable by its own tag.
    ///
    /// This is what catches a typo in the tag table that generates the reader:
    /// a misspelled tag makes its variant unreachable, and the value lands in
    /// `Unknown` — where it would still round trip, so nothing but this notices.
    #[test]
    fn every_block_variant_is_reachable_by_its_tag() {
        let source = json!({"type": "base64", "media_type": "image/png", "data": "aGk="});
        let blocks = [
            json!({"type": "text", "text": "hi"}),
            json!({"type": "image", "source": source}),
            json!({"type": "document", "source": source}),
            json!({"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}),
            json!({"type": "tool_result", "tool_use_id": "toolu_1"}),
            json!({"type": "thinking", "thinking": "hmm"}),
            json!({"type": "redacted_thinking", "data": "EroB"}),
        ];
        for block in blocks {
            assert!(
                !matches!(
                    round_trip::<ContentBlock>(block.clone()),
                    ContentBlock::Unknown(_)
                ),
                "{block} fell through to Unknown"
            );
        }
    }

    /// ...and every media source, which is the other union nothing else sweeps.
    #[test]
    fn every_source_variant_is_reachable_by_its_tag() {
        let sources = [
            json!({"type": "base64", "media_type": "image/png", "data": "aGk="}),
            json!({"type": "url", "url": "https://example.com/a.png"}),
            json!({"type": "text", "media_type": "text/plain", "data": "hi"}),
            json!({"type": "file", "file_id": "file_1"}),
        ];
        for source in sources {
            assert!(
                !matches!(round_trip::<Source>(source.clone()), Source::Unknown(_)),
                "{source} fell through to Unknown"
            );
        }
    }

    /// A KNOWN tag carrying a body that does not parse is a DECODE FAILURE,
    /// not an unknown shape.
    ///
    /// Serde's own untagged fallback cannot draw that line — it falls through
    /// on any failure — and the difference is not cosmetic. A malformed
    /// `error` event read as `Unknown` is an event
    /// `transport::ends_the_stream` does not recognize, so the stream waits
    /// for a `message_stop` that is never coming and reports the socket
    /// closing as a TRUNCATION: the provider's stated reason replaced by a
    /// wrong one. The same applies to a block a conversion would then be
    /// unable to read.
    #[test]
    fn a_known_tag_with_a_malformed_body_is_refused() {
        let malformed = [
            json!({"type": "error"}),
            json!({"type": "message_delta", "delta": {}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta"}}),
        ];
        for body in malformed {
            let parsed = serde_json::from_value::<MessageStreamEvent>(body.clone());
            assert!(parsed.is_err(), "{body} was accepted as {:?}", parsed.ok());
        }
        // ...and the same rule one level down, on blocks and on sources.
        assert!(serde_json::from_value::<ContentBlock>(json!({"type": "text"})).is_err());
        assert!(
            serde_json::from_value::<ContentBlock>(json!({"type": "tool_use", "id": "t"})).is_err()
        );
        assert!(serde_json::from_value::<Source>(json!({"type": "url"})).is_err());
    }

    /// A payload with NO usable discriminator is malformed, not forward
    /// compatible.
    ///
    /// Forward compatibility is a promise about values of `type` — a name
    /// warpllm has not heard of yet. A payload that omits the key, or gives it
    /// something that is not a name, is not a member of the union in any API
    /// version, and letting it pass as `Unknown` has the same consequence as
    /// the malformed bodies above: a terminal event that stops being terminal,
    /// so a stream that ended is reported as one that broke.
    #[test]
    fn a_payload_without_a_string_discriminator_is_refused() {
        let malformed = [
            // The reviewer's case: `event: error` whose data lost its `type`.
            // The framing ignores the `event:` line, so this is all the reader
            // ever sees.
            json!({"error": {"type": "overloaded_error", "message": "Overloaded"}}),
            json!({"type": 5}),
            json!({"type": null}),
            json!({"type": ["error"]}),
            json!({}),
        ];
        for body in malformed {
            let parsed = serde_json::from_value::<MessageStreamEvent>(body.clone());
            assert!(parsed.is_err(), "{body} was accepted as {:?}", parsed.ok());
        }
        // A payload that is not even an object cannot be a union member.
        assert!(serde_json::from_value::<ContentBlock>(json!("text")).is_err());
        assert!(serde_json::from_value::<ContentBlock>(json!([])).is_err());
        assert!(serde_json::from_value::<Source>(json!({"url": "x"})).is_err());

        // ...while an unrecognized NAME is still the forward-compatible case,
        // which is the whole distinction being drawn.
        assert!(matches!(
            serde_json::from_value::<MessageStreamEvent>(json!({"type": "message_pause"})).unwrap(),
            MessageStreamEvent::Unknown(_)
        ));
    }

    /// A block type warpllm does not model reaches the provider verbatim
    /// rather than failing the message it sits in.
    #[test]
    fn an_unmodelled_block_survives_as_unknown() {
        let block = round_trip::<ContentBlock>(json!({
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [{"type": "web_search_result", "url": "https://example.com"}]
        }));
        assert!(matches!(block, ContentBlock::Unknown(_)));
    }

    /// ...and so does a field on a block warpllm DOES model.
    #[test]
    fn an_unmodelled_field_on_a_modelled_block_survives() {
        let block = round_trip::<ContentBlock>(json!({
            "type": "text",
            "text": "The grass is green.",
            "citations": [{"type": "char_location", "document_index": 0}]
        }));
        let ContentBlock::Text(text) = block else {
            panic!("not a text block");
        };
        assert!(text.unknown_fields.contains_key("citations"));
    }

    /// The string and block forms of `content` and `system` are different
    /// bytes, and both come back as themselves.
    #[test]
    fn the_string_and_block_forms_both_stay_as_they_arrived() {
        round_trip::<CreateMessageRequest>(json!({
            "model": "claude-opus-5",
            "max_tokens": 1024,
            "system": "Be brief.",
            "messages": [{"role": "user", "content": "Hello"}]
        }));
        round_trip::<CreateMessageRequest>(json!({
            "model": "claude-opus-5",
            "max_tokens": 1024,
            "system": [{"type": "text", "text": "Be brief."}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        }));
    }

    /// A custom tool and a server tool are told apart by `input_schema`, which
    /// only the first has — there is no `type` on a custom tool to key on.
    #[test]
    fn a_custom_tool_and_a_server_tool_take_different_arms() {
        assert!(matches!(
            round_trip::<Tool>(json!({
                "name": "get_weather",
                "description": "Get the current weather for a given location.",
                "input_schema": {"type": "object", "properties": {}}
            })),
            Tool::Custom(_)
        ));
        assert!(matches!(
            round_trip::<Tool>(json!({"type": "web_search_20260209", "name": "web_search"})),
            Tool::Other(_)
        ));
    }

    /// Every tool choice Anthropic documents, including the `any` that means
    /// what chat completions calls `required`.
    #[test]
    fn every_tool_choice_round_trips() {
        assert!(matches!(
            round_trip::<ToolChoice>(json!({"type": "auto", "disable_parallel_tool_use": true})),
            ToolChoice::Auto(_)
        ));
        assert!(matches!(
            round_trip::<ToolChoice>(json!({"type": "any"})),
            ToolChoice::Any(_)
        ));
        assert!(matches!(
            round_trip::<ToolChoice>(json!({"type": "none"})),
            ToolChoice::None(_)
        ));
        assert!(matches!(
            round_trip::<ToolChoice>(json!({"type": "tool", "name": "get_weather"})),
            ToolChoice::Tool(_)
        ));
    }

    /// Both thinking vocabularies parse. They are mutually exclusive per
    /// model, not per protocol version, so neither can be dropped.
    #[test]
    fn both_thinking_vocabularies_round_trip() {
        assert!(matches!(
            round_trip::<ThinkingConfig>(json!({"type": "enabled", "budget_tokens": 4096})),
            ThinkingConfig::Enabled(_)
        ));
        assert!(matches!(
            round_trip::<ThinkingConfig>(json!({"type": "adaptive", "display": "summarized"})),
            ThinkingConfig::Adaptive(_)
        ));
        assert!(matches!(
            round_trip::<ThinkingConfig>(json!({"type": "disabled"})),
            ThinkingConfig::Disabled(_)
        ));
    }

    /// `stop_reason` is required and nullable: a `null` must come back as a
    /// `null`, not vanish. `message_start` sends exactly that, and a reader
    /// that saw the key disappear would be unable to tell a reply still being
    /// generated from one that stopped for an unstated reason.
    #[test]
    fn a_null_stop_reason_stays_a_null() {
        let message = round_trip::<Message>(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-opus-5",
            "stop_reason": null,
            "stop_sequence": null,
            "usage": {"input_tokens": 25, "output_tokens": 1}
        }));
        assert!(message.stop_reason.is_none());
        // ...and the cache counts, absent here, must not come back as nulls.
        assert_eq!(message.usage.cache_read_input_tokens, None);
    }

    /// The cache counts are optional AND nullable, so all three states are
    /// distinct on the wire and stay that way.
    #[test]
    fn the_cache_counts_keep_absent_and_null_apart() {
        let usage = round_trip::<Usage>(json!({
            "input_tokens": 25,
            "output_tokens": 1,
            "cache_read_input_tokens": null
        }));
        assert_eq!(
            usage.cache_read_input_tokens,
            Some(None),
            "an explicit null"
        );
        assert_eq!(usage.cache_creation_input_tokens, None, "an absent key");
        round_trip::<Usage>(json!({
            "input_tokens": 25,
            "output_tokens": 1,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 0
        }));
    }

    /// The documented transcript's events, each as its own arm.
    #[test]
    fn every_documented_stream_event_round_trips() {
        let events = [
            json!({"type": "message_start", "message": {
                "id": "msg_1", "type": "message", "role": "assistant", "content": [],
                "model": "claude-opus-5", "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": 25, "output_tokens": 1}
            }}),
            json!({"type": "content_block_start", "index": 0,
                   "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "text_delta", "text": "Hello"}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "message_delta",
                   "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                   "usage": {"output_tokens": 15}}),
            json!({"type": "message_stop"}),
            json!({"type": "ping"}),
            json!({"type": "error",
                   "error": {"type": "overloaded_error", "message": "Overloaded"}}),
        ];
        for event in events {
            let parsed = round_trip::<MessageStreamEvent>(event.clone());
            assert!(
                !matches!(parsed, MessageStreamEvent::Unknown(_)),
                "{event} fell through to Unknown"
            );
        }
    }

    /// An event type Anthropic adds tomorrow must not fail the stream.
    #[test]
    fn an_unmodelled_stream_event_survives_as_unknown() {
        assert!(matches!(
            round_trip::<MessageStreamEvent>(json!({"type": "message_pause", "reason": "quota"})),
            MessageStreamEvent::Unknown(_)
        ));
    }

    /// Every delta the streaming docs name, plus one they do not.
    #[test]
    fn every_content_block_delta_round_trips() {
        assert!(matches!(
            round_trip::<ContentBlockDelta>(json!({"type": "text_delta", "text": "ello frien"})),
            ContentBlockDelta::TextDelta(_)
        ));
        assert!(matches!(
            round_trip::<ContentBlockDelta>(
                json!({"type": "input_json_delta", "partial_json": "{\"location\": \"San Fra"})
            ),
            ContentBlockDelta::InputJsonDelta(_)
        ));
        assert!(matches!(
            round_trip::<ContentBlockDelta>(json!({"type": "thinking_delta", "thinking": "1071"})),
            ContentBlockDelta::ThinkingDelta(_)
        ));
        assert!(matches!(
            round_trip::<ContentBlockDelta>(json!({"type": "signature_delta", "signature": "Eq"})),
            ContentBlockDelta::SignatureDelta(_)
        ));
        assert!(matches!(
            round_trip::<ContentBlockDelta>(json!({
                "type": "citations_delta",
                "citation": {"type": "char_location", "document_index": 0}
            })),
            ContentBlockDelta::Unknown(_)
        ));
    }

    /// The block fields Anthropic types as `T | null` keep an explicit null.
    ///
    /// `cache_control` and a document's `title` are `?:` AND `| null` in
    /// Anthropic's own schema, so all three states are reachable and a plain
    /// `Option` would forward two of them as one. The fields NOT listed here —
    /// `is_error`, a tool result's `content`, a tool's `description` and
    /// `strict`, `disable_parallel_tool_use`, a cache control's `ttl` — are
    /// optional but NOT nullable upstream, and stay plain `Option`s on purpose.
    #[test]
    fn a_nullable_block_field_keeps_its_explicit_null() {
        let block = round_trip::<ContentBlock>(json!({
            "type": "document",
            "source": {"type": "file", "file_id": "file_1"},
            "title": null,
            "cache_control": null
        }));
        let ContentBlock::Document(document) = block else {
            panic!("not a document block");
        };
        assert_eq!(document.title, Some(None), "an explicit null");
        assert!(
            matches!(document.cache_control, Some(None)),
            "an explicit null"
        );

        // ...and absent stays absent, which is the state a plain `Option`
        // would have collapsed the null into.
        let block = round_trip::<ContentBlock>(json!({
            "type": "document",
            "source": {"type": "file", "file_id": "file_1"}
        }));
        let ContentBlock::Document(document) = block else {
            panic!("not a document block");
        };
        assert_eq!(document.title, None);
        assert!(document.cache_control.is_none());
    }

    /// A request warpllm never sets a field on still carries it: the untyped
    /// parameters ride the catch-all rather than being dropped.
    #[test]
    fn unmodelled_request_parameters_reach_the_provider() {
        let request = round_trip::<CreateMessageRequest>(json!({
            "model": "claude-opus-5",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "top_k": 40,
            "metadata": {"user_id": "u1"},
            "service_tier": "auto"
        }));
        assert_eq!(request.unknown_fields["top_k"], json!(40));
    }
}
