//! The gateway chat request.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{CacheHint, IngestSource, Message, ProviderExt};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct ChatRequest {
    /// Upstream model name, provider prefix already stripped.
    pub model: String,

    /// Full conversation, system messages included (hoisted per target).
    pub messages: Vec<Message>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Whether the model may open more than one tool call in a turn.
    ///
    /// Typed on neither wire — chat completions spells it `parallel_tool_calls`
    /// among its unknown fields, Anthropic spells it inverted as
    /// `disable_parallel_tool_use` hanging off `tool_choice` — and promoted
    /// anyway, for the reason [`GenerationParams`] gives: `ext` is
    /// same-protocol only, so a value left there is a value the OTHER
    /// protocol's renderer is forbidden to see. Left there, a caller who
    /// forbade parallel calls got them anyway.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    #[serde(default)]
    pub params: GenerationParams,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,

    /// Request-level cache hint the renderer applies to the LAST cacheable
    /// block, so callers get caching without hand-placing breakpoints.
    /// Block-level hints take precedence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheHint>,

    #[serde(default)]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_include_usage: Option<bool>,

    #[serde(default, skip_serializing_if = "ProviderExt::is_empty")]
    pub ext: ProviderExt,

    /// Where this request came from; see [`IngestSource`].
    ///
    /// Read by the Anthropic renderer to tell a caller who arrived on that
    /// protocol — and therefore still holds its retained residue — from one
    /// who was translated onto it and holds nothing. The passthrough fast path
    /// is the other reader this was staged for.
    #[serde(skip)]
    pub source: Option<IngestSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ToolDef {
    /// Verbatim; length/charset limits differ per provider.
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema, verbatim. Providers that reject some keywords prune in
    /// their renderer, not here.
    pub input_schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheHint>,
    /// Provider built-in tools (web search, computer use) ride here rather
    /// than masquerading as function tools.
    #[serde(default, skip_serializing_if = "ProviderExt::is_empty")]
    pub ext: ProviderExt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub(crate) enum ToolChoice {
    Auto,
    None,
    Required,
    Tool { name: String },
}

/// Store what the caller sent, unconverted. A param lives here when a renderer
/// for ANOTHER protocol has to read it; everything else (seed, penalties, ...)
/// rides the namespaced ext bags untouched until that becomes true of it too.
///
/// A typed wire field is the usual reason, but not the test — `top_k` is typed
/// on neither wire and still belongs here, because `ext` is same-protocol only
/// and [`Protocol::may_read`](crate::types::Protocol::may_read) forbids the
/// other protocol's renderer from reaching into this one's bag. That is the
/// same rule that promotes `max_completion_tokens` at ingest.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct GenerationParams {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    /// Anthropic takes this natively; chat completions carries it as a
    /// provider extension. Promoted so the routed protocol can honor it.
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
}

/// Unified over provider thinking budgets / reasoning-effort spellings,
/// carrying BOTH so cross-protocol conversion is an explicit render action.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct ReasoningConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// "minimal" | "low" | "medium" | "high" | "xhigh".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
    /// Reason but don't return the reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::{ContentBlock, Role};
    use super::*;
    use crate::types::Protocol;

    #[test]
    fn chat_request_round_trips_and_skips_source() {
        let mut request = ChatRequest {
            model: "gpt-5.6".into(),
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "hi".into(),
                    cache: None,
                }],
                ext: ProviderExt::new(),
            }],
            params: GenerationParams {
                temperature: Some(0.7),
                stop: vec!["END".into()],
                ..Default::default()
            },
            stream: false,
            source: Some(IngestSource {
                protocol: Protocol::OpenAiCompat,
                body: json!({"model": "openai/gpt-5.6"}),
            }),
            ..Default::default()
        };
        request
            .ext
            .insert("openai_compat".into(), json!({"top_k": 40}));

        let value = serde_json::to_value(&request).unwrap();
        assert!(value.get("source").is_none(), "source must not serialize");
        assert_eq!(value["params"]["temperature"], 0.7);

        let back: ChatRequest = serde_json::from_value(value.clone()).unwrap();
        assert!(back.source.is_none());
        assert_eq!(serde_json::to_value(&back).unwrap(), value);
    }
}
