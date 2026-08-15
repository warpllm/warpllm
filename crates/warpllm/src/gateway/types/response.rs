//! The gateway chat response.

use serde::{Deserialize, Serialize};

use super::{IngestSource, Message, ProviderExt};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ChatResponse {
    pub id: String,
    /// Upstream model name as returned; the client overwrites the wire
    /// echo with the caller's prefixed string after rendering.
    pub model: String,
    /// Not every provider reports a creation timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<u64>,
    pub completions: Vec<Completion>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(default, skip_serializing_if = "ProviderExt::is_empty")]
    pub ext: ProviderExt,
    #[serde(skip)]
    #[allow(dead_code)] // staged: read once the passthrough fast path lands
    pub source: Option<IngestSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Completion {
    pub message: Message,
    /// Derived from the raw string; for programmatic matching only.
    pub finish_reason: FinishReason,
    /// Authoritative on render — losslessly echoes the provider's value.
    pub finish_reason_raw: String,
    #[serde(default, skip_serializing_if = "ProviderExt::is_empty")]
    pub ext: ProviderExt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Other,
}

impl FinishReason {
    pub(crate) fn from_raw(raw: &str) -> Self {
        match raw {
            "stop" => Self::Stop,
            "length" => Self::Length,
            "tool_calls" => Self::ToolCalls,
            "content_filter" => Self::ContentFilter,
            _ => Self::Other,
        }
    }
}

/// Token accounting in the widest cross-provider units; fields a provider
/// doesn't report stay `None`. Protocol-specific residue (detail objects,
/// unknown fields) rides `ext` so rendering back is lossless.
///
/// # The cached counts are a BREAKDOWN, never addends
///
/// [`input_tokens`](Self::input_tokens) is the WHOLE input, cached tokens
/// included, and [`cache_read_tokens`](Self::cache_read_tokens) and
/// [`cache_write_tokens`](Self::cache_write_tokens) say how much of it was
/// which. [`total_tokens`](Self::total_tokens) is then input + output, and
/// adding the cache counts on top double-counts them.
///
/// Stated here rather than left to each protocol because the two wire formats
/// disagree and the difference is invisible in the field name: OpenAI's
/// `prompt_tokens` already includes its `cached_tokens`, while Anthropic's
/// `input_tokens` counts only what came AFTER the last cache breakpoint. A
/// protocol whose wire value is exclusive has to add the cache counts back on
/// ingest — otherwise the same conversation reports a different `prompt_tokens`
/// depending on which backend served it, and `cached_tokens` can exceed the
/// prompt it is supposedly part of.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct Usage {
    /// The whole input, INCLUDING whatever was read from or written to cache.
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    /// `input_tokens + output_tokens`. Not a third independent number.
    pub total_tokens: Option<u64>,
    /// Part of `output_tokens`, not additional to it.
    pub reasoning_tokens: Option<u64>,
    /// Part of `input_tokens`, not additional to it.
    pub cache_read_tokens: Option<u64>,
    /// Part of `input_tokens`, not additional to it.
    pub cache_write_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "ProviderExt::is_empty")]
    pub ext: ProviderExt,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finish_reason_from_raw_table() {
        assert_eq!(FinishReason::from_raw("stop"), FinishReason::Stop);
        assert_eq!(FinishReason::from_raw("length"), FinishReason::Length);
        assert_eq!(
            FinishReason::from_raw("tool_calls"),
            FinishReason::ToolCalls
        );
        assert_eq!(
            FinishReason::from_raw("content_filter"),
            FinishReason::ContentFilter
        );
        assert_eq!(FinishReason::from_raw("function_call"), FinishReason::Other);
        assert_eq!(FinishReason::from_raw("junk"), FinishReason::Other);
    }

    #[test]
    fn usage_round_trips() {
        let mut usage = Usage {
            input_tokens: Some(9),
            output_tokens: Some(12),
            total_tokens: Some(21),
            reasoning_tokens: Some(5),
            cache_read_tokens: Some(3),
            cache_write_tokens: Some(2),
            ..Default::default()
        };
        usage
            .ext
            .insert("openai_compat".into(), serde_json::json!({"x": 1}));
        let value = serde_json::to_value(&usage).unwrap();
        let back: Usage = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(&back).unwrap(), value);
    }
}
