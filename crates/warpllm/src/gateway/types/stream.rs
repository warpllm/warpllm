//! The gateway chat response, streamed: one value per chunk.
//!
//! [`ChatResponseChunk`] mirrors [`ChatResponse`](super::ChatResponse) field for
//! field, with delta semantics — same names, same `ext` discipline, same
//! vocabulary ([`Role`], [`FinishReason`], [`Usage`]). Reading the two side by
//! side is how you see what a chunk is: the completed response with every
//! guarantee relaxed to "some of this may arrive later, or never".
//!
//! # Why the completed form could not be reused
//!
//! Four of its invariants are exactly what a chunk breaks:
//!
//! - `Completion::finish_reason_raw` is a required `String`; a chunk sends
//!   `null` on every chunk of a choice but the last.
//! - `Message::role` is required; a delta carries it once, on the chunk that
//!   opens the choice.
//! - `ContentBlock::Text` cannot say it holds a *fragment*, and cannot tell
//!   absent from `null` from `""` — three states one live OpenAI stream sends.
//! - `ContentBlock::ToolCall` requires `id`, `name`, and complete `arguments`,
//!   while a streamed call is correlated by `index` and its arguments arrive
//!   split mid-JSON. `index` has nowhere to live, and a fragment like
//!   `{"city":` is not the "exact JSON text" [`RawJson`](super::RawJson)
//!   promises to hold.
//!
//! Relaxing those on the completed form would make every whole-response
//! consumer unwrap `Option`s that are never `None` on its path, and would still
//! leave `index` homeless. So this is a second shape — built from the first
//! one's vocabulary, not beside it.
//!
//! # One value per chunk
//!
//! The typed union is at the *content* level ([`ContentDelta`]), which is
//! richer than the wire's one sparse delta object: reasoning, text, and
//! tool-call fragments are distinct variants rather than sibling keys. The
//! envelope stays one-for-one with a wire chunk, because a chunk carries fields
//! that belong to no single semantic event — `system_fingerprint`,
//! `service_tier`, a repeated `logprobs: null`, and OpenAI's per-chunk
//! `obfuscation`, which changes on every one. Exploding a chunk into several
//! events would strand that residue and make re-chunking a guess.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{FinishReason, IngestSource, ProviderExt, Role, Usage};

/// One chunk of a streaming chat response, normalized.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ChatResponseChunk {
    /// Identifies the completion, not the chunk: every chunk of one stream
    /// carries the same value.
    pub id: String,
    /// Upstream model name as returned; the client overwrites the wire echo
    /// with the caller's prefixed string after rendering, on every chunk.
    pub model: String,
    /// Not every provider reports a creation timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<u64>,
    /// Empty on the usage-only chunk `stream_options.include_usage` adds, and
    /// on any keepalive a provider sends with no choices.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub completions: Vec<CompletionDelta>,
    /// Totals for the whole request, not this chunk. Present only when the
    /// caller asked for them, and only on the chunk that carries them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(default, skip_serializing_if = "ProviderExt::is_empty")]
    pub ext: ProviderExt,
    #[serde(skip)]
    #[allow(dead_code)] // staged: read once the passthrough fast path lands
    pub source: Option<IngestSource>,
}

/// The streaming counterpart of [`Completion`](super::Completion).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CompletionDelta {
    pub delta: MessageDelta,
    /// Derived from the raw string; for programmatic matching only. `None`
    /// until the choice ends — which is every chunk of it but the last.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    /// Authoritative on render — losslessly echoes the provider's value.
    /// `None` and `Some` track [`Self::finish_reason`] exactly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason_raw: Option<String>,
    /// Carries the wire choice's `index` and `logprobs`.
    #[serde(default, skip_serializing_if = "ProviderExt::is_empty")]
    pub ext: ProviderExt,
}

/// The streaming counterpart of [`Message`](super::Message): every field is a
/// fragment, so every field is optional — a chunk carrying only `role`, only a
/// slice of text, or nothing at all is ordinary.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct MessageDelta {
    /// Sent once, on the chunk that opens the choice.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content: Vec<ContentDelta>,
    /// Carries the raw role spelling, refusal fragments, and the deprecated
    /// `function_call`.
    #[serde(default, skip_serializing_if = "ProviderExt::is_empty")]
    pub ext: ProviderExt,
}

/// A fragment of one piece of a message.
///
/// Deliberately narrower than [`ContentBlock`](super::ContentBlock): media,
/// tool results, and cache hints cannot arrive in fragments, so they have no
/// variant here. What a stream can split, this can hold.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub(crate) enum ContentDelta {
    /// A slice of the answer. An empty string is a real, distinct value — an
    /// OpenAI stream opens with `"content": ""` — and is not the same as no
    /// delta at all.
    Text { text: String },

    /// A slice of chain-of-thought.
    ///
    /// No [`ReasoningDetail`](super::ReasoningDetail), unlike the completed
    /// form: a signature covers a whole reasoning block and cannot be carried
    /// by the fragments it is computed over, so a delta that offered the field
    /// could only ever leave it `None`.
    Reasoning {
        text: String,
        /// Native protocol this fragment originated from; see
        /// [`ContentBlock::Reasoning`](super::ContentBlock).
        #[serde(skip_serializing_if = "Option::is_none")]
        provenance: Option<String>,
    },

    /// A slice of one tool call. Every part but the index is optional, because
    /// every part of a streamed call arrives when it arrives.
    ToolCall {
        /// The correlation key, and the reason this variant exists rather than
        /// reusing the completed block: `id` and `name` arrive once, on
        /// whichever fragment opens the call, and every fragment after it
        /// carries only this index and more argument text.
        index: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// What kind of call this is — a function, or whatever else a protocol
        /// distinguishes. Carried rather than assumed: it arrives on the
        /// fragment that opens a call and is absent on the rest, so a form
        /// that restored a constant would put a key on the wire the provider
        /// never sent.
        #[serde(skip_serializing_if = "Option::is_none")]
        kind: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// A FRAGMENT of the JSON-encoded arguments, so a `String` and not a
        /// [`RawJson`](super::RawJson): concatenating one call's fragments
        /// yields the whole, which is model-generated and so may still be
        /// invalid JSON. `None` is a fragment that carried no argument text at
        /// all, which is not the same as one that carried `""`.
        #[serde(skip_serializing_if = "Option::is_none")]
        arguments: Option<String>,
    },

    /// Forward-compat, matching the completed form: pass through when
    /// source == target, else policy.
    #[serde(untagged)]
    Unknown(Value),
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::types::Protocol;

    fn round_trip(delta: &ContentDelta) -> Value {
        let value = serde_json::to_value(delta).unwrap();
        let back: ContentDelta = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(&back).unwrap(), value);
        value
    }

    #[test]
    fn every_delta_variant_round_trips() {
        let deltas = [
            ContentDelta::Text {
                text: "Hello".into(),
            },
            ContentDelta::Reasoning {
                text: "step by step".into(),
                provenance: Some("openai_compat".into()),
            },
            ContentDelta::ToolCall {
                index: 0,
                id: Some("call-1".into()),
                kind: Some("function".into()),
                name: Some("search".into()),
                arguments: Some(String::new()),
            },
            ContentDelta::ToolCall {
                index: 0,
                id: None,
                kind: None,
                name: None,
                arguments: Some(r#"{"city":"#.into()),
            },
        ];
        for delta in &deltas {
            round_trip(delta);
        }
    }

    /// The empty string is a value, not an absence: an OpenAI stream opens
    /// with `"content": ""`, and a form that collapsed it into "no delta"
    /// would lose the chunk that told a client the answer had started.
    #[test]
    fn an_empty_text_delta_is_not_an_absent_one() {
        let value = round_trip(&ContentDelta::Text {
            text: String::new(),
        });
        assert_eq!(value, json!({"type": "text", "text": ""}));

        let empty = MessageDelta::default();
        assert!(empty.content.is_empty());
        assert_eq!(serde_json::to_value(&empty).unwrap(), json!({}));
    }

    /// A fragment that opens a call names it; the ones after carry only the
    /// index and more argument text. Both must survive as themselves.
    #[test]
    fn a_tool_call_fragment_keeps_its_index_and_omits_what_it_lacks() {
        let opener = round_trip(&ContentDelta::ToolCall {
            index: 2,
            id: Some("call_0_5f3a91c2".into()),
            kind: Some("function".into()),
            name: Some("get_weather".into()),
            arguments: Some(String::new()),
        });
        assert_eq!(opener["index"], 2);
        assert_eq!(opener["id"], "call_0_5f3a91c2");
        assert_eq!(opener["arguments"], "");

        let continuation = round_trip(&ContentDelta::ToolCall {
            index: 2,
            id: None,
            kind: None,
            name: None,
            arguments: Some(r#""Seoul"}"#.into()),
        });
        assert!(continuation.get("id").is_none(), "{continuation}");
        assert!(continuation.get("kind").is_none(), "{continuation}");
        assert!(continuation.get("name").is_none(), "{continuation}");
        assert_eq!(continuation["arguments"], r#""Seoul"}"#);
    }

    #[test]
    fn an_unrecognized_delta_becomes_unknown_verbatim() {
        let value = json!({"type": "hologram", "payload": {"z": 1, "a": 2}});
        let delta: ContentDelta = serde_json::from_value(value.clone()).unwrap();
        assert!(matches!(&delta, ContentDelta::Unknown(v) if *v == value));
        assert_eq!(serde_json::to_value(&delta).unwrap(), value);
    }

    #[test]
    fn chunk_round_trips_and_skips_source() {
        let mut chunk = ChatResponseChunk {
            id: "chatcmpl-123".into(),
            model: "gpt-5.6".into(),
            created: Some(1_700_000_000),
            completions: vec![CompletionDelta {
                delta: MessageDelta {
                    role: Some(Role::Assistant),
                    content: vec![ContentDelta::Text {
                        text: "Hello".into(),
                    }],
                    ext: ProviderExt::new(),
                },
                finish_reason: Some(FinishReason::Stop),
                finish_reason_raw: Some("stop".into()),
                ext: ProviderExt::new(),
            }],
            usage: None,
            ext: ProviderExt::new(),
            source: Some(IngestSource {
                protocol: Protocol::OpenAiCompat,
                body: json!({"id": "chatcmpl-123"}),
            }),
        };
        chunk
            .ext
            .insert("openai_compat".into(), json!({"obfuscation": "KtQ3nZ8w"}));

        let value = serde_json::to_value(&chunk).unwrap();
        assert!(value.get("source").is_none(), "source must not serialize");

        let back: ChatResponseChunk = serde_json::from_value(value.clone()).unwrap();
        assert!(back.source.is_none());
        assert_eq!(serde_json::to_value(&back).unwrap(), value);
    }

    /// The usage-only chunk: no choices at all, and the totals for the whole
    /// request. An empty `completions` must not serialize as `[]` noise.
    #[test]
    fn a_usage_only_chunk_carries_no_completions() {
        let chunk = ChatResponseChunk {
            id: "chatcmpl-123".into(),
            model: "gpt-5.6".into(),
            created: Some(1_700_000_000),
            completions: Vec::new(),
            usage: Some(Usage {
                input_tokens: Some(11),
                output_tokens: Some(3),
                total_tokens: Some(14),
                ..Default::default()
            }),
            ext: ProviderExt::new(),
            source: None,
        };
        let value = serde_json::to_value(&chunk).unwrap();
        assert!(value.get("completions").is_none(), "{value}");
        assert_eq!(value["usage"]["total_tokens"], 14);
    }

    /// `finish_reason` and its raw spelling are absent together on every chunk
    /// of a choice but the last, and present together on that one.
    #[test]
    fn an_unfinished_completion_omits_both_finish_fields() {
        let unfinished = CompletionDelta {
            delta: MessageDelta::default(),
            finish_reason: None,
            finish_reason_raw: None,
            ext: ProviderExt::new(),
        };
        assert_eq!(
            serde_json::to_value(&unfinished).unwrap(),
            json!({"delta": {}})
        );
    }
}
