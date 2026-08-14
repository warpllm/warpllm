//! Protocols: the wire shapes each protocol defines, and the transport that
//! puts them on the wire. A protocol owns its request/response types, its HTTP
//! binding (URL, auth scheme), and its error envelope.
//!
//! The conversions between these shapes and the gateway forms do NOT live
//! here — they live in `crate::gateway`, one child module per protocol,
//! because a conversion is a statement about the canonical model as much as
//! about the wire. That makes this module a leaf: it depends on nothing but
//! `error` and `http`, and `gateway` depends on it.
//!
//! The module path spells out [`crate::types::Api`]'s own name, not the HTTP
//! path: `Api::OpenAiCompatChatCompletions` deserializes from
//! `"openai_compat_chat_completions"` and is implemented at
//! `openai_compat::chat_completions`, protocol segment and all. The URL an API
//! is reached at is a transport detail and lives in that module's
//! `transport.rs` — two protocols may serve the same API at different paths.
//!
//! [`crate::types::Api`] itself is declared in `crate::types`, above this
//! module and `crate::gateway` both, since both layers speak it and neither
//! owns it. What lives here are the protocols its names refer to.
//!
//! Provider-specific logic does NOT live here either: per-model specs are
//! contributed by `crate::registry`, and per-provider conversion deltas belong
//! with the conversions under `crate::gateway`.

use serde::Deserialize;

pub mod anthropic;
pub mod openai_compat;

/// Catch-all for fields a protocol introduces that this crate does not model
/// yet.
///
/// Every request and response struct carries a `#[serde(flatten)]`
/// `unknown_fields` of this type, so a field added upstream still reaches
/// clients (and an unmodeled request parameter still reaches the provider)
/// instead of being silently dropped — clients can adopt new API fields
/// before this crate ships explicit support for them.
pub type UnknownFields = serde_json::Map<String, serde_json::Value>;

// Optionality is spelled three ways across these modules, one per state the
// wire can be in, and which one a field gets follows the upstream spec rather
// than taste:
// - required: a plain field.
// - required but nullable, like a message's `content` or Anthropic's
//   `stop_reason`: an `Option` that always serializes, `null` when there is
//   none.
// - optional AND nullable, like OpenAI's `usage` and `logprobs` or Anthropic's
//   `cache_read_input_tokens`: `Option<Option<_>>`, read through
//   [`present_or_null`]. Absent and `null` are different bytes, and a provider
//   sends both — OpenAI streams `"logprobs": null` on every chunk and DeepSeek
//   omits the key outright — so warpllm holds the difference rather than
//   normalizing it. Absent stays absent; `null` stays `null`.
//
// That third kind is what makes these shapes a superset of the vendor's rather
// than merely a permissive reader of them: everything the vendor can put on
// the wire, warpllm can hold AND hand back unchanged. Narrowing one to a plain
// `Option` would still ACCEPT a provider's null and would still be lossless in
// meaning, but the null would come back out as an absent key and the generated
// TypeScript would stop admitting a value the vendor's own types allow.

/// Deserializes an optional, nullable field: `None` when the key was absent,
/// `Some(None)` when it was present and `null`.
///
/// Serde's own `Option` collapses the two — a nested `Option<Option<T>>` reads
/// an explicit `null` as `None`, the same value a missing key produces — so
/// the outer layer only means "the key was there" with this in front of it.
pub(crate) fn present_or_null<'de, T, D>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    T: Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    Option::deserialize(deserializer).map(Some)
}
