//! The gateway chat forms: warpllm's canonical request/response, the
//! internal shape every protocol converts to and from. Closest in spirit to the
//! OpenAI Responses API's item/block model, which subsumes chat completions.
//!
//! Contract: converting INTO these types is lossless (unmodeled fields ride
//! namespaced `ext` bags); converting OUT to a specific protocol or
//! provider may be lossy. Parameters the target does not document are NOT
//! filtered here — they render onto the wire and the provider, the only
//! authority on what it supports, rejects them itself.
//!
//! The serde-derived canonical JSON form is UNSTABLE and internal-only —
//! it exists for tests, logging, and future caching, not as a wire contract.
//!
//! This module also PERFORMS those conversions, which is what makes it the
//! gateway rather than a schema: [`types`] holds the canonical shapes, and one
//! sibling per protocol holds the translation to and from that protocol's wire
//! shapes — [`crate::protocol`] owns the shapes themselves and the transport,
//! and knows nothing about this layer.
//!
//! That one-way dependency is the point. Conversion needs both sides, so it
//! lives here, where naming [`crate::protocol`] is normal; putting it beside
//! the wire shapes instead would make `protocol` depend on the gateway and
//! leave a per-provider adapter nowhere clean to sit.

// Nothing routes to Anthropic yet: `client.rs` picks an exchange by the routed
// model's surface, and no surface names this protocol until the registry and
// the dispatch land. Being unreachable is what makes these conversions safe to
// land ahead of that — but it also means every item here reads as dead. Drop
// this attribute on the change that wires the client; it should stop being
// needed there.
#[allow(dead_code)]
pub(crate) mod anthropic;
pub(crate) mod error;
pub(crate) mod openai_compat;
pub(crate) mod types;
