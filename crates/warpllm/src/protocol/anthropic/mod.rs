//! The Anthropic protocol: `api.anthropic.com`'s own wire format, as spoken
//! by every model whose roster entry lists an `anthropic_*` surface.
//!
//! Sibling to [`openai_compat`](super::openai_compat), and deliberately not a
//! variation on it. Where that module is a permissive superset many providers
//! implement independently, this one has a single implementer, so its shapes
//! track <https://platform.claude.com/docs/en/api/messages> directly and the
//! `type` discriminators on its unions can be trusted rather than worked
//! around.
//!
//! Bedrock (#24) and Vertex (#25) also speak Messages, over different
//! transports. When they arrive they reuse [`messages::types`] whole and
//! contribute a transport apiece — which is the reason those two live at
//! different layers here.
//!
//! A transport apiece and NOT an auth apiece: what those two also differ in is
//! the credential, and that is a fact about the provider rather than about this
//! wire format, so it lives at [`crate::auth`] and a transport is handed one.
//! Vertex is what makes the distinction load-bearing — one token there serves
//! both this protocol and Google's own — so a credential filed under either
//! would have to be built twice.
//!
//! SHAPES AND TRANSPORT ONLY. Turning an Anthropic error body into warpllm's
//! canonical failure form is an ingest conversion and lives with every other
//! one at `crate::gateway::anthropic::error` — the same rule that keeps
//! `transport` handing a non-2xx back as data rather than deciding what it
//! means.

pub mod error;
pub mod messages;
