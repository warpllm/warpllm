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
//! transports and auth. When they arrive they reuse [`messages::types`] whole
//! and contribute a transport apiece — which is the reason those two live at
//! different layers here.
//!
//! SHAPES AND TRANSPORT ONLY. Turning an Anthropic error body into warpllm's
//! canonical failure form is an ingest conversion and lives with every other
//! one at `crate::gateway::anthropic::error` — the same rule that keeps
//! `transport` handing a non-2xx back as data rather than deciding what it
//! means.

pub mod error;
pub mod messages;
