//! Provider implementations: how each upstream is called (HTTP, auth, error
//! mapping, per-provider quirks). The module path mirrors the HTTP path:
//! `POST /chat/completions` lives at `openai::chat::completions`.
//!
//! Adding a provider: create `<provider>/<path>/mod.rs`. An
//! OpenAI-compatible provider reuses `crate::types::openai` wholesale and
//! owns only its wire quirks here; a provider with its own wire format also
//! adds a shape family under `crate::types` and translates in its impl.

pub(crate) mod openai;
