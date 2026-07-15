//! Wire shapes, grouped by shape family then endpoint — pure serde data,
//! zero I/O. The module path mirrors the HTTP path: the `chat.completion`
//! types live at `openai::chat::completions`.
//!
//! OpenAI-compatible providers reuse `openai::*` wholesale; a provider with
//! its own wire format gets a sibling family here (e.g.
//! `anthropic::messages`) and its impl under `crate::providers`
//! translates to the public OpenAI shape.

pub mod openai;
