//! The Google protocol: `generateContent`, as spoken by Gemini models on AI
//! Studio and on Vertex AI.
//!
//! Sibling to [`anthropic`](super::anthropic) and
//! [`openai_compat`](super::openai_compat). The shapes track
//! <https://ai.google.dev/api/generate-content>. AI Studio and Vertex differ in
//! URL and credential, not in body, so they will share [`generate_content`]
//! and contribute a transport apiece, the way Anthropic's two hosts do.
//!
//! SHAPES AND TRANSPORT ONLY. Conversions to and from the gateway forms live
//! with every other one under `crate::gateway`.

pub mod generate_content;
