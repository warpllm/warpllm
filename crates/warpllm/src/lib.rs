//! Core engine for warpllm, a warp-speed, robust AI gateway.

mod auth;
mod client;
mod config;
mod credentials;
mod error;
mod gateway;
mod http;
mod json_client;
mod registry;

pub mod protocol;
pub mod types;

pub use client::{ChatCompletionStream, Client};
pub use config::{ClientConfig, ProviderConfig};
pub use error::{Error, Origin, Result};
/// The gateway's canonical form for an upstream failure — the error-side
/// counterpart to the request and response forms, and the payload every
/// provider-driven [`Error`] variant carries. `gateway` itself stays private,
/// exactly as `registry` does; this is its only public shape.
pub use gateway::types::ProviderError;
pub use json_client::{JsonChatStream, JsonClient};
/// Everything a caller must NAME to make a call and hold its result: the
/// request, everything it is built from, and the response handed back by
/// [`Client::chat_completions`].
///
/// That is the whole rule, and it is why the response's insides are absent.
/// Reading `completion.choices[0].message.content` names none of them — field
/// access needs no import — so `Choice` and `CompletionUsage` stay at
/// `protocol::openai_compat::chat_completions::types`, where the path states
/// what they are: shapes of the OpenAI protocol, which warpllm speaks, rather
/// than warpllm's own vocabulary. Requests are the asymmetric half — you
/// cannot build one without spelling every type it reaches.
///
/// Which is why the list is long. Setting `tools`, `tool_choice`,
/// `response_format` or a multimodal `content` means naming the type of each,
/// and the type each of those reaches in turn; a caller who has to write
/// `protocol::openai_compat::chat_completions::types::ImageUrl` to attach an
/// image is being sent through a path that reads like an internal one for a
/// thing the request cannot be built without.
///
/// The tool-call forms are here for the same reason and only that reason. They
/// are still not needed to READ a reply — the rule above holds — but an
/// assistant turn replayed back into a conversation has to be spelled out, and
/// that is a request.
///
/// A glob would have claimed all of it as warpllm's, and widened the public
/// API every time that module gained a type.
pub use protocol::openai_compat::chat_completions::types::{
    ChatCompletionMessageCustomToolCall, ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallUnion, ChatCompletionNamedToolChoice,
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContent,
    ChatCompletionRequestMessageContentPart, ChatCompletionRequestMessageContentPartAudio,
    ChatCompletionRequestMessageContentPartFile, ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartRefusal, ChatCompletionRequestMessageContentPartText,
    ChatCompletionResponseFormat, ChatCompletionStop, ChatCompletionStreamOptions,
    ChatCompletionTool, ChatCompletionToolChoiceOption, CreateChatCompletionRequest,
    CreateChatCompletionResponse, CreateChatCompletionStreamResponse, Custom, FileContent,
    Function, FunctionObject, ImageUrl, InputAudio, JsonSchemaDefinition, ResponseFormatJsonSchema,
    ResponseFormatSimple, ToolChoiceFunction,
};
/// A failure rendered the way an OpenAI-compatible surface reports it, and
/// the only error shape warpllm shows anyone who is not writing Rust.
///
/// [`Error`] stays internal — it knows a quota exhaustion from a rate limit,
/// which is what lets [`Error::to_openai`] emit OpenAI's spelling for either
/// no matter which provider served the request.
pub use protocol::openai_compat::error::{ErrorBody, ErrorClass, OpenAiError};
/// The registry's public face: a provider, a model, and the lookup that hands
/// back one of each. `registry` itself stays private — the roster, the schema,
/// and the loading are not API.
///
/// Read-only by construction: every field is private and there is no public
/// constructor, so [`fetch_model`] is the way to obtain either half.
pub use registry::{Capabilities, ModelSpec, ProviderSpec, SupportedApi, fetch_model};
pub use types::Api;

/// Returns the warpllm version.
///
/// ```
/// let version = warpllm::version();
/// assert_eq!(version.split('.').count(), 3);
/// ```
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_matches_manifest() {
        assert_eq!(version(), env!("CARGO_PKG_VERSION"));
    }
}
