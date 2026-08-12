//! Request conversions: OpenAI-compatible wire → gateway (ingest) and
//! gateway → wire (render).

use serde_json::Value;

use crate::error::{Error, Result};
use crate::gateway::types::{self, ContentBlock, IngestSource};
use crate::protocol::openai_compat::chat_completions::types::{
    ChatCompletionRequestMessage, CreateChatCompletionRequest,
};
use crate::types::Protocol;

use crate::gateway::openai_compat::{merged_ext, namespaced, role_from_wire, role_to_wire};

/// Permissive and infallible: capture, don't validate. `model` is the
/// prefix-stripped name from `parse_model`; `stream` is resolved before
/// ingest (`Some(true)` already rejected; false/`None` are equivalent).
///
/// The destructuring is deliberately exhaustive (no `..`): adding a typed
/// field to the wire request without mapping it here is a compile error,
/// so normalization can never silently drop data. Everything without a
/// typed wire field (seed, top_k, ...) rides `ext["openai_compat"]`
/// verbatim.
pub(crate) fn ingest_request(
    request: CreateChatCompletionRequest,
    model: &str,
) -> types::ChatRequest {
    // Wire structs are plain serde data; serialization cannot fail.
    let body = serde_json::to_value(&request).expect("wire request serializes");
    let CreateChatCompletionRequest {
        model: _,
        messages,
        temperature,
        max_tokens,
        top_p,
        stop,
        stream,
        unknown_fields,
    } = request;
    types::ChatRequest {
        model: model.to_string(),
        messages: messages.into_iter().map(ingest_message).collect(),
        params: types::GenerationParams {
            max_tokens,
            temperature,
            top_p,
            stop: stop.unwrap_or_default(),
        },
        stream: stream.unwrap_or(false),
        ext: namespaced(unknown_fields),
        source: Some(IngestSource {
            protocol: Protocol::OpenAiCompat,
            body,
        }),
        ..Default::default()
    }
}

fn ingest_message(message: ChatCompletionRequestMessage) -> types::Message {
    let ChatCompletionRequestMessage {
        role,
        content,
        unknown_fields,
    } = message;
    let (role, raw_role) = role_from_wire(role);
    let mut compat = unknown_fields;
    if let Some(raw) = raw_role {
        // Unrecognized roles (e.g. "developer") pass through verbatim:
        // render restores the raw spelling from ext.
        compat.insert("role".into(), Value::String(raw));
    }
    types::Message {
        role,
        content: vec![ContentBlock::Text {
            text: content,
            cache: None,
        }],
        ext: namespaced(compat),
    }
}

/// Renders the gateway request back onto the wire. Mechanical by
/// design: nothing is filtered against what the target documents — the
/// provider is the authority on its own parameters and rejects what it
/// doesn't accept. Same-protocol round trips are lossless modulo three
/// documented transformations: the provider prefix is stripped from
/// `model`, `stream: false` is dropped, and an empty `stop` list is omitted,
/// each of them exactly reversible or resolved before ingest.
///
/// `stream` is emitted only when it is true. The wire field is optional and
/// defaults to false, so an absent key and an explicit `false` ask the
/// provider for the same thing — and ingest already reads them as the same
/// thing, which is what keeps dropping it lossless.
pub(crate) fn render_request(
    request: &types::ChatRequest,
    provider: &'static str,
) -> Result<CreateChatCompletionRequest> {
    ensure_no_staged_features(request)?;
    let unknown_fields = merged_ext(&request.ext, provider);
    let params = &request.params;
    Ok(CreateChatCompletionRequest {
        model: request.model.clone(),
        messages: request
            .messages
            .iter()
            .map(|m| render_message(m, provider))
            .collect::<Result<_>>()?,
        temperature: params.temperature,
        max_tokens: params.max_tokens,
        top_p: params.top_p,
        stop: (!params.stop.is_empty()).then(|| params.stop.clone()),
        stream: request.stream.then_some(true),
        unknown_fields,
    })
}

/// The compat wire types don't model these yet (they arrive via
/// `ext["openai_compat"]` passthrough instead); a typed rendering is the
/// documented follow-up before the first translated protocol.
fn ensure_no_staged_features(request: &types::ChatRequest) -> Result<()> {
    if !request.tools.is_empty()
        || request.tool_choice.is_some()
        || request.response_format.is_some()
        || request.reasoning.is_some()
        || request.cache.is_some()
        || request.stream_include_usage.is_some()
    {
        return Err(Error::NotImplemented(
            "typed tools/response_format on the openai_compat renderer",
        ));
    }
    Ok(())
}

fn render_message(
    message: &types::Message,
    provider: &str,
) -> Result<ChatCompletionRequestMessage> {
    let mut unknown_fields = merged_ext(&message.ext, provider);
    let role = match unknown_fields.remove("role") {
        Some(Value::String(raw)) => raw,
        _ => role_to_wire(message.role).to_string(),
    };
    let mut texts = Vec::new();
    for block in &message.content {
        match block {
            ContentBlock::Text { text, .. } => texts.push(text.as_str()),
            // Only reachable cross-protocol; typed rendering is the
            // documented follow-up alongside the wire-type expansion.
            _ => {
                return Err(Error::NotImplemented(
                    "typed content blocks on the openai_compat renderer",
                ));
            }
        }
    }
    Ok(ChatCompletionRequestMessage {
        role,
        // Same-protocol messages carry exactly one text block, so the join
        // is exact; joining >1 only occurs cross-protocol.
        content: texts.join("\n"),
        unknown_fields,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::gateway::types::ProviderExt;

    fn full_wire_request() -> CreateChatCompletionRequest {
        let mut request = CreateChatCompletionRequest {
            model: "openai/gpt-5.6".into(),
            messages: vec![
                ChatCompletionRequestMessage {
                    role: "developer".into(),
                    content: "be brief".into(),
                    ..Default::default()
                },
                ChatCompletionRequestMessage {
                    role: "user".into(),
                    content: "hi".into(),
                    ..Default::default()
                },
            ],
            temperature: Some(0.7),
            max_tokens: Some(256),
            top_p: Some(0.9),
            stop: Some(vec!["END".into(), "STOP".into()]),
            stream: Some(false),
            ..Default::default()
        };
        request.unknown_fields.insert("top_k".into(), json!(40));
        request.unknown_fields.insert("seed".into(), json!(1234));
        request
            .unknown_fields
            .insert("vendor_beta".into(), json!(true));
        request.messages[1]
            .unknown_fields
            .insert("name".into(), json!("alice"));
        request.messages[1]
            .unknown_fields
            .insert("vendor_tag".into(), json!("vt"));
        request
    }

    /// An OpenAI-compatible request must survive normalization and come
    /// back out with zero data loss. The only permitted transformations:
    /// the provider prefix is stripped from `model`, `stream` is dropped,
    /// and an empty `stop` list is omitted.
    #[test]
    fn openai_compat_round_trip_is_lossless() {
        let original = full_wire_request();
        let normalized = ingest_request(original.clone(), "gpt-5.6");
        let wire = render_request(&normalized, "openai").unwrap();

        assert_eq!(wire.model, "gpt-5.6");
        assert_eq!(wire.messages.len(), 2);
        // The unrecognized "developer" role survives verbatim.
        assert_eq!(wire.messages[0].role, "developer");
        assert_eq!(wire.messages[0].content, "be brief");
        assert!(wire.messages[0].unknown_fields.is_empty());
        assert_eq!(wire.messages[1].role, "user");
        assert_eq!(wire.messages[1].content, "hi");
        // Message-level passthrough fields ride ext and come back verbatim.
        assert_eq!(wire.messages[1].unknown_fields["name"], json!("alice"));
        assert_eq!(wire.messages[1].unknown_fields["vendor_tag"], json!("vt"));
        assert_eq!(wire.temperature, Some(0.7));
        assert_eq!(wire.max_tokens, Some(256));
        assert_eq!(wire.top_p, Some(0.9));
        assert_eq!(wire.stop, Some(vec!["END".to_string(), "STOP".to_string()]));
        assert_eq!(wire.stream, None);
        // Request-level passthrough fields ride ext and come back verbatim.
        assert_eq!(wire.unknown_fields["top_k"], json!(40));
        assert_eq!(wire.unknown_fields["seed"], json!(1234));
        assert_eq!(wire.unknown_fields["vendor_beta"], json!(true));

        // Whole-value check so a future typed field can't slip past the
        // per-field assertions above.
        let mut expected = serde_json::to_value(&original).unwrap();
        expected["model"] = json!("gpt-5.6");
        expected.as_object_mut().unwrap().remove("stream");
        assert_eq!(serde_json::to_value(&wire).unwrap(), expected);
    }

    #[test]
    fn ingest_maps_typed_params_and_strips_prefix() {
        let normalized = ingest_request(full_wire_request(), "gpt-5.6");
        assert_eq!(normalized.model, "gpt-5.6");
        assert_eq!(normalized.params.temperature, Some(0.7));
        assert_eq!(normalized.params.max_tokens, Some(256));
        assert_eq!(normalized.params.top_p, Some(0.9));
        assert_eq!(normalized.params.stop, vec!["END", "STOP"]);
        assert!(!normalized.stream);
    }

    #[test]
    fn ingest_keeps_passthrough_fields_in_the_protocol_namespace() {
        let normalized = ingest_request(full_wire_request(), "gpt-5.6");
        // Every field without a typed wire home rides ext verbatim.
        assert_eq!(normalized.ext["openai_compat"]["top_k"], json!(40));
        assert_eq!(normalized.ext["openai_compat"]["seed"], json!(1234));
        assert_eq!(normalized.ext["openai_compat"]["vendor_beta"], json!(true));
        assert_eq!(
            normalized.messages[1].ext["openai_compat"]["name"],
            json!("alice")
        );
        assert_eq!(
            normalized.messages[1].ext["openai_compat"]["vendor_tag"],
            json!("vt")
        );
        // The unrecognized role is stashed for exact restoration.
        assert_eq!(
            normalized.messages[0].ext["openai_compat"]["role"],
            json!("developer")
        );
    }

    #[test]
    fn ingest_populates_source() {
        let original = full_wire_request();
        let normalized = ingest_request(original.clone(), "gpt-5.6");
        let source = normalized.source.as_ref().unwrap();
        assert_eq!(source.protocol, Protocol::OpenAiCompat);
        assert_eq!(source.body, serde_json::to_value(&original).unwrap());
    }

    #[test]
    fn render_merges_provider_ext_namespace_over_source_protocol() {
        let mut normalized = ingest_request(full_wire_request(), "deepseek-v4-flash");
        normalized.ext.insert(
            "deepseek".into(),
            json!({"vendor_beta": "ds-version", "ds_only": true}),
        );

        let for_deepseek = render_request(&normalized, "deepseek").unwrap();
        // The provider namespace overlays the protocol namespace...
        assert_eq!(
            for_deepseek.unknown_fields["vendor_beta"],
            json!("ds-version")
        );
        assert_eq!(for_deepseek.unknown_fields["ds_only"], json!(true));

        // ...and rendering for openai ignores it entirely.
        let for_openai = render_request(&normalized, "openai").unwrap();
        assert_eq!(for_openai.unknown_fields["vendor_beta"], json!(true));
        assert!(for_openai.unknown_fields.get("ds_only").is_none());
    }

    /// Rendering never filters params against what a target documents:
    /// the provider is the authority and rejects unsupported params
    /// itself.
    #[test]
    fn render_forwards_params_regardless_of_provider() {
        let normalized = ingest_request(full_wire_request(), "deepseek-v4-flash");
        let wire = render_request(&normalized, "deepseek").unwrap();
        // DeepSeek documents neither seed nor top_k; both still render.
        assert_eq!(wire.unknown_fields["seed"], json!(1234));
        assert_eq!(wire.unknown_fields["top_k"], json!(40));
    }

    #[test]
    fn render_rejects_staged_features() {
        let mut normalized = ingest_request(full_wire_request(), "gpt-5.6");
        normalized.tools.push(types::ToolDef {
            name: "search".into(),
            description: None,
            input_schema: json!({"type": "object"}),
            strict: None,
            cache: None,
            ext: ProviderExt::new(),
        });
        let err = render_request(&normalized, "openai").unwrap_err();
        assert!(matches!(err, Error::NotImplemented(_)), "{err:?}");
    }

    #[test]
    fn empty_stop_is_omitted_and_stream_never_rendered() {
        let mut wire_request = full_wire_request();
        wire_request.stop = None;
        wire_request.stream = Some(false);
        let normalized = ingest_request(wire_request, "gpt-5.6");
        let wire = render_request(&normalized, "openai").unwrap();
        assert_eq!(wire.stop, None);
        assert_eq!(wire.stream, None);
    }
}
