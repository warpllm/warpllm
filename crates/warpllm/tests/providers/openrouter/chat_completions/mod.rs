use crate::openai_common::{
    OPENROUTER_KEY, client_for, openai_completion_body, request, with_openrouter_key,
};
use serde_json::{Value, json};
use warpllm::Error;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// An OpenRouter completion: the OpenAI shape, plus the aggregator's own
/// envelope — `provider` (which backend served it), and the `usage` cost
/// extensions OpenRouter appends.
fn openrouter_completion_body() -> Value {
    let mut body = openai_completion_body();
    body["model"] = json!("anthropic/claude-sonnet-4");
    body["provider"] = json!("StreamLake");
    body["usage"]["is_byok"] = json!(false);
    body["usage"]["cost"] = json!(6.1642e-7);
    body["usage"]["cost_details"]["upstream_inference_cost"] = json!(6.1642e-7);
    body["usage"]["cost_details"]["upstream_inference_prompt_cost"] = json!(4.403e-7);
    body["usage"]["cost_details"]["upstream_inference_completions_cost"] = json!(1.7612e-7);
    body
}

/// OpenRouter's chat completions follow the OpenAI-compatible protocol, so
/// the happy path proves routing only: OPENROUTER_API_KEY as bearer, the
/// caller's string echoed back, and — unlike OpenAI or DeepSeek — the FULL
/// two-segment slug on the wire. The key's last-segment default would
/// truncate `anthropic/claude-sonnet-4` to `claude-sonnet-4`; the roster's
/// explicit `model:` field is what ships it whole.
#[test]
fn openrouter_happy_path() {
    with_openrouter_key(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", format!("Bearer {OPENROUTER_KEY}")))
            .respond_with(ResponseTemplate::new(200).set_body_json(openrouter_completion_body()))
            .expect(1)
            .mount(&server)
            .await;

        let completion = client_for(&server)
            .chat_completions(request("openrouter/anthropic/claude-sonnet-4"))
            .await
            .unwrap();

        // Echoes the caller's provider-prefixed string, not the upstream name.
        assert_eq!(completion.model, "openrouter/anthropic/claude-sonnet-4");

        let sent: serde_json::Value =
            serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
        // The FULL slug ships upstream — the thing that makes an aggregator
        // entry different from a direct-provider one.
        assert_eq!(sent["model"], "anthropic/claude-sonnet-4");
    });
}

/// Passthrough philosophy over an aggregator: OpenRouter adds its own
/// envelope around the OpenAI-compatible shape, and warpllm must hand the
/// caller exactly what it sent. `provider`, the `usage` cost extensions, and
/// `is_byok` all survive ingest → normalize → render.
#[test]
fn openrouter_extensions_round_trip_to_the_caller() {
    with_openrouter_key(async {
        let body = openrouter_completion_body();

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body.clone()))
            .expect(1)
            .mount(&server)
            .await;

        let completion = client_for(&server)
            .chat_completions(request("openrouter/anthropic/claude-sonnet-4"))
            .await
            .unwrap();

        // Model echo aside; the explicit `"logprobs": null` reaches the
        // caller as the null it was sent as.
        let mut expected = body;
        expected["model"] = json!("openrouter/anthropic/claude-sonnet-4");
        assert_eq!(serde_json::to_value(&completion).unwrap(), expected);
    });
}

/// Params pass through to OpenRouter verbatim, including ones it alone
/// understands. The `provider` routing hint (which backend should serve the
/// request) and `route` are OpenRouter extensions, not OpenAI's; the provider
/// is the authority on its own parameters, never warpllm.
#[test]
fn openrouter_forwards_params_for_the_provider_to_judge() {
    with_openrouter_key(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(openrouter_completion_body()))
            .expect(1)
            .mount(&server)
            .await;

        let mut req = request("openrouter/anthropic/claude-sonnet-4");
        req.temperature = Some(Some(3.0));
        req.unknown_fields
            .insert("provider".into(), json!("StreamLake"));
        req.unknown_fields.insert("route".into(), json!("fallback"));
        client_for(&server).chat_completions(req).await.unwrap();

        let sent: serde_json::Value =
            serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
        assert_eq!(sent["temperature"], 3.0);
        assert_eq!(sent["provider"], "StreamLake");
        assert_eq!(sent["route"], "fallback");
    });
}

/// OpenRouter's error envelope is the OpenAI one, so a faithful provider on
/// the protocol inherits the taxonomy without a line of Rust — the error just
/// has to be attributed to `openrouter`, not to a protocol default.
#[test]
fn openrouter_errors_name_openrouter() {
    with_openrouter_key(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429).set_body_json(json!({
                "error": {"message": "Rate limit reached", "type": "rate_limit_exceeded"}
            })))
            .mount(&server)
            .await;

        let err = client_for(&server)
            .chat_completions(request("openrouter/anthropic/claude-sonnet-4"))
            .await
            .unwrap_err();

        assert!(matches!(err, Error::RateLimited(_)), "{err:?}");
        let upstream = err.provider_error().expect("a provider failure");
        assert_eq!(upstream.provider, "openrouter");
        assert_eq!(upstream.status, 429);
    });
}

/// OpenRouter is an aggregator, so its roster keys carry a vendor prefix that
/// the direct providers do not. The registry is closed either way: an
/// unlisted slug is an error before any request, and a bare model name under
/// `openrouter/` cannot silently fall back to a direct provider's entry.
#[test]
fn openrouter_unlisted_slugs_are_rejected() {
    with_openrouter_key(async {
        let server = MockServer::start().await;
        let client = client_for(&server);

        let err = client
            .chat_completions(request("openrouter/anthropic/claude-3.5-sonnet"))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("no registered model spec"),
            "{err}"
        );
        assert!(server.received_requests().await.unwrap().is_empty());
    });
}
