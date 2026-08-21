use crate::openai_common::{
    DEEPSEEK_KEY, client_for, openai_completion_body, request, with_deepseek_key,
};
use serde_json::json;
use warpllm::Error;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// `reasoning_content` is promoted to a normalized Reasoning block by the
/// protocol's own ingest, for every provider (unit-tested in
/// `gateway::openai_compat::api::chat_completions::response`). This is the other
/// half of that contract, asserted here because DeepSeek is the provider that
/// actually sends the field: it must ALSO still reach the caller byte-for-byte.
///
/// That only holds because ingest RETAINS the `ext` entry rather than consuming
/// it — the renderer drops Reasoning blocks, so nothing else could rebuild the
/// field. Consuming it would break this test.
///
/// It cannot assert the block exists: render drops Reasoning blocks by design,
/// so the block is invisible from out here.
#[test]
fn reasoning_content_survives_the_round_trip_to_the_caller() {
    with_deepseek_key(async {
        let mut body = openai_completion_body();
        body["model"] = json!("deepseek-v4-pro");
        body["choices"][0]["message"]["reasoning_content"] = json!("step by step");

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body.clone()))
            .expect(1)
            .mount(&server)
            .await;

        let completion = client_for(&server)
            .chat_completions(request("deepseek/deepseek-v4-pro"))
            .await
            .unwrap();

        // Model echo aside; the explicit `"logprobs": null` reaches the
        // caller as the null it was sent as.
        let mut expected = body;
        expected["model"] = json!("deepseek/deepseek-v4-pro");
        assert_eq!(serde_json::to_value(&completion).unwrap(), expected);
    });
}

/// DeepSeek is otherwise a pure data entry over the OpenAI-compatible protocol:
/// the same endpoint impl serves it, so this only proves routing — prefix
/// stripping, echo, and error attribution under the `deepseek` name.
#[test]
fn deepseek_happy_path() {
    with_deepseek_key(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", format!("Bearer {DEEPSEEK_KEY}")))
            .respond_with(ResponseTemplate::new(200).set_body_json(openai_completion_body()))
            .expect(1)
            .mount(&server)
            .await;

        let completion = client_for(&server)
            .chat_completions(request("deepseek/deepseek-v4-flash"))
            .await
            .unwrap();

        // Echoes the caller's provider-prefixed string, not the upstream name.
        assert_eq!(completion.model, "deepseek/deepseek-v4-flash");

        let sent: serde_json::Value =
            serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
        // The provider prefix must be stripped from the outbound model.
        assert_eq!(sent["model"], "deepseek-v4-flash");
    });
}

/// Passthrough philosophy: warpllm never filters params against what a
/// provider documents. seed is absent from DeepSeek's parameters and 3.0
/// exceeds its documented temperature range — both are forwarded verbatim
/// and DeepSeek itself is the authority that rejects them.
#[test]
fn deepseek_forwards_params_for_the_provider_to_judge() {
    with_deepseek_key(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(openai_completion_body()))
            .expect(1)
            .mount(&server)
            .await;

        let mut req = request("deepseek/deepseek-v4-flash");
        req.temperature = Some(Some(3.0));
        req.unknown_fields.insert("seed".into(), json!(7));
        client_for(&server).chat_completions(req).await.unwrap();

        let sent: serde_json::Value =
            serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
        assert_eq!(sent["temperature"], 3.0);
        assert_eq!(sent["seed"], 7);
    });
}

/// DeepSeek's logprobs object has no `refusal` array and adds
/// `reasoning_content`; a successful upstream response must decode and
/// reach the caller verbatim.
#[test]
fn deepseek_logprobs_without_refusal_decodes() {
    with_deepseek_key(async {
        let mut body = openai_completion_body();
        body["model"] = json!("deepseek-v4-flash");
        body["choices"][0]["logprobs"] = json!({
            "content": [
                {"token": "Hello", "logprob": -0.05, "bytes": null, "top_logprobs": []}
            ],
            "reasoning_content": []
        });

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body.clone()))
            .expect(1)
            .mount(&server)
            .await;

        let completion = client_for(&server)
            .chat_completions(request("deepseek/deepseek-v4-flash"))
            .await
            .unwrap();

        let value = serde_json::to_value(&completion).unwrap();
        assert_eq!(
            value["choices"][0]["logprobs"]["content"][0]["token"],
            "Hello"
        );
        assert_eq!(
            value["choices"][0]["logprobs"]["reasoning_content"],
            json!([])
        );
        assert!(
            value["choices"][0]["logprobs"].get("refusal").is_none(),
            "absent refusal must stay absent"
        );
    });
}

#[test]
fn deepseek_errors_name_deepseek() {
    with_deepseek_key(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429).set_body_json(json!({
                "error": {"message": "Rate limit reached", "type": "rate_limit_exceeded"}
            })))
            .mount(&server)
            .await;

        let err = client_for(&server)
            .chat_completions(request("deepseek/deepseek-v4-flash"))
            .await
            .unwrap_err();

        // Classified by the protocol, so a second provider on it inherits
        // the taxonomy without a line of Rust.
        assert!(matches!(err, Error::RateLimited(_)), "{err:?}");
        let upstream = err.provider_error().expect("a provider failure");
        assert_eq!(upstream.provider, "deepseek");
        assert_eq!(upstream.status, 429);
    });
}
