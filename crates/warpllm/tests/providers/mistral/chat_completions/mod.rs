use crate::openai_common::{
    MISTRAL_KEY, client_for, openai_completion_body, request, with_mistral_key,
};
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Mistral is a pure data entry over the OpenAI-compatible protocol: the
/// same endpoint impl serves it, so this only proves routing — prefix
/// stripping, echo, and auth under the `mistral` name.
#[test]
fn mistral_happy_path() {
    with_mistral_key(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", format!("Bearer {MISTRAL_KEY}")))
            .respond_with(ResponseTemplate::new(200).set_body_json(openai_completion_body()))
            .expect(1)
            .mount(&server)
            .await;

        let completion = client_for(&server)
            .chat_completions(request("mistral/mistral-large-2411"))
            .await
            .unwrap();

        assert_eq!(completion.model, "mistral/mistral-large-2411");

        let sent: serde_json::Value =
            serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
        assert_eq!(sent["model"], "mistral-large-2411");
    });
}

/// Passthrough philosophy: warpllm never filters params against what a
/// provider documents. Mistral is the authority that accepts or rejects them.
#[test]
fn mistral_forwards_params_for_the_provider_to_judge() {
    with_mistral_key(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(openai_completion_body()))
            .expect(1)
            .mount(&server)
            .await;

        let mut req = request("mistral/mistral-large-2411");
        req.temperature = Some(Some(0.5));
        req.unknown_fields.insert("safe_prompt".into(), json!(true));
        client_for(&server).chat_completions(req).await.unwrap();

        let sent: serde_json::Value =
            serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
        assert_eq!(sent["temperature"], 0.5);
        assert_eq!(sent["safe_prompt"], true);
    });
}
