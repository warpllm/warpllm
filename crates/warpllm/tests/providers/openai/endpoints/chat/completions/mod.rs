use crate::openai_common::{client_for, openai_completion_body, request};
use serde_json::json;
use warpllm::Error;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn openai_happy_path() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer sk-test-openai"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_completion_body()))
        .expect(1)
        .mount(&server)
        .await;

    let completion = client_for(&server)
        .chat_completion(request("openai/gpt-4o"))
        .await
        .unwrap();

    assert_eq!(completion.id, "chatcmpl-123");
    assert_eq!(completion.object, "chat.completion");
    // Echoes the caller's provider-prefixed string, not the upstream name.
    assert_eq!(completion.model, "openai/gpt-4o");
    assert_eq!(
        completion.choices[0].message.content.as_deref(),
        Some("Hello there!")
    );
    assert_eq!(completion.choices[0].finish_reason, "stop");
    assert_eq!(completion.service_tier.as_deref(), Some("default"));
    assert_eq!(
        completion.system_fingerprint.as_deref(),
        Some("fp_44709d6fcb")
    );
    let usage = completion.usage.as_ref().unwrap();
    assert_eq!(usage.total_tokens, 21);
    let prompt_details = usage.prompt_tokens_details.as_ref().unwrap();
    assert_eq!(prompt_details.cached_tokens, Some(3));
    assert_eq!(prompt_details.cache_write_tokens, Some(2));
    assert_eq!(
        usage
            .completion_tokens_details
            .as_ref()
            .unwrap()
            .reasoning_tokens,
        Some(5)
    );

    let sent: serde_json::Value =
        serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
    // The provider prefix must be stripped from the outbound model.
    assert_eq!(sent["model"], "gpt-4o");
    assert_eq!(sent["messages"][0]["content"], "hi");
    assert!(sent.get("stream").is_none());
}

#[tokio::test]
async fn openai_error_statuses_map_to_provider_error() {
    for (status, error_type, message) in [
        (401, "invalid_request_error", "Incorrect API key provided"),
        (429, "rate_limit_exceeded", "Rate limit reached"),
        (500, "server_error", "The server had an error"),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(status).set_body_json(json!({
                "error": {"message": message, "type": error_type}
            })))
            .mount(&server)
            .await;

        let err = client_for(&server)
            .chat_completion(request("openai/gpt-4o"))
            .await
            .unwrap_err();

        match err {
            Error::Provider {
                provider,
                status: got_status,
                error_type: got_type,
                message: got_message,
            } => {
                assert_eq!(provider, "openai");
                assert_eq!(got_status, status);
                assert_eq!(got_type.as_deref(), Some(error_type));
                assert_eq!(got_message, message);
            }
            other => panic!("expected Provider error, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn malformed_success_body_maps_to_decode_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&server)
        .await;

    let err = client_for(&server)
        .chat_completion(request("openai/gpt-4o"))
        .await
        .unwrap_err();
    assert!(
        matches!(
            err,
            Error::Decode {
                provider: "openai",
                ..
            }
        ),
        "{err:?}"
    );
}

#[tokio::test]
async fn unparseable_error_body_falls_back_to_raw_text() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(503).set_body_string("upstream overloaded"))
        .mount(&server)
        .await;

    let err = client_for(&server)
        .chat_completion(request("openai/gpt-4o"))
        .await
        .unwrap_err();
    match err {
        Error::Provider {
            status,
            error_type,
            message,
            ..
        } => {
            assert_eq!(status, 503);
            assert_eq!(error_type, None);
            assert_eq!(message, "upstream overloaded");
        }
        other => panic!("expected Provider error, got {other:?}"),
    }
}

#[tokio::test]
async fn stream_true_is_rejected_before_any_request() {
    let server = MockServer::start().await;
    // No mock mounted: a request reaching the server would 404 into a
    // Provider error, so getting NotImplemented proves we rejected early.
    let mut req = request("openai/gpt-4o");
    req.stream = Some(true);

    let err = client_for(&server).chat_completion(req).await.unwrap_err();
    assert!(matches!(err, Error::NotImplemented(_)), "{err:?}");
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn invalid_model_strings_are_rejected() {
    let server = MockServer::start().await;
    let client = client_for(&server);

    let err = client.chat_completion(request("gpt-4o")).await.unwrap_err();
    assert!(
        err.to_string().contains("not a supported provider"),
        "{err}"
    );

    let err = client
        .chat_completion(request("mistral/large"))
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("not a supported provider"),
        "{err}"
    );
}
