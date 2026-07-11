use crate::openai_common::{openai_completion_body, request};
use warpllm::{Client, ClientConfig, Error};
use wiremock::matchers::{header, method};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Env mutation is process-global, so these three scenarios run inside one
/// test body (temp-env serializes the unsafe set/unset around the closure).
#[test]
fn env_fallback_explicit_override_and_missing_key() {
    let runtime = tokio::runtime::Runtime::new().unwrap();

    temp_env::with_var("OPENAI_API_KEY", Some("sk-from-env"), || {
        runtime.block_on(async {
            // 1. Env key is picked up.
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(header("authorization", "Bearer sk-from-env"))
                .respond_with(ResponseTemplate::new(200).set_body_json(openai_completion_body()))
                .expect(1)
                .mount(&server)
                .await;
            let client = Client::new(ClientConfig {
                base_url: Some(server.uri()),
                ..Default::default()
            })
            .unwrap();
            client
                .chat_completion(request("openai/gpt-4o"))
                .await
                .unwrap();

            // 2. Explicit constructor key wins over env.
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(header("authorization", "Bearer sk-explicit"))
                .respond_with(ResponseTemplate::new(200).set_body_json(openai_completion_body()))
                .expect(1)
                .mount(&server)
                .await;
            let client = Client::new(ClientConfig {
                openai_api_key: Some("sk-explicit".into()),
                base_url: Some(server.uri()),
                ..Default::default()
            })
            .unwrap();
            client
                .chat_completion(request("openai/gpt-4o"))
                .await
                .unwrap();
        });
    });

    temp_env::with_var("OPENAI_API_KEY", None::<&str>, || {
        runtime.block_on(async {
            // 3. Missing key errors at request time, naming the env var.
            let client = Client::new(ClientConfig::default()).unwrap();
            let err = client
                .chat_completion(request("openai/gpt-4o"))
                .await
                .unwrap_err();
            match err {
                Error::MissingApiKey { provider, env_var } => {
                    assert_eq!(provider, "openai");
                    assert_eq!(env_var, "OPENAI_API_KEY");
                }
                other => panic!("expected MissingApiKey, got {other:?}"),
            }
        });
    });
}

#[tokio::test]
async fn missing_base_url_errors_without_calling_openai() {
    let client = Client::new(ClientConfig {
        openai_api_key: Some("sk-test-openai".into()),
        ..Default::default()
    })
    .unwrap();

    let err = client
        .chat_completion(request("openai/gpt-4o"))
        .await
        .unwrap_err();

    match err {
        Error::InvalidInput(message) => assert_eq!(message, "missing base_url"),
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}
