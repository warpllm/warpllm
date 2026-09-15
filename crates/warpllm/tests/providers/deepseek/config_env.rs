use crate::openai_common::{openai_completion_body, request};
use std::collections::BTreeMap;

use warpllm::{Client, ClientConfig, Error, ProviderConfig};
use wiremock::matchers::{header, method};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Env mutation is process-global, so these scenarios run inside one test
/// body (temp-env serializes the unsafe set/unset around the closure).
#[test]
fn deepseek_key_resolves_per_provider() {
    let runtime = tokio::runtime::Runtime::new().unwrap();

    temp_env::with_var("DEEPSEEK_API_KEY", Some("sk-deepseek-env"), || {
        runtime.block_on(async {
            // 1. DeepSeek requests use DEEPSEEK_API_KEY as bearer.
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(header("authorization", "Bearer sk-deepseek-env"))
                .respond_with(ResponseTemplate::new(200).set_body_json(openai_completion_body()))
                .expect(1)
                .mount(&server)
                .await;
            let client = Client::new(ClientConfig {
                providers: Some(BTreeMap::from([(
                    "deepseek".to_string(),
                    ProviderConfig {
                        api_key: None,
                        base_url: Some(server.uri()),
                    },
                )])),
                ..Default::default()
            })
            .unwrap();
            client
                .chat_completions(request("deepseek/deepseek-v4-flash"))
                .await
                .unwrap();
        });
    });

    temp_env::with_vars(
        [
            ("OPENAI_API_KEY", Some("sk-openai-env")),
            ("DEEPSEEK_API_KEY", None),
        ],
        || {
            runtime.block_on(async {
                // 2. An OpenAI key must not satisfy DeepSeek. The model IS
                //    registered, so the roster admits it and only the second
                //    gate — the providers this client authenticated when it was
                //    built — can reject it. The mock would answer 200 to
                //    anything, so a request reaching it means the gate opened
                //    on a provider warpllm holds no key for.
                let server = MockServer::start().await;
                Mock::given(method("POST"))
                    .respond_with(
                        ResponseTemplate::new(200).set_body_json(openai_completion_body()),
                    )
                    .mount(&server)
                    .await;
                let client = Client::new(ClientConfig {
                    providers: Some(BTreeMap::from([(
                        "deepseek".to_string(),
                        ProviderConfig {
                            api_key: None,
                            base_url: Some(server.uri()),
                        },
                    )])),
                    ..Default::default()
                })
                .unwrap();
                let err = client
                    .chat_completions(request("deepseek/deepseek-v4-flash"))
                    .await
                    .unwrap_err();
                match err {
                    Error::MissingApiKey { provider, env_var } => {
                        assert_eq!(provider, "deepseek");
                        assert_eq!(env_var, Some("DEEPSEEK_API_KEY"));
                    }
                    other => panic!("expected MissingApiKey, got {other:?}"),
                }
                assert!(
                    server.received_requests().await.unwrap().is_empty(),
                    "a provider with no key was still sent a request"
                );
            });
        },
    );
}
