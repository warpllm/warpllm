use crate::openai_common::{openai_completion_body, request};
use std::collections::BTreeMap;

use warpllm::{Client, ClientConfig, Error, ProviderConfig};
use wiremock::matchers::{header, method};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Env mutation is process-global, so these scenarios run inside one test
/// body (temp-env serializes the unsafe set/unset around the closure).
#[test]
fn env_api_key_with_override_and_missing_key() {
    let runtime = tokio::runtime::Runtime::new().unwrap();

    temp_env::with_var("OPENAI_API_KEY", Some("sk-from-env"), || {
        runtime.block_on(async {
            // 1. The env key is resolved at request time and used as bearer.
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(header("authorization", "Bearer sk-from-env"))
                .respond_with(ResponseTemplate::new(200).set_body_json(openai_completion_body()))
                .expect(1)
                .mount(&server)
                .await;
            let client = Client::new(ClientConfig {
                providers: Some(BTreeMap::from([(
                    "openai".to_string(),
                    ProviderConfig {
                        api_key: None,
                        base_url: Some(server.uri()),
                    },
                )])),
                ..Default::default()
            })
            .unwrap();
            client
                .chat_completions(request("openai/gpt-5.6"))
                .await
                .unwrap();
        });
    });

    temp_env::with_var("OPENAI_API_KEY", None::<&str>, || {
        runtime.block_on(async {
            // 2. Missing key errors at request time, naming the env var.
            let client = Client::new(ClientConfig::default()).unwrap();
            let err = client
                .chat_completions(request("openai/gpt-5.6"))
                .await
                .unwrap_err();
            match err {
                Error::MissingApiKey { provider, env_var } => {
                    assert_eq!(provider, "openai");
                    assert_eq!(env_var, Some("OPENAI_API_KEY"));
                }
                other => panic!("expected MissingApiKey, got {other:?}"),
            }
        });
    });
}
