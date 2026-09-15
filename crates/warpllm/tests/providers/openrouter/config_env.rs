use crate::openai_common::{openai_completion_body, request};
use std::collections::BTreeMap;

use warpllm::{Client, ClientConfig, Error, ProviderConfig};
use wiremock::matchers::{header, method};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Env mutation is process-global, so these scenarios run inside one test
/// body (temp-env serializes the unsafe set/unset around the closure).
#[test]
fn openrouter_key_resolves_per_provider() {
    let runtime = tokio::runtime::Runtime::new().unwrap();

    temp_env::with_var("OPENROUTER_API_KEY", Some("sk-openrouter-env"), || {
        runtime.block_on(async {
            // 1. OpenRouter requests use OPENROUTER_API_KEY as bearer, and the
            //    full slug (`anthropic/claude-sonnet-4`) goes upstream, not the
            //    key's last segment.
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(header("authorization", "Bearer sk-openrouter-env"))
                .respond_with(ResponseTemplate::new(200).set_body_json(openai_completion_body()))
                .expect(1)
                .mount(&server)
                .await;
            let client = Client::new(ClientConfig {
                providers: Some(BTreeMap::from([(
                    "openrouter".to_string(),
                    ProviderConfig {
                        api_key: None,
                        base_url: Some(server.uri()),
                    },
                )])),
                ..Default::default()
            })
            .unwrap();
            client
                .chat_completions(request("openrouter/anthropic/claude-sonnet-4"))
                .await
                .unwrap();

            let sent = &server.received_requests().await.unwrap()[0];
            let body: serde_json::Value = serde_json::from_slice(&sent.body).unwrap();
            assert_eq!(body["model"], "anthropic/claude-sonnet-4");
        });
    });

    temp_env::with_vars(
        [
            ("OPENAI_API_KEY", Some("sk-openai-env")),
            ("OPENROUTER_API_KEY", None),
        ],
        || {
            runtime.block_on(async {
                // 2. An OpenAI key must not satisfy OpenRouter: the missing
                //    key errors at request time, naming OpenRouter's env var.
                let client = Client::new(ClientConfig::default()).unwrap();
                let err = client
                    .chat_completions(request("openrouter/anthropic/claude-sonnet-4"))
                    .await
                    .unwrap_err();
                match err {
                    Error::MissingApiKey { provider, env_var } => {
                        assert_eq!(provider, "openrouter");
                        assert_eq!(env_var, Some("OPENROUTER_API_KEY"));
                    }
                    other => panic!("expected MissingApiKey, got {other:?}"),
                }
            });
        },
    );
}
