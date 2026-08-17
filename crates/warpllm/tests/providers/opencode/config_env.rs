use crate::openai_common::{openai_completion_body, request};
use warpllm::{Client, ClientConfig, Error};
use wiremock::matchers::{header, method};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Env mutation is process-global, so these scenarios run inside one test
/// body (temp-env serializes the unsafe set/unset around the closure).
#[test]
fn opencode_key_resolves_per_provider() {
    let runtime = tokio::runtime::Runtime::new().unwrap();

    temp_env::with_var("OPENCODE_API_KEY", Some("sk-opencode-env"), || {
        runtime.block_on(async {
            // 1. Zen requests use OPENCODE_API_KEY as bearer, and the key's
            //    last segment is what goes upstream — Zen's own identifiers
            //    are one segment, so no entry carries a `model:` override the
            //    way every `openrouter/` one has to.
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(header("authorization", "Bearer sk-opencode-env"))
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
                .chat_completions(request("opencode/kimi-k3"))
                .await
                .unwrap();

            let sent = &server.received_requests().await.unwrap()[0];
            let body: serde_json::Value = serde_json::from_slice(&sent.body).unwrap();
            assert_eq!(body["model"], "kimi-k3");
        });
    });

    temp_env::with_vars(
        [
            ("MOONSHOT_API_KEY", Some("sk-moonshot-env")),
            ("OPENCODE_API_KEY", None),
        ],
        || {
            runtime.block_on(async {
                // 2. Zen re-hosts Moonshot's weights, and a Moonshot key must
                //    still not satisfy it: `opencode/kimi-k3` and
                //    `kimi/kimi-k3` are two accounts at two hosts that happen
                //    to serve one model. The error names Zen's own variable.
                let client = Client::new(ClientConfig::default()).unwrap();
                let err = client
                    .chat_completions(request("opencode/kimi-k3"))
                    .await
                    .unwrap_err();
                match err {
                    Error::MissingApiKey { provider, env_var } => {
                        assert_eq!(provider, "opencode");
                        assert_eq!(env_var, Some("OPENCODE_API_KEY"));
                    }
                    other => panic!("expected MissingApiKey, got {other:?}"),
                }
            });
        },
    );
}
