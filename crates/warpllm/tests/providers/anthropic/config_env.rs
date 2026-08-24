//! Anthropic's credential, which is the one on the roster that is not a bearer
//! token.

use crate::openai_common::{
    ANTHROPIC_KEY, anthropic_message_body, client_for, request, with_anthropic_key,
};
use warpllm::{Client, ClientConfig, Error};
use wiremock::matchers::{header, header_exists, method};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// The header, and the header that must NOT also be set.
///
/// Anthropic reads `x-api-key` and nothing else, and an `Authorization` header
/// alongside it is not merely redundant: a proxy in front of Anthropic may read
/// one and be answered by the other. The scheme is picked per PROVIDER in
/// `Credentials::scheme`, so this is the end-to-end proof that the table is
/// reached from a real request rather than only unit-tested.
#[test]
fn an_anthropic_request_authenticates_with_x_api_key() {
    with_anthropic_key(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(header("x-api-key", ANTHROPIC_KEY))
            .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_message_body()))
            .expect(1)
            .mount(&server)
            .await;

        client_for(&server)
            .chat_completions(request("anthropic/claude-opus-5"))
            .await
            .unwrap();

        let sent = &server.received_requests().await.unwrap()[0];
        assert!(
            sent.headers.get("authorization").is_none(),
            "the key also went out as a bearer token"
        );
    });
}

/// An OpenAI key does not satisfy Anthropic, and the refusal names the variable
/// to set.
///
/// The model IS registered, so the roster admits it and only the credential gate
/// can reject it. The mock would answer 200 to anything, so a request reaching
/// it means the gate opened on a provider warpllm holds no key for.
#[test]
fn another_providers_key_does_not_reach_anthropic() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    temp_env::with_vars(
        [
            ("OPENAI_API_KEY", Some("sk-openai-env")),
            ("ANTHROPIC_API_KEY", None),
            ("WARPLLM_SPECS", None),
        ],
        || {
            runtime.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("POST"))
                    .respond_with(
                        ResponseTemplate::new(200).set_body_json(anthropic_message_body()),
                    )
                    .mount(&server)
                    .await;
                let client = Client::new(ClientConfig {
                    base_url: Some(server.uri()),
                    ..Default::default()
                })
                .unwrap();

                let err = client
                    .chat_completions(request("anthropic/claude-opus-5"))
                    .await
                    .unwrap_err();
                match err {
                    Error::MissingApiKey { provider, env_var } => {
                        assert_eq!(provider, "anthropic");
                        assert_eq!(env_var, Some("ANTHROPIC_API_KEY"));
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

/// The version header is a fact about this WIRE FORMAT rather than about the
/// provider, so it rides every request whatever the credential is. Asserted
/// separately from the happy path because the two go wrong for different
/// reasons: this one breaks when the transport is edited, not when the roster
/// is.
#[test]
fn every_anthropic_request_names_the_api_version() {
    with_anthropic_key(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(header_exists("anthropic-version"))
            .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_message_body()))
            .expect(1)
            .mount(&server)
            .await;

        client_for(&server)
            .chat_completions(request("anthropic/claude-opus-5"))
            .await
            .unwrap();

        assert_eq!(
            server.received_requests().await.unwrap()[0]
                .headers
                .get("anthropic-version")
                .unwrap(),
            "2023-06-01"
        );
    });
}
