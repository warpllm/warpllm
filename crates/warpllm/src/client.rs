//! The client: one pooled HTTP connection set, one entrypoint.

use std::time::Duration;

use crate::config::{ClientConfig, DEFAULT_TIMEOUT_SECS};
use crate::credentials::Credentials;
use crate::error::{Error, Result};
use crate::gateway::openai_compat;
use crate::protocol::openai_compat::chat_completions::types::{
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
};
use crate::registry::{ModelSpec, ProviderSpec, fetch_model};
use crate::types::Api;

pub struct Client {
    http: reqwest::Client,
    config: ClientConfig,
    credentials: Credentials,
}

impl Client {
    /// Reads the environment once and keeps the providers it can authenticate.
    ///
    /// Constructing a client still never *requires* credentials: an environment
    /// holding none builds a client that reaches no provider, and each request
    /// says which variable it wanted. What construction does is answer that
    /// question up front, so it can be logged at the moment a caller is set up
    /// to read it rather than discovered one failed request at a time.
    pub fn new(config: ClientConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(
                config.timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS),
            ))
            .build()
            .map_err(|e| Error::Internal(e.to_string()))?;
        Ok(Self {
            http,
            config,
            credentials: Credentials::from_env(),
        })
    }

    /// Serves one OpenAI-compatible chat completion.
    ///
    /// A request is admitted only when BOTH halves hold: the roster registers
    /// the model, and this client holds a key for the provider serving it. They
    /// are checked in that order deliberately — a name nothing registers is a
    /// typo, and hearing "set `DEEPSEEK_API_KEY`" for `deepseek/typo` would send
    /// someone after a credential they already have.
    pub async fn chat_completions(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse> {
        if request.stream == Some(true) {
            // Not "unimplemented": it IS implemented, on a method whose return
            // type can carry chunks. A whole reply cannot, so this entrypoint
            // says where to go rather than quietly serving the wrong shape.
            return Err(Error::InvalidInput(
                "stream: true asks for chunks; call chat_completions_stream".into(),
            ));
        }
        let requested_model = request.model.clone();
        // Half one: the roster. Closed, so an unregistered name stops here.
        let (provider, model) = fetch_model(&requested_model)?;
        Self::validate_api(
            model,
            Api::OpenAiCompatChatCompletions,
            provider,
            &requested_model,
        )?;
        // Half two: this client. A registered model whose provider was not
        // authenticated at construction never reaches the network.
        let api_key = self.api_key(provider)?;

        // Ingest answers to the protocol warpllm was CALLED with, which is
        // openai_compat and only ever will be for this entrypoint. The ENTRY's
        // model name goes in, not the caller's string: they differ whenever
        // warpllm's routing alias differs from the provider's own name.
        let normalized =
            openai_compat::api::chat_completions::ingest_request(request, model.model());
        // No dispatch to do: the surface above names its own protocol, so
        // asking for `openai_compat_chat_completions` IS the choice of module.
        // A second protocol arrives as a second entrypoint, not as an arm here
        // — this one takes an OpenAI-shaped request by signature.
        let response = openai_compat::api::chat_completions::exchange(
            &normalized,
            &self.http,
            provider.name(),
            self.base_url(provider),
            api_key,
        )
        .await?;
        let mut completion =
            openai_compat::api::chat_completions::render_response(&response, provider.name());
        // Echo the caller's provider-prefixed string, not the upstream echo.
        completion.model = requested_model;
        Ok(completion)
    }

    /// Serves one OpenAI-compatible chat completion as a stream of chunks.
    ///
    /// The same two admission halves as [`Client::chat_completions`], in the
    /// same order, against a DIFFERENT surface: streaming is its own entry in
    /// the roster, so a model that serves whole replies says nothing about
    /// whether it serves streamed ones.
    ///
    /// `stream` is set on the caller's behalf rather than required: the method
    /// name already states the intent, and a request that contradicts it would
    /// have nothing useful to mean.
    ///
    /// The [`Result`] covers only what can fail before the first chunk. Once
    /// the stream is open, failures arrive as items on it.
    pub async fn chat_completions_stream(
        &self,
        mut request: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionStream> {
        request.stream = Some(true);
        let requested_model = request.model.clone();
        let (provider, model) = fetch_model(&requested_model)?;
        Self::validate_api(
            model,
            Api::OpenAiCompatChatCompletionsStream,
            provider,
            &requested_model,
        )?;
        let api_key = self.api_key(provider)?;

        let normalized =
            openai_compat::api::chat_completions::ingest_request(request, model.model());
        Ok(ChatCompletionStream {
            chunks: openai_compat::api::chat_completions::exchange_stream(
                &normalized,
                &self.http,
                provider.name(),
                self.base_url(provider),
                api_key,
                self.config
                    .stream_read_timeout_secs
                    .map(Duration::from_secs),
            )
            .await?,
            provider: provider.name(),
            model: requested_model,
        })
    }

    /// The routed provider's key, from the snapshot this client took of the
    /// environment when it was built.
    ///
    /// A miss is not "the variable is unset now" but "it was unset then", and
    /// the error still names the variable to set, because that is the remedy
    /// either way. A provider with no `env_api_key` has no key source at all,
    /// so the error names the roster rather than a variable nothing reads.
    fn api_key(&self, provider: &'static ProviderSpec) -> Result<&str> {
        self.credentials
            .get(provider.name())
            .ok_or(Error::MissingApiKey {
                provider: provider.name(),
                env_var: provider.env_api_key(),
            })
    }

    /// Whether the routed MODEL serves `api`, which is what every entrypoint
    /// has to establish before it sends anything.
    ///
    /// Asking the model is the whole point, and the only place the answer
    /// exists: a provider is a host, and one host commonly serves chat
    /// completions, embeddings, and moderation from three disjoint sets of
    /// models. There is nothing at the provider level to ask.
    ///
    /// Takes the surface rather than naming one, so the second entrypoint
    /// reuses this rather than copying it — and so the refusal cannot claim
    /// the wrong surface, which a hard-coded message eventually would.
    ///
    /// `model` and `api` are the check; `provider` and `requested` only build
    /// the message. Split out from [`Client::chat_completions`] so the refusal
    /// can be tested at all: every model the roster ships today serves chat
    /// completions, so the failing branch is otherwise unreachable from the
    /// public entrypoint until one lands that does not.
    fn validate_api(
        model: &'static ModelSpec,
        api: Api,
        provider: &'static ProviderSpec,
        requested: &str,
    ) -> Result<()> {
        if model.supports_api(api) {
            return Ok(());
        }
        // The roster's own spelling for the surface, and the provider because
        // the roster is where what-is-served is recorded — between them they
        // name the line a reader would go and fix.
        Err(Error::InvalidInput(format!(
            "{}: {requested} does not serve {}",
            provider.name(),
            api.as_str()
        )))
    }

    /// A configured `base_url` overrides the provider default (proxies,
    /// tests); otherwise each provider talks to its own API.
    fn base_url(&self, provider: &'static ProviderSpec) -> &str {
        self.config
            .base_url
            .as_deref()
            .unwrap_or(provider.base_url())
    }
}

/// The chunks of one streamed reply, in the shape the caller asked in.
///
/// Returned by [`Client::chat_completions_stream`]. Iterate it to exhaustion:
///
/// ```no_run
/// # async fn demo(client: &warpllm::Client, request: warpllm::CreateChatCompletionRequest)
/// # -> warpllm::Result<()> {
/// let mut stream = client.chat_completions_stream(request).await?;
/// while let Some(chunk) = stream.next().await {
///     for choice in &chunk?.choices {
///         if let Some(Some(text)) = &choice.delta.content {
///             print!("{text}");
///         }
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// An inherent `next` rather than a [`Stream`](std::iter::Iterator)
/// implementation: a `while let` loop needs no combinators, and the bindings
/// that wrap this iterate it the same way.
#[derive(Debug)]
pub struct ChatCompletionStream {
    chunks: openai_compat::api::chat_completions::ChatChunkStream,
    provider: &'static str,
    /// The caller's provider-prefixed string, echoed onto every chunk in place
    /// of the upstream's own — the streaming counterpart of the one
    /// [`Client::chat_completions`] performs on a whole reply.
    model: String,
}

impl ChatCompletionStream {
    /// The next chunk, or `None` once the stream ends.
    ///
    /// `None` means the reply is COMPLETE. An upstream that stopped early ends
    /// with [`Error::StreamTruncated`](crate::Error::StreamTruncated) instead,
    /// after every chunk that did arrive — so a caller never has to wonder
    /// whether the answer it collected is the whole one.
    ///
    /// An error item is terminal: whatever produced it also ended the stream,
    /// so the next call returns `None`.
    pub async fn next(&mut self) -> Option<Result<CreateChatCompletionStreamResponse>> {
        let chunk = self.chunks.next().await?;
        Some(chunk.map(|chunk| {
            let mut rendered =
                openai_compat::api::chat_completions::render_chunk(&chunk, self.provider);
            rendered.model = self.model.clone();
            rendered
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credentials::with_env;
    use crate::protocol::openai_compat::chat_completions::types::ChatCompletionRequestMessage;
    use crate::registry::{Capabilities, ModelSpec, SupportedApi};

    /// A client built under the environment lock.
    ///
    /// `Client::new` reads the environment, so every construction in this
    /// binary has to hold temp-env's lock even when the test has no opinion
    /// about what is set — see [`with_env`] for why.
    fn client(config: ClientConfig) -> Client {
        with_env(&[], || Client::new(config).unwrap())
    }

    /// The two halves the client works from, for a model the shipped roster
    /// does have.
    fn pair_for(model: &str) -> (&'static ProviderSpec, &'static ModelSpec) {
        fetch_model(model).unwrap()
    }

    /// Leaked because the client takes `&'static` specs, which costs nothing
    /// in a test process that is about to exit.
    fn demo_provider(base_url: &str, env_api_key: Option<&str>) -> &'static ProviderSpec {
        Box::leak(Box::new(ProviderSpec {
            name: "demo".into(),
            base_url: base_url.into(),
            env_api_key: env_api_key.map(str::to_string),
        }))
    }

    /// Leaked for the same reason as [`demo_provider`]. Takes its surfaces so
    /// a caller can express the case the roster cannot yet: a model under a
    /// chat-serving host that does not itself serve chat.
    fn demo_model(supported_apis: Vec<SupportedApi>) -> &'static ModelSpec {
        Box::leak(Box::new(ModelSpec {
            provider: "demo".into(),
            model: "demo-embed".into(),
            supported_apis,
            capabilities: Capabilities::blank(),
            deprecation_date: None,
        }))
    }

    /// The gate the whole model-level `supported_apis` split exists for. This
    /// model sits under a perfectly ordinary chat-serving provider and does
    /// not serve chat itself, which only the model's own list can say.
    #[test]
    fn a_model_that_does_not_serve_the_api_is_refused() {
        let err = Client::validate_api(
            demo_model(vec![SupportedApi {
                api: Api::OpenAiCompatResponses,
            }]),
            Api::OpenAiCompatChatCompletions,
            demo_provider("https://api.demo.test", Some("DEMO_API_KEY")),
            "demo/embed",
        )
        .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("demo/embed"), "{message}");
        // The roster's spelling, not the variant's — this is the string a
        // reader greps `specs.yaml` for.
        assert!(
            message.contains("does not serve openai_compat_chat_completions"),
            "{message}"
        );

        // A 400: the caller asked for something this model cannot do, which is
        // theirs to fix, not the provider's to fail.
        let wire: serde_json::Value = serde_json::from_str(&err.to_openai_json()).unwrap();
        assert_eq!(wire["status"], 400);
    }

    /// The other side of the same gate: a model that does serve the surface
    /// passes, so the check cannot be one that refuses everything. Listing a
    /// second surface alongside it changes nothing — each is its own claim.
    #[test]
    fn a_model_that_serves_the_api_is_admitted() {
        Client::validate_api(
            demo_model(vec![
                SupportedApi {
                    api: Api::OpenAiCompatChatCompletions,
                },
                SupportedApi {
                    api: Api::OpenAiCompatResponses,
                },
            ]),
            Api::OpenAiCompatChatCompletions,
            demo_provider("https://api.demo.test", Some("DEMO_API_KEY")),
            "demo/chat",
        )
        .unwrap();
    }

    /// The point of passing the surface in: one model, and the answer depends
    /// on which surface is asked about. A check that ignored its argument
    /// would pass both of the tests above and fail here.
    #[test]
    fn the_answer_depends_on_which_api_is_asked_about() {
        let model = demo_model(vec![SupportedApi {
            api: Api::OpenAiCompatResponses,
        }]);
        let provider = demo_provider("https://api.demo.test", Some("DEMO_API_KEY"));

        Client::validate_api(model, Api::OpenAiCompatResponses, provider, "demo/x").unwrap();
        for refused in [
            Api::OpenAiCompatChatCompletions,
            Api::OpenAiCompatChatCompletionsStream,
        ] {
            let message = Client::validate_api(model, refused, provider, "demo/x")
                .unwrap_err()
                .to_string();
            // Each refusal names the surface it was asked about, so a caller
            // is never told the wrong thing is missing.
            assert!(
                message.contains(refused.as_str()),
                "asked about `{}`: {message}",
                refused.as_str()
            );
        }
    }

    /// Streaming is its own surface, so serving chat completions says nothing
    /// about it. The roster documents that; this is where it holds.
    #[test]
    fn chat_completions_does_not_imply_its_streaming_surface() {
        let err = Client::validate_api(
            demo_model(vec![SupportedApi {
                api: Api::OpenAiCompatChatCompletions,
            }]),
            Api::OpenAiCompatChatCompletionsStream,
            demo_provider("https://api.demo.test", Some("DEMO_API_KEY")),
            "demo/chat",
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("does not serve openai_compat_chat_completions_stream"),
            "{err}"
        );
    }

    /// A provider that names no environment variable has no key source at all,
    /// so the error must say that rather than send someone off to set a
    /// variable nothing reads.
    #[test]
    fn a_provider_with_no_env_api_key_names_the_roster() {
        let err = client(ClientConfig::default())
            .api_key(demo_provider("https://api.demo.test", None))
            .unwrap_err();
        match &err {
            Error::MissingApiKey { provider, env_var } => {
                assert_eq!(*provider, "demo");
                assert_eq!(*env_var, None, "named a variable that does not exist");
            }
            other => panic!("expected MissingApiKey, got {other:?}"),
        }
        let message = err.to_string();
        assert!(
            message.contains("names no environment variable"),
            "{message}"
        );
        assert!(!message.contains("set the"), "{message}");

        // The wire form keeps one code either way. The remedy rides in the
        // message rather than a warpllm-specific `env_var` field, which the
        // OpenAI envelope has no place for.
        let wire: serde_json::Value = serde_json::from_str(&err.to_openai_json()).unwrap();
        assert_eq!(wire["status"], 401);
        // OpenAI's spelling for an unusable key, not warpllm's.
        assert_eq!(wire["error"]["code"], "invalid_api_key");
        assert!(
            wire["error"]["message"]
                .as_str()
                .unwrap()
                .contains("names no environment variable")
        );
    }

    /// What ships upstream is the ENTRY's name, never the string the caller
    /// routed with. That is the whole point of `model:` — a warpllm alias may
    /// differ from the provider's own name for the same model.
    #[tokio::test]
    async fn an_alias_sends_the_entrys_own_name_upstream() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1_700_000_000,
                "model": "demo-chat-20240101",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop"
                }]
            })))
            .mount(&server)
            .await;

        let aliased = Box::leak(Box::new(ModelSpec {
            provider: "demo".into(),
            model: "demo-chat-20240101".into(),
            supported_apis: vec![SupportedApi {
                api: Api::OpenAiCompatChatCompletions,
            }],
            capabilities: Capabilities::blank(),
            deprecation_date: None,
        }));
        let client = client(ClientConfig::default());
        let request = CreateChatCompletionRequest {
            model: "demo/chat".into(),
            messages: vec![ChatCompletionRequestMessage {
                role: "user".into(),
                content: "hi".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        // What `chat_completions` does, minus the routing it already proved:
        // the SPEC's model name is what ingest is handed.
        let normalized =
            openai_compat::api::chat_completions::ingest_request(request, aliased.model());
        openai_compat::api::chat_completions::exchange(
            &normalized,
            &client.http,
            "demo",
            &server.uri(),
            "k",
        )
        .await
        .unwrap();

        let sent = &server.received_requests().await.unwrap()[0];
        let body: serde_json::Value = serde_json::from_slice(&sent.body).unwrap();
        assert_eq!(
            body["model"], "demo-chat-20240101",
            "the caller's routing string shipped instead of the entry's name"
        );
    }

    /// The same path for a concrete entry still ships the ENTRY's name, which
    /// is what lets a routing alias differ from the provider's own name.
    #[tokio::test]
    async fn a_concrete_entry_sends_its_own_name_upstream() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1_700_000_000,
                "model": "gpt-5.6",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop"
                }]
            })))
            .mount(&server)
            .await;

        let client = client(ClientConfig {
            base_url: Some(server.uri()),
            ..Default::default()
        });
        let request = CreateChatCompletionRequest {
            model: "openai/gpt-5.6".into(),
            messages: vec![ChatCompletionRequestMessage {
                role: "user".into(),
                content: "hi".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let (provider, model) = pair_for("openai/gpt-5.6");
        let normalized =
            openai_compat::api::chat_completions::ingest_request(request, model.model());
        openai_compat::api::chat_completions::exchange(
            &normalized,
            &client.http,
            provider.name(),
            client.base_url(provider),
            "k",
        )
        .await
        .unwrap();

        let sent = &server.received_requests().await.unwrap()[0];
        let body: serde_json::Value = serde_json::from_slice(&sent.body).unwrap();
        assert_eq!(body["model"], "gpt-5.6");
    }

    #[test]
    fn base_url_defaults_to_each_providers_api() {
        let client = client(ClientConfig::default());
        assert_eq!(
            client.base_url(pair_for("openai/gpt-5.6").0),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            client.base_url(pair_for("deepseek/deepseek-v4-flash").0),
            "https://api.deepseek.com"
        );
    }

    #[test]
    fn configured_base_url_wins_over_the_default() {
        let client = client(ClientConfig {
            base_url: Some("http://localhost:9999".into()),
            ..Default::default()
        });
        assert_eq!(
            client.base_url(pair_for("openai/gpt-5.6").0),
            "http://localhost:9999"
        );
    }
}
