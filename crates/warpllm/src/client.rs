//! The client: one pooled HTTP connection set, one entrypoint.

use std::time::Duration;

use crate::config::{ClientConfig, DEFAULT_TIMEOUT_SECS};
use crate::credentials::Credentials;
use crate::error::{Error, Result};
use crate::gateway::openai_compat;
use crate::protocol::openai_compat::chat_completions::types::{
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
};
use crate::registry::{self, ModelSpec, ProviderSpec, fetch_model};
use crate::types::Api;

pub struct Client {
    http: reqwest::Client,
    config: ClientConfig,
    credentials: Credentials,
}

/// Everything one validated request needs to reach its model: where to send it,
/// what the model is called upstream, and what to authenticate with.
///
/// A struct rather than a tuple, because both call sites read all three by
/// name and a fourth would otherwise renumber them.
struct ModelDefinition<'a> {
    provider: &'static ProviderSpec,
    model: &'static ModelSpec,
    api_key: &'a str,
}

impl Client {
    /// Resolves the providers this client can authenticate, once.
    ///
    /// Constructing a client still never *requires* credentials: an environment
    /// holding none builds a client that reaches no provider, and each request
    /// says which variable it wanted. What construction does is answer that
    /// question up front, so it can be logged at the moment a caller is set up
    /// to read it rather than discovered one failed request at a time.
    ///
    /// A declaration is checked here for the same reason, one step earlier: a
    /// misspelled provider is wrong the moment it is written, and is not a
    /// condition to discover at request time.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if
    /// [`ClientConfig::providers`] names a provider the roster does not hold.
    pub fn new(config: ClientConfig) -> Result<Self> {
        // Before the transport: a caller's spelling mistake should not be
        // reported after, or masked by, a TLS-init failure.
        Self::validate_declarations(&config)?;
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(
                config.timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS),
            ))
            .build()
            .map_err(|e| Error::Internal(e.to_string()))?;
        let credentials = Credentials::resolve(config.providers.as_ref());
        Ok(Self {
            http,
            config,
            credentials,
        })
    }

    /// Every declared name is one the roster holds.
    ///
    /// A client that accepted a misspelling would go on refusing perfectly
    /// good requests for a provider it believed it did not serve, pointing at
    /// the configuration rather than at the typo in it. It is also what lets
    /// [`Credentials::resolve`] look a declared name up without a fallible
    /// path — nothing downstream has to re-ask this question.
    ///
    /// The message LISTS the roster, because the caller cannot: the registry
    /// is private by design, the list is short, and the failure is nearly
    /// always a spelling. Diagnostic text, not an API.
    fn validate_declarations(config: &ClientConfig) -> Result<()> {
        let Some(declared) = &config.providers else {
            return Ok(());
        };
        // An empty declaration is legal and almost certainly a mistake — it is
        // what a caller building the map from an empty list produces, and it
        // reaches nothing. Refusing it would make `providers: their_list` fail
        // where the equivalent `None` succeeds by accident, so it warns
        // instead, beside the "no keys" warning `resolve` may be about to log.
        if declared.is_empty() {
            tracing::warn!(
                "`providers` is declared and empty, so no request can be routed; \
                 leave it absent to serve warpllm's whole roster"
            );
        }
        // BTreeMap, so the name reported first is the first alphabetically
        // rather than the first a hash seed happened to yield.
        for name in declared.keys() {
            if registry::provider(name).is_none() {
                let mut known: Vec<&str> = registry::providers().map(ProviderSpec::name).collect();
                known.sort_unstable();
                return Err(Error::InvalidInput(format!(
                    "providers: `{name}` is not a provider warpllm serves; the roster has {}",
                    known.join(", ")
                )));
            }
        }
        Ok(())
    }

    /// Serves one OpenAI-compatible chat completion.
    ///
    /// Validation is [`Client::validate`]'s, which is where the order of the
    /// checks and the reason for it are written down. This entrypoint differs
    /// from the streaming one only in the surface it asks about.
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
        let ModelDefinition {
            provider,
            model,
            api_key,
        } = self.validate(&requested_model, Api::OpenAiCompatChatCompletions)?;

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
    /// The same validation as [`Client::chat_completions`], against a DIFFERENT
    /// surface: streaming is its own entry in the roster, so a model that
    /// serves whole replies says nothing about whether it serves streamed ones.
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
        let ModelDefinition {
            provider,
            model,
            api_key,
        } = self.validate(&requested_model, Api::OpenAiCompatChatCompletionsStream)?;

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

    /// The whole validation sequence, in the one order that keeps each refusal
    /// about the thing that is actually wrong.
    ///
    /// FOUR gates, coarse to fine: the roster registers the name, this client
    /// serves the provider, the model serves the surface, and the client holds
    /// a key. The order is the design.
    ///
    /// The roster comes first because a name nothing registers is a typo, and
    /// hearing "declare deepseek" or "set `DEEPSEEK_API_KEY`" for
    /// `deepseek/typo` would send someone after a configuration edit or a
    /// credential they already have.
    ///
    /// The declaration comes before the key, and that is load-bearing rather
    /// than tidy: a provider left undeclared holds no key HERE by construction,
    /// so a key check reached first would report a missing credential for one
    /// the caller has and deliberately withheld.
    ///
    /// One helper rather than the sequence written twice, because the two
    /// entrypoints differ in exactly one argument — and a fifth gate added to
    /// one copy and not the other is a hole nothing would catch.
    fn validate(&self, requested: &str, api: Api) -> Result<ModelDefinition<'_>> {
        let (provider, model) = fetch_model(requested)?;
        self.validate_declared(provider, requested)?;
        Self::validate_api(model, api, provider, requested)?;
        Ok(ModelDefinition {
            provider,
            model,
            api_key: self.api_key(provider)?,
        })
    }

    /// Whether this client serves the routed provider at all.
    ///
    /// An absent [`ClientConfig::providers`] is NO OPINION, not an empty list:
    /// the whole roster stays routable, which is what every client did before
    /// the field existed and what makes declaring purely opt-in. The [`Option`]
    /// is the entire mechanism keeping "I did not say" apart from "I said
    /// none".
    ///
    /// Read from the config per request rather than precomputed onto the
    /// client: it is a lookup over a handful of entries, invisible beside an
    /// HTTP request, and it keeps one source of truth instead of derived state
    /// to hold in step.
    fn validate_declared(&self, provider: &'static ProviderSpec, requested: &str) -> Result<()> {
        match &self.config.providers {
            None => Ok(()),
            Some(declared) if declared.contains_key(provider.name()) => Ok(()),
            Some(_) => Err(Error::ProviderNotDeclared {
                provider: provider.name(),
                requested: requested.to_string(),
            }),
        }
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
    use std::collections::BTreeMap;

    use super::*;
    use crate::config::ProviderConfig;
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

    // ------------------------------------------------------- declared clients

    /// A config declaring these providers, each reading its own variable.
    fn declaring(names: &[&str]) -> ClientConfig {
        ClientConfig {
            providers: Some(
                names
                    .iter()
                    .map(|name| ((*name).to_string(), ProviderConfig::default()))
                    .collect(),
            ),
            ..Default::default()
        }
    }

    /// The load-bearing ordering claim, tested where it is hardest: the key
    /// for the undeclared provider IS in the environment. A client that
    /// checked credentials first would answer "set `DEEPSEEK_API_KEY`" to
    /// someone holding that key and deliberately withholding the provider.
    #[test]
    fn an_undeclared_provider_is_refused_before_the_key_check() {
        with_env(
            &[
                ("OPENAI_API_KEY", Some("sk-openai")),
                ("DEEPSEEK_API_KEY", Some("sk-deepseek")),
            ],
            || {
                let client = Client::new(declaring(&["openai"])).unwrap();
                let err = client
                    .validate(
                        "deepseek/deepseek-v4-flash",
                        Api::OpenAiCompatChatCompletions,
                    )
                    .err()
                    .expect("an undeclared provider is refused");
                match &err {
                    Error::ProviderNotDeclared {
                        provider,
                        requested,
                    } => {
                        assert_eq!(*provider, "deepseek");
                        assert_eq!(requested, "deepseek/deepseek-v4-flash");
                    }
                    other => panic!("expected ProviderNotDeclared, got {other:?}"),
                }
                // And the declared one still routes, so the gate is not one
                // that refuses everything.
                client
                    .validate("openai/gpt-5.6", Api::OpenAiCompatChatCompletions)
                    .unwrap();
            },
        );
    }

    /// The same refusal with no key anywhere. It is still the DECLARATION that
    /// is wrong, not the credential — a caller who declared nothing for
    /// deepseek is not missing a key for it.
    #[test]
    fn an_undeclared_provider_is_refused_even_with_no_key_anywhere() {
        with_env(&[], || {
            let err = Client::new(declaring(&["openai"]))
                .unwrap()
                .validate(
                    "deepseek/deepseek-v4-flash",
                    Api::OpenAiCompatChatCompletions,
                )
                .err()
                .expect("an undeclared provider is refused");
            assert!(matches!(err, Error::ProviderNotDeclared { .. }), "{err:?}");
        });
    }

    /// A name nothing registers is a typo first, whether or not its provider
    /// was declared. Sending someone to edit `providers` for `openai/nope`
    /// would point at the wrong line.
    #[test]
    fn an_unregistered_model_is_a_typo_before_it_is_a_declaration_problem() {
        with_env(&[("OPENAI_API_KEY", Some("sk-openai"))], || {
            let client = Client::new(declaring(&["openai"])).unwrap();
            for typo in ["openai/nope", "deepseek/nope"] {
                let err = client
                    .validate(typo, Api::OpenAiCompatChatCompletions)
                    .err()
                    .expect("validation refuses it");
                assert!(
                    matches!(&err, Error::InvalidModel { given } if given == typo),
                    "`{typo}`: {err:?}"
                );
            }
        });
    }

    /// Declaring a provider does not conjure its key. The remedy here really
    /// is the variable, and the message still names it.
    #[test]
    fn a_declared_provider_with_no_key_still_names_its_variable() {
        with_env(&[], || {
            let err = Client::new(declaring(&["openai"]))
                .unwrap()
                .validate("openai/gpt-5.6", Api::OpenAiCompatChatCompletions)
                .err()
                .expect("validation refuses it");
            assert!(
                matches!(
                    &err,
                    Error::MissingApiKey {
                        env_var: Some("OPENAI_API_KEY"),
                        ..
                    }
                ),
                "{err:?}"
            );
        });
    }

    /// An inline key routes a provider whose variable is absent — the point of
    /// carrying one at all.
    #[test]
    fn an_inline_key_admits_a_provider_the_environment_cannot() {
        with_env(&[], || {
            let config = ClientConfig {
                providers: Some(BTreeMap::from([(
                    "openai".to_string(),
                    ProviderConfig {
                        api_key: Some("sk-inline".into()),
                    },
                )])),
                ..Default::default()
            };
            let client = Client::new(config).unwrap();
            let admitted = client
                .validate("openai/gpt-5.6", Api::OpenAiCompatChatCompletions)
                .unwrap();
            assert_eq!(admitted.api_key, "sk-inline");
        });
    }

    /// The compatibility claim at the routing gate: saying nothing leaves the
    /// whole roster reachable, exactly as before the field existed.
    #[test]
    fn an_absent_declaration_routes_the_whole_roster() {
        with_env(
            &[
                ("OPENAI_API_KEY", Some("sk-openai")),
                ("DEEPSEEK_API_KEY", Some("sk-deepseek")),
            ],
            || {
                let client = Client::new(ClientConfig::default()).unwrap();
                for model in ["openai/gpt-5.6", "deepseek/deepseek-v4-flash"] {
                    client
                        .validate(model, Api::OpenAiCompatChatCompletions)
                        .unwrap();
                }
            },
        );
    }

    /// And declaring none reaches none — the other half of the distinction the
    /// `Option` exists to keep.
    #[test]
    fn an_empty_declaration_routes_nothing() {
        with_env(&[("OPENAI_API_KEY", Some("sk-openai"))], || {
            let err = Client::new(declaring(&[]))
                .unwrap()
                .validate("openai/gpt-5.6", Api::OpenAiCompatChatCompletions)
                .err()
                .expect("validation refuses it");
            assert!(matches!(err, Error::ProviderNotDeclared { .. }), "{err:?}");
        });
    }

    /// A misspelling is caught at construction, not at the request that
    /// happened to route there — and the message hands back the roster,
    /// because a caller cannot list it themselves.
    #[test]
    fn an_unknown_declared_name_fails_at_construction() {
        with_env(&[], || {
            let err = Client::new(declaring(&["openia"]))
                .err()
                .expect("a misspelled provider is refused");
            let message = err.to_string();
            assert!(message.contains("openia"), "{message}");
            assert!(
                message.contains("deepseek, kimi, openai, openrouter"),
                "{message}"
            );
            assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
        });
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
            messages: vec![ChatCompletionRequestMessage::new("user", "hi")],
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
            messages: vec![ChatCompletionRequestMessage::new("user", "hi")],
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
