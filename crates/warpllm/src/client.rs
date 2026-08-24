//! The client: one pooled HTTP connection set, one roster, one entrypoint.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

use crate::auth::Authenticator;
use crate::config::{ClientConfig, DEFAULT_TIMEOUT_SECS};
use crate::credentials::Credentials;
use crate::error::{Error, Result};
use crate::gateway::{anthropic, openai_compat};
use crate::protocol::openai_compat::chat_completions::types::{
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
};
use crate::registry::{self, ModelSpec, ProviderSpec, Registry};
use crate::types::{Api, Protocol};

/// Where a roster file is named when [`ClientConfig::specs_path`] does not name
/// one. The last resort before the shipped roster alone.
///
/// It is read here rather than by the server, so that pointing a container at a
/// mounted roster works for every surface at once — the gateway, the Rust
/// client, and both bindings — with no per-surface plumbing. Symmetric with
/// keys, which are read from the environment at this same moment.
pub(crate) const SPECS_ENV_VAR: &str = "WARPLLM_SPECS";

pub struct Client {
    http: reqwest::Client,
    config: ClientConfig,
    credentials: Credentials,
    /// This client's roster, and its alone. Shared with every other client
    /// that was given no file of its own, since they all route against the
    /// shipped tables.
    registry: Arc<Registry>,
}

/// Everything one validated request needs to reach its model: where to send it,
/// what the model is called upstream, what to authenticate with, and which
/// protocol to speak on the way out.
///
/// A struct rather than a tuple, because both call sites read all four by
/// name and a fifth would otherwise renumber them.
struct ModelDefinition<'a> {
    provider: &'a ProviderSpec,
    model: &'a ModelSpec,
    /// `None` where the roster says `auth: none`: the host takes no credential,
    /// so the request goes out with no `Authorization` header rather than with
    /// an empty one.
    auth: Option<&'a Authenticator>,
    egress: Egress,
}

/// Which protocol a routed request is spoken in UPSTREAM.
///
/// Not the protocol warpllm was called in — that is fixed by the entrypoint,
/// and for both entrypoints here it is openai_compat by signature. This is the
/// other half, and it is a property of the routed MODEL:
/// `anthropic/claude-opus-5` serves `anthropic_messages` and nothing else, so a
/// chat-completions request for it is translated on the way out and its reply
/// on the way back.
///
/// A two-variant enum rather than dispatching on [`Api`] at the call site.
/// `Api` has five variants and only two can reach either entrypoint, so a match
/// on it would carry arms for cases the validation just ruled out — and the
/// admission list already knows which module serves which surface. Pairing the
/// two in [`Client::WHOLE_REPLY`] and [`Client::STREAMED`] makes those arms
/// unrepresentable rather than unreachable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Egress {
    OpenAiCompat,
    Anthropic,
}

/// Refuses a request whose meaning the routed protocol has no way to carry.
///
/// SILENCE is the failure this exists to prevent. warpllm's contract everywhere
/// else is passthrough — forward what the caller wrote, let the provider judge
/// it — and that contract quietly loses its second half on a translated route:
/// a field the provider never receives is a field the provider cannot reject.
/// `n: 2` would come back as one choice, indistinguishable from a model that
/// ignored the request.
///
/// It sits at DISPATCH rather than in the Anthropic renderer because of where
/// the value lives. `n` rides `ext["openai_compat"]`, and
/// [`Protocol::may_read`] forbids that renderer from ever seeing it — by
/// design, since another protocol's bag is exactly what a renderer must not
/// read. Dispatch is the one layer above both.
///
/// Only what CHANGES the answer is refused, and only when it cannot be
/// translated. `top_k` and `parallel_tool_calls` have equivalents on the far
/// side and are promoted at ingest and rendered, not refused.
fn reject_untranslatable(request: &crate::gateway::types::ChatRequest) -> Result<()> {
    let Some(bag) = request.ext.get(Protocol::OpenAiCompat.as_str()) else {
        return Ok(());
    };
    for (field, means_it) in UNTRANSLATABLE {
        if bag.get(field).is_some_and(means_it) {
            return Err(Error::InvalidInput(format!(
                "`{field}` has no equivalent on Anthropic's Messages API, which is what \
                 serves this model; warpllm refuses it rather than answering as though \
                 it were never written"
            )));
        }
    }
    Ok(())
}

/// The fields chat completions can state, Anthropic cannot, and that change the
/// answer or its shape — each paired with the test for whether the caller
/// actually MEANT it.
///
/// The predicate is what keeps the list from being blunt. Every one of these
/// has a value meaning "no opinion" — zero penalties, `logprobs: false`, an
/// empty bias map, one choice — and those are exactly what an SDK fills in by
/// default. Rejecting on the KEY would turn away requests that ask for nothing,
/// which is most of them.
///
/// A DENY list and not an allow list, deliberately. An allow list would also
/// refuse every harmless field a caller or a future SDK sends (`user`, `store`,
/// a provider's own extension), and warpllm's whole contract is that those ride
/// through untouched. The cost is that this list has to be maintained: a field
/// nobody adds here is still dropped in silence, which is the same failure this
/// function exists to end.
type MeansIt = fn(&Value) -> bool;
const UNTRANSLATABLE: &[(&str, MeansIt)] = &[
    ("n", |value| value.as_u64().is_some_and(|n| n > 1)),
    ("frequency_penalty", nonzero),
    ("presence_penalty", nonzero),
    ("logprobs", |value| value == &Value::Bool(true)),
    ("top_logprobs", |value| {
        value.as_u64().is_some_and(|count| count > 0)
    }),
    ("logit_bias", |value| {
        value.as_object().is_some_and(|bias| !bias.is_empty())
    }),
    // Unlike the rest, ANY value means it: `seed` exists only to ask for
    // reproducibility, and a caller who believes they have it and does not is
    // worse off than one told plainly that they cannot.
    ("seed", Value::is_number),
];

fn nonzero(value: &Value) -> bool {
    value.as_f64().is_some_and(|penalty| penalty != 0.0)
}

impl Client {
    /// Reads the roster and the environment once, and keeps what it found.
    ///
    /// Constructing a client still never *requires* credentials: an environment
    /// holding none builds a client that reaches no provider, and each request
    /// says which variable it wanted. What construction does is answer that
    /// question up front, so it can be logged at the moment a caller is set up
    /// to read it rather than discovered one failed request at a time.
    ///
    /// A roster file is the other half of the same idea, and the reason this
    /// can now fail: a malformed one is reported here, where the caller is
    /// holding the path and can go fix it, rather than as a closed registry
    /// refusing a request hours later.
    ///
    /// A declaration is checked here too, and it is checked against the roster
    /// this client actually loaded — a caller who wrote their own `local`
    /// provider may declare it, and validating against the shipped list would
    /// refuse them their own file.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidRoster`] if a roster file was named and cannot be used.
    /// [`Error::InvalidInput`] if [`ClientConfig::providers`] names a provider
    /// this client's roster does not hold.
    /// [`Error::Internal`] if the HTTP client will not build.
    pub fn new(config: ClientConfig) -> Result<Self> {
        // Both before the transport: a mistake in the caller's roster or in
        // their declaration should not be reported after, or masked by, a
        // TLS-init failure. The roster comes first of the two because the
        // declaration is checked against it.
        let specs_path = Self::specs_path(&config);
        // Both are global redirections of where a request goes, and the first
        // wins over the second — including over the local address that was the
        // whole reason for writing the roster. Nobody means that.
        if let (Some(base_url), Some(path)) = (&config.base_url, &specs_path) {
            tracing::warn!(
                base_url,
                roster = %path.display(),
                "base_url overrides EVERY provider, the roster file's own \
                 included, so nothing will reach the addresses it names"
            );
        }
        let registry = registry::load_for_client(specs_path.as_deref())?;
        Self::validate_declarations(&config, &registry)?;
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(
                config.timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS),
            ))
            .build()
            .map_err(|e| Error::Internal(e.to_string()))?;
        let credentials = Credentials::resolve(&registry, config.providers.as_ref());
        Ok(Self {
            http,
            credentials,
            config,
            registry,
        })
    }

    /// The roster file to load, if any: what the caller configured, then
    /// [`SPECS_ENV_VAR`], then nothing.
    ///
    /// An empty environment variable counts as unset, matching the reading
    /// `Credentials` gives an empty API key — a variable that is exported and
    /// blank is a configuration mistake, not an answer.
    fn specs_path(config: &ClientConfig) -> Option<PathBuf> {
        config.specs_path.clone().or_else(|| {
            std::env::var(SPECS_ENV_VAR)
                .ok()
                .filter(|path| !path.is_empty())
                .map(PathBuf::from)
        })
    }

    /// What THIS client knows about a `model_str`: the provider that serves it,
    /// and the model itself.
    ///
    /// The per-client counterpart of [`fetch_model`](crate::fetch_model), and
    /// the one to ask about a client built with a roster of its own — that free
    /// function answers about the roster warpllm ships, which is a different
    /// question and stops being the same answer the moment a file is loaded.
    ///
    /// Matching is unchanged and exact: the key matches its own entry or
    /// nothing at all. A roster file adds names; it does not add guessing.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidModel`] if this client's roster registers nothing for
    /// `model_str`.
    pub fn fetch_model(&self, model_str: &str) -> Result<(&ProviderSpec, &ModelSpec)> {
        registry::resolve(&self.registry, model_str).ok_or_else(|| Error::InvalidModel {
            given: model_str.to_string(),
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
    ///
    /// `registry` is the MERGED roster, so a caller may declare a provider only
    /// their own file names. Checking the shipped list instead would refuse
    /// somebody their own `local`, which is the pair of features working
    /// against each other rather than together.
    fn validate_declarations(config: &ClientConfig, registry: &Registry) -> Result<()> {
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
            if registry::provider(registry, name).is_none() {
                let mut known: Vec<&str> = registry::providers(registry)
                    .map(ProviderSpec::name)
                    .collect();
                known.sort_unstable();
                return Err(Error::InvalidInput(format!(
                    "providers: `{name}` is not a provider warpllm serves; the roster has {}",
                    known.join(", ")
                )));
            }
        }
        Ok(())
    }

    /// The surfaces a whole-reply request may be served by, each with the
    /// protocol that serves it.
    ///
    /// Order is preference, and it only matters for a model that lists both —
    /// nothing on the roster does, and a model that did would be one host
    /// offering the same completion two ways. Chat completions first, because
    /// it is the protocol the caller is already speaking and needs no
    /// translation.
    const WHOLE_REPLY: &'static [(Api, Egress)] = &[
        (Api::OpenAiCompatChatCompletions, Egress::OpenAiCompat),
        (Api::AnthropicMessages, Egress::Anthropic),
    ];

    /// The streamed counterpart of [`Client::WHOLE_REPLY`]. A separate list
    /// rather than derived from it, because streaming is its own roster entry:
    /// a model serving whole replies says nothing about whether it serves
    /// streamed ones, and that is true per protocol.
    const STREAMED: &'static [(Api, Egress)] = &[
        (Api::OpenAiCompatChatCompletionsStream, Egress::OpenAiCompat),
        (Api::AnthropicMessagesStream, Egress::Anthropic),
    ];

    /// Serves one OpenAI-compatible chat completion.
    ///
    /// Validation is [`Client::validate`]'s, which is where the order of the
    /// checks and the reason for it are written down. This entrypoint differs
    /// from the streaming one only in the surfaces it admits.
    ///
    /// The request shape is OpenAI's whichever provider serves it. A model that
    /// serves Anthropic's `/messages` and nothing else is reached by
    /// translating on the way out and back, and the caller sees a
    /// chat-completions reply either way — that is the whole point of the
    /// gateway form in between.
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
            auth,
            egress,
        } = self.validate(&requested_model, Self::WHOLE_REPLY)?;

        // Ingest answers to the protocol warpllm was CALLED with, which is
        // openai_compat and only ever will be for this entrypoint. The ENTRY's
        // model name goes in, not the caller's string: they differ whenever
        // warpllm's routing alias differs from the provider's own name.
        let normalized =
            openai_compat::api::chat_completions::ingest_request(request, model.model());
        // Ingress by entrypoint, EGRESS by the routed model's surface. Both
        // arms take and return gateway types, which is what keeps this a
        // two-line choice rather than two request paths.
        let response = match egress {
            Egress::OpenAiCompat => {
                openai_compat::api::chat_completions::exchange(
                    &normalized,
                    &self.http,
                    provider.name(),
                    self.base_url(provider),
                    auth,
                )
                .await?
            }
            Egress::Anthropic => {
                reject_untranslatable(&normalized)?;
                anthropic::api::messages::exchange(
                    &normalized,
                    &self.http,
                    provider.name(),
                    self.base_url(provider),
                    auth,
                    // Anthropic REQUIRES a `max_tokens` and the gateway form's
                    // is optional, so the roster's ceiling is the fallback. A
                    // model documenting none and a caller naming none is a
                    // refusal, not an invented default — see
                    // `anthropic::…::request::resolve_max_tokens`.
                    model.capabilities().max_output_tokens(),
                )
                .await?
            }
        };
        let mut completion =
            openai_compat::api::chat_completions::render_response(&response, provider.name());
        // Echo the caller's provider-prefixed string, not the upstream echo.
        completion.model = requested_model;
        Ok(completion)
    }

    /// Serves one OpenAI-compatible chat completion as a stream of chunks.
    ///
    /// The same validation as [`Client::chat_completions`], against DIFFERENT
    /// surfaces: streaming is its own entry in the roster, so a model that
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
            auth,
            egress,
        } = self.validate(&requested_model, Self::STREAMED)?;

        let normalized =
            openai_compat::api::chat_completions::ingest_request(request, model.model());
        let read_timeout = self
            .config
            .stream_read_timeout_secs
            .map(Duration::from_secs);
        let chunks = match egress {
            Egress::OpenAiCompat => Chunks::OpenAiCompat(Box::new(
                openai_compat::api::chat_completions::exchange_stream(
                    &normalized,
                    &self.http,
                    provider.name(),
                    self.base_url(provider),
                    auth,
                    read_timeout,
                )
                .await?,
            )),
            Egress::Anthropic => {
                reject_untranslatable(&normalized)?;
                Chunks::Anthropic(Box::new(
                    anthropic::api::messages::exchange_stream(
                        &normalized,
                        &self.http,
                        provider.name(),
                        self.base_url(provider),
                        auth,
                        model.capabilities().max_output_tokens(),
                        read_timeout,
                    )
                    .await?,
                ))
            }
        };
        Ok(ChatCompletionStream {
            chunks,
            provider: provider.name(),
            model: requested_model,
            ordinals: openai_compat::api::chat_completions::ToolCallOrdinals::default(),
        })
    }

    /// The whole validation sequence, in the one order that keeps each refusal
    /// about the thing that is actually wrong.
    ///
    /// FOUR gates, coarse to fine: this client's roster registers the name, the
    /// client serves the provider, the model serves the surface, and the client
    /// holds a credential or the roster says none is wanted. The order is the
    /// design.
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
    fn validate(&self, requested: &str, admitted: &[(Api, Egress)]) -> Result<ModelDefinition<'_>> {
        let (provider, model) = self.fetch_model(requested)?;
        self.validate_declared(provider, requested)?;
        let egress = Self::validate_api(model, admitted, provider, requested)?;
        Ok(ModelDefinition {
            provider,
            model,
            auth: self.authenticator(provider)?,
            egress,
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
    fn validate_declared(&self, provider: &ProviderSpec, requested: &str) -> Result<()> {
        match &self.config.providers {
            None => Ok(()),
            Some(declared) if declared.contains_key(provider.name()) => Ok(()),
            Some(_) => Err(Error::ProviderNotDeclared {
                provider: provider.name(),
                requested: requested.to_string(),
            }),
        }
    }

    /// The routed provider's credential, from the snapshot this client took of
    /// the environment when it was built.
    ///
    /// Three answers, not two, matching the three states a roster entry can be
    /// in. `Ok(Some(auth))` is the ordinary case. `Ok(None)` is a provider that
    /// declared `auth: none` — a host that takes no credential, which is what a
    /// self-hosted box on a private network usually is; no `Authorization`
    /// header is sent for it at all.
    ///
    /// `Err` is everything else: a variable that was unset when this client was
    /// built, or an entry naming no way to authenticate. A miss is not "the
    /// variable is unset now" but "it was unset then", and the error still names
    /// the variable to set, because that is the remedy either way. A provider
    /// with no `env_api_key` has no key source at all, so the error names the
    /// roster rather than a variable nothing reads.
    ///
    /// A resolved credential is consulted BEFORE `auth: none`, so a caller who
    /// declared an inline key for their own provider still sends it. The roster
    /// says what the host wants in general; a caller who put a token in front of
    /// their box has said something more specific, and that is the same
    /// precedence [`Credentials::key_for`] gives inline keys over the
    /// environment.
    fn authenticator(&self, provider: &ProviderSpec) -> Result<Option<&Authenticator>> {
        if let Some(auth) = self.credentials.get(provider.name()) {
            return Ok(Some(auth));
        }
        if provider.unauthenticated() {
            return Ok(None);
        }
        Err(Error::MissingApiKey {
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
    /// `model` and `admitted` are the check; `provider` and `requested` only
    /// build the message. Split out from [`Client::chat_completions`] so the
    /// refusal can be tested at all: every model the roster ships today serves
    /// one of the admitted surfaces, so the failing branch is otherwise
    /// unreachable from the public entrypoint.
    ///
    /// Takes a LIST and returns what matched, which is the whole of the
    /// dispatch decision: an entrypoint that serves two protocols has to say
    /// which surfaces it will take, and then the answer to "which one did"
    /// exists in exactly one place. Returning [`Egress`] rather than the [`Api`]
    /// is what keeps the caller's match down to the protocols that can actually
    /// have matched.
    ///
    /// The refusal names EVERY surface tried, in the roster's own spelling. A
    /// model serving only `openai_compat_responses` is refused by both
    /// entrypoints, and hearing about only one of them would send someone
    /// looking for a roster line that is already right.
    fn validate_api(
        model: &ModelSpec,
        admitted: &[(Api, Egress)],
        provider: &ProviderSpec,
        requested: &str,
    ) -> Result<Egress> {
        for &(api, egress) in admitted {
            if model.supports_api(api) {
                return Ok(egress);
            }
        }
        // The roster's own spelling for each surface, and the provider because
        // the roster is where what-is-served is recorded — between them they
        // name the line a reader would go and fix.
        let tried: Vec<&str> = admitted.iter().map(|(api, _)| api.as_str()).collect();
        Err(Error::InvalidInput(format!(
            "{}: {requested} serves none of {}",
            provider.name(),
            tried.join(", ")
        )))
    }

    /// A configured `base_url` overrides the provider default (proxies,
    /// tests); otherwise each provider talks to its own API.
    ///
    /// One lifetime for both, spelled out: the answer borrows from whichever
    /// won, and elision would otherwise take it from `&self` alone.
    fn base_url<'a>(&'a self, provider: &'a ProviderSpec) -> &'a str {
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
    chunks: Chunks,
    provider: &'static str,
    /// The caller's provider-prefixed string, echoed onto every chunk in place
    /// of the upstream's own — the streaming counterpart of the one
    /// [`Client::chat_completions`] performs on a whole reply.
    model: String,
    /// A stream's tool-call numbering, which only a scope that outlives a chunk
    /// can hold. See
    /// [`ToolCallOrdinals`](openai_compat::api::chat_completions::ToolCallOrdinals)
    /// for what it is correcting and why it is an identity on a same-protocol
    /// stream.
    ordinals: openai_compat::api::chat_completions::ToolCallOrdinals,
}

/// The protocol-specific half of an open stream.
///
/// The enum is here rather than behind a trait for the reason the crate gives
/// everywhere else it makes this choice: the set of protocols warpllm speaks
/// grows an issue at a time, and a closed match is what makes the next one fail
/// to compile until every site handles it.
///
/// BOTH arms are boxed, not just the larger one. They are 248 and 464 bytes, so
/// an unboxed enum is the size of its biggest member and every stream pays for
/// the widest protocol; boxing only the larger one just inverts which arm clippy
/// names. One allocation per STREAM, which lives for as many chunks as the reply
/// has, against copying a half-kilobyte enum on every move.
#[derive(Debug)]
enum Chunks {
    OpenAiCompat(Box<openai_compat::api::chat_completions::ChatChunkStream>),
    Anthropic(Box<anthropic::api::messages::ChatChunkStream>),
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
        // Gateway chunks in, whichever protocol produced them; ONE renderer
        // out, because the caller asked in chat completions and gets chat
        // completions back.
        let chunk = match &mut self.chunks {
            Chunks::OpenAiCompat(chunks) => chunks.next().await?,
            Chunks::Anthropic(chunks) => chunks.next().await?,
        };
        Some(chunk.map(|chunk| {
            let mut rendered = openai_compat::api::chat_completions::render_chunk(
                &chunk,
                self.provider,
                &mut self.ordinals,
            );
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
    use crate::registry::{Capabilities, Credential, ModelSpec, SupportedApi};

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
        crate::fetch_model(model).unwrap()
    }

    /// A provider spec built by hand, for the cases the shipped roster cannot
    /// express. Owned outright — the client borrows its specs from its own
    /// roster now, so nothing here has to be leaked to satisfy a lifetime.
    fn demo_provider(base_url: &str, credential: Credential) -> ProviderSpec {
        ProviderSpec {
            name: "demo",
            base_url: base_url.into(),
            credential,
        }
    }

    /// Takes its surfaces so a caller can express the case the roster cannot:
    /// a model under a chat-serving host that does not itself serve chat.
    fn demo_model(supported_apis: Vec<SupportedApi>) -> ModelSpec {
        ModelSpec {
            provider: "demo".into(),
            model: "demo-embed".into(),
            supported_apis,
            capabilities: Capabilities::blank(),
            deprecation_date: None,
        }
    }

    /// A provider standing in for any host, so the surface tests below read as
    /// being about surfaces.
    fn demo_host() -> ProviderSpec {
        demo_provider("https://api.demo.test", Credential::EnvVar("DEMO_API_KEY"))
    }

    /// The gate the whole model-level `supported_apis` split exists for. This
    /// model sits under a perfectly ordinary chat-serving provider and does
    /// not serve chat itself, which only the model's own list can say.
    #[test]
    fn a_model_that_serves_none_of_the_admitted_surfaces_is_refused() {
        let err = Client::validate_api(
            &demo_model(vec![SupportedApi {
                api: Api::OpenAiCompatResponses,
            }]),
            Client::WHOLE_REPLY,
            &demo_host(),
            "demo/embed",
        )
        .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("demo/embed"), "{message}");
        // EVERY surface tried, in the roster's spelling rather than the
        // variant's — these are the strings a reader greps `specs.yaml` for,
        // and hearing about only one of two would send them after a line that
        // is already right.
        for tried in ["openai_compat_chat_completions", "anthropic_messages"] {
            assert!(message.contains(tried), "missing `{tried}`: {message}");
        }

        // A 400: the caller asked for something this model cannot do, which is
        // theirs to fix, not the provider's to fail.
        let wire: serde_json::Value = serde_json::from_str(&err.to_openai_json()).unwrap();
        assert_eq!(wire["status"], 400);
    }

    /// The other side of the same gate, and the dispatch decision itself: which
    /// surface a model serves is what picks the protocol it is reached over.
    ///
    /// Both directions in one test, because the risk is not that either answer
    /// is wrong on its own — it is that the match returns a constant. A version
    /// that always answered `OpenAiCompat` passes the first assertion and fails
    /// the second.
    #[test]
    fn the_matched_surface_picks_the_protocol_spoken_upstream() {
        for (surface, expected) in [
            (Api::OpenAiCompatChatCompletions, Egress::OpenAiCompat),
            (Api::AnthropicMessages, Egress::Anthropic),
        ] {
            let egress = Client::validate_api(
                &demo_model(vec![
                    SupportedApi { api: surface },
                    // A second, unadmitted surface beside it changes nothing:
                    // each listing is its own claim.
                    SupportedApi {
                        api: Api::OpenAiCompatResponses,
                    },
                ]),
                Client::WHOLE_REPLY,
                &demo_host(),
                "demo/chat",
            )
            .unwrap();
            assert_eq!(egress, expected, "{}", surface.as_str());
        }
    }

    /// The point of passing the surfaces in: one model, and the answer depends
    /// on which list is asked about. A check that ignored its argument would
    /// pass both of the tests above and fail here.
    #[test]
    fn the_answer_depends_on_which_surfaces_are_admitted() {
        let model = &demo_model(vec![SupportedApi {
            api: Api::AnthropicMessages,
        }]);
        let provider = &demo_host();

        assert_eq!(
            Client::validate_api(model, Client::WHOLE_REPLY, provider, "demo/x").unwrap(),
            Egress::Anthropic
        );
        // The same model, against the STREAMED list: it serves the whole-reply
        // surface of this protocol and not the streamed one, so admitting it
        // here would route a stream at an endpoint the roster never claimed.
        let message = Client::validate_api(model, Client::STREAMED, provider, "demo/x")
            .unwrap_err()
            .to_string();
        for tried in [
            "openai_compat_chat_completions_stream",
            "anthropic_messages_stream",
        ] {
            assert!(message.contains(tried), "missing `{tried}`: {message}");
        }
    }

    /// Streaming is its own surface, so serving whole replies says nothing
    /// about it. The roster documents that; this is where it holds — and it
    /// holds per PROTOCOL, which is why both are swept.
    #[test]
    fn a_whole_reply_surface_does_not_imply_its_streaming_one() {
        for (whole, streamed) in [
            (
                Api::OpenAiCompatChatCompletions,
                "openai_compat_chat_completions_stream",
            ),
            (Api::AnthropicMessages, "anthropic_messages_stream"),
        ] {
            let err = Client::validate_api(
                &demo_model(vec![SupportedApi { api: whole }]),
                Client::STREAMED,
                &demo_host(),
                "demo/chat",
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains(streamed), "{err}");
        }
    }

    /// The two admission lists must not overlap, and neither may admit a
    /// surface warpllm cannot serve on it. A streamed surface in
    /// [`Client::WHOLE_REPLY`] would have the non-streaming entrypoint open a
    /// stream it has no return type for.
    #[test]
    fn the_admission_lists_are_disjoint_and_correctly_halved() {
        let whole: Vec<&str> = Client::WHOLE_REPLY
            .iter()
            .map(|(api, _)| api.as_str())
            .collect();
        let streamed: Vec<&str> = Client::STREAMED
            .iter()
            .map(|(api, _)| api.as_str())
            .collect();
        for name in &whole {
            assert!(!name.ends_with("_stream"), "`{name}` is a streamed surface");
            assert!(!streamed.contains(name), "`{name}` is in both lists");
        }
        for name in &streamed {
            assert!(name.ends_with("_stream"), "`{name}` is not streamed");
        }
        // One entry per protocol in each, so a protocol cannot be reachable
        // for whole replies and silently unreachable for streams.
        assert_eq!(whole.len(), streamed.len());
    }

    /// A provider that names no environment variable has no key source at all,
    /// so the error must say that rather than send someone off to set a
    /// variable nothing reads.
    #[test]
    fn a_provider_with_no_env_api_key_names_the_roster() {
        let err = client(ClientConfig::default())
            .authenticator(&demo_provider(
                "https://api.demo.test",
                Credential::Unavailable,
            ))
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
                    .validate("deepseek/deepseek-v4-flash", Client::WHOLE_REPLY)
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
                    .validate("openai/gpt-5.6", Client::WHOLE_REPLY)
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
                .validate("deepseek/deepseek-v4-flash", Client::WHOLE_REPLY)
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
                    .validate(typo, Client::WHOLE_REPLY)
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
                .validate("openai/gpt-5.6", Client::WHOLE_REPLY)
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
    #[tokio::test]
    async fn an_inline_key_admits_a_provider_the_environment_cannot() {
        // Built inside the lock, asserted outside it: `with_env` takes a
        // synchronous closure, and reading what a credential puts on a request
        // is now an await.
        let client = with_env(&[], || {
            let config = ClientConfig {
                providers: Some(BTreeMap::from([(
                    "openai".to_string(),
                    ProviderConfig {
                        api_key: Some("sk-inline".into()),
                    },
                )])),
                ..Default::default()
            };
            Client::new(config).unwrap()
        });
        let admitted = client
            .validate("openai/gpt-5.6", Client::WHOLE_REPLY)
            .unwrap();
        assert_eq!(
            crate::auth::testing::applied(
                admitted
                    .auth
                    .expect("openai reads a variable, so it is never `auth: none`"),
                "authorization",
            )
            .await
            .as_deref(),
            Some("Bearer sk-inline")
        );
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
                    client.validate(model, Client::WHOLE_REPLY).unwrap();
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
                .validate("openai/gpt-5.6", Client::WHOLE_REPLY)
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
                message.contains("deepseek, kimi, mistral, openai, opencode, openrouter"),
                "{message}"
            );
            assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
        });
    }

    /// The other side of the same coin, and the whole reason `auth: none` is a
    /// separate spelling: a provider that declares it takes no credential is
    /// ADMITTED with none, where the test above is refused.
    ///
    /// Both run under an empty environment, so nothing but the roster entry
    /// separates them.
    #[test]
    fn a_provider_that_takes_no_credential_needs_no_key() {
        let client = client(ClientConfig::default());
        let auth = client
            .authenticator(&demo_provider(
                "http://localhost:8000/v1",
                Credential::NotRequired,
            ))
            .expect("a host that wants no credential is not a missing key");
        assert!(auth.is_none(), "an unauthenticated host is sent no token");
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
            Some(&Authenticator::bearer("k".into())),
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
            Some(&Authenticator::bearer("k".into())),
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
