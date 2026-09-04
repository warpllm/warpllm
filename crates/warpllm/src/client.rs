//! The client: one pooled HTTP connection set, one roster, one entrypoint.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::auth::Authenticator;
use crate::config::{ClientConfig, DEFAULT_TIMEOUT_SECS};
use crate::credentials::Credentials;
use crate::error::{Error, Result};
use crate::gateway::openai_compat;
use crate::protocol::openai_compat::chat_completions::types::{
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
};
use crate::registry::{self, ModelSpec, ProviderSpec, Registry};
use crate::types::Api;

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
/// what the model is called upstream, and what to authenticate with.
///
/// A struct rather than a tuple, because both call sites read all three by
/// name and a fourth would otherwise renumber them.
struct ModelDefinition<'a> {
    provider: &'a ProviderSpec,
    model: &'a ModelSpec,
    /// `None` where the roster says `auth: none`: the host takes no credential,
    /// so the request goes out with no `Authorization` header rather than with
    /// an empty one.
    auth: Option<&'a Authenticator>,
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
            return Err(Error::InvalidInput(
                "stream: true asks for chunks; call chat_completions_stream".into(),
            ));
        }
        let candidates = build_candidates(&request)?;

        // Validate ALL candidates before any exchange. An unroutable
        // candidate fails the whole request — it is not skipped. Each of
        // those four failures is a caller mistake, not a transient upstream
        // condition, and a typo in candidate 3 believes they have three-way
        // redundancy and has two — that is worth a refusal at admission,
        // where the message can name the candidate and the gate it failed.
        let validated: Vec<(String, ModelDefinition<'_>)> = candidates
            .list
            .iter()
            .map(|c| {
                self.validate(c, Api::OpenAiCompatChatCompletions)
                    .map(|def| (c.clone(), def))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut failover = Failover::new(
            validated.iter().map(|(c, _)| c.clone()).collect(),
            self.config.timeout_secs,
            candidates.requested_models,
        );

        // A single candidate gets NO chain deadline. Its guarantee is the
        // reqwest per-request timeout, exactly as before this feature
        // existed: a slow upstream reports Network, never a gateway-claimed
        // timeout that might surface as the caller's fault. The chain
        // deadline exists for failover to bound the TOTAL across candidates.
        let multi = failover.candidates.len() > 1;
        // Snapshot BEFORE `run` borrows `failover` mutably. The chain clock
        // starts at construction; microseconds of setup don't change the
        // deadline meaningfully.
        let remaining = failover.remaining();

        let run = async {
            loop {
                let (candidate, def) = match failover.next() {
                    Some(c) => {
                        // We validated above, so look up the pre-validated def.
                        let def = validated
                            .iter()
                            .find(|(name, _)| name == c)
                            .map(|(_, def)| def)
                            .expect("validated list matches failover candidates");
                        (c.to_string(), def)
                    }
                    None => break Err(failover.exhausted()),
                };

                let normalized = openai_compat::api::chat_completions::ingest_request(
                    request.clone(),
                    def.model.model(),
                );

                match openai_compat::api::chat_completions::exchange(
                    &normalized,
                    &self.http,
                    def.provider.name(),
                    self.base_url(def.provider),
                    def.auth,
                )
                .await
                {
                    Ok(response) => {
                        let mut completion = openai_compat::api::chat_completions::render_response(
                            &response,
                            def.provider.name(),
                        );
                        // Echo the candidate that served, not the caller's
                        // original model string.
                        completion.model = candidate;
                        tracing::info!(
                            candidate = %completion.model,
                            "failover candidate serving"
                        );
                        break Ok(completion);
                    }
                    Err(e) => {
                        if is_retryable(&e) {
                            tracing::warn!(
                                candidate = %candidate,
                                error = %e,
                                "failover candidate failed, trying next"
                            );
                            failover.record_failure(candidate, e);
                            continue;
                        }
                        break Err(e);
                    }
                }
            }
        };

        if multi {
            match tokio::time::timeout(remaining, run).await {
                Ok(r) => r,
                Err(_elapsed) => Err(failover.deadline()),
            }
        } else {
            run.await
        }
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
        let built = build_candidates(&request)?;

        // Validate ALL candidates upfront, same as chat_completions.
        let validated: Vec<(String, ModelDefinition<'_>)> = built
            .list
            .iter()
            .map(|c| {
                self.validate(c, Api::OpenAiCompatChatCompletionsStream)
                    .map(|def| (c.clone(), def))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut failover = Failover::new(
            validated.iter().map(|(c, _)| c.clone()).collect(),
            self.config.timeout_secs,
            built.requested_models,
        );

        // Same single-candidate rule as chat_completions: no chain deadline
        // when there is no chain, so the streaming path's guarantee for one
        // candidate is exactly what it was before failover existed.
        let multi = failover.candidates.len() > 1;
        let remaining = failover.remaining();

        let run = async {
            loop {
                let (candidate, def) = match failover.next() {
                    Some(c) => {
                        let def = validated
                            .iter()
                            .find(|(name, _)| name == c)
                            .map(|(_, def)| def)
                            .expect("validated list matches failover candidates");
                        (c.to_string(), def)
                    }
                    None => break Err(failover.exhausted()),
                };

                let normalized = openai_compat::api::chat_completions::ingest_request(
                    request.clone(),
                    def.model.model(),
                );

                match openai_compat::api::chat_completions::exchange_stream(
                    &normalized,
                    &self.http,
                    def.provider.name(),
                    self.base_url(def.provider),
                    def.auth,
                    self.config
                        .stream_read_timeout_secs
                        .map(Duration::from_secs),
                )
                .await
                {
                    Ok(chunks) => {
                        // COMMIT BOUNDARY. A stream is committed only once its
                        // first chunk is in hand; until then nothing has
                        // reached the caller, so a pre-first-chunk failure is
                        // an ordinary failed attempt — record it and try the
                        // next candidate. Once an item is yielded the
                        // candidate is locked in and the rest of the chain is
                        // moot: chunks already emitted cannot be unsent, and
                        // failing over mid-stream would splice a second reply
                        // onto the first.
                        let mut chunks = chunks;
                        match chunks.next().await {
                            Some(Ok(first)) => {
                                let mut rendered =
                                    openai_compat::api::chat_completions::render_chunk(
                                        &first,
                                        def.provider.name(),
                                    );
                                rendered.model = candidate.clone();
                                tracing::info!(
                                    candidate = %candidate,
                                    "failover candidate serving (stream)"
                                );
                                break Ok(ChatCompletionStream {
                                    chunks,
                                    provider: def.provider.name(),
                                    model: candidate,
                                    first: Some(Some(Ok(rendered))),
                                });
                            }
                            Some(Err(e)) if fails_over_before_first_chunk(&e) => {
                                tracing::warn!(
                                    candidate = %candidate,
                                    error = %e,
                                    "failover stream candidate failed before its first chunk, \
                                     trying next"
                                );
                                failover.record_failure(candidate, e);
                                continue;
                            }
                            // A non-retryable error (an event that will not
                            // decode, or a billed one) or a clean-but-empty
                            // stream is a committed outcome. An error is
                            // replayed as the stream's first item and an empty
                            // stream ends as a complete one — both exactly as
                            // they would have using a single candidate, which
                            // is what keeps this path backward compatible.
                            outcome => {
                                tracing::info!(
                                    candidate = %candidate,
                                    "failover candidate serving (stream)"
                                );
                                let model = candidate.clone();
                                break Ok(ChatCompletionStream {
                                    chunks,
                                    provider: def.provider.name(),
                                    model,
                                    first: Some(outcome.map(|item| {
                                        item.map(|chunk| {
                                            let mut rendered =
                                                openai_compat::api::chat_completions::render_chunk(
                                                    &chunk,
                                                    def.provider.name(),
                                                );
                                            rendered.model = candidate.clone();
                                            rendered
                                        })
                                    })),
                                });
                            }
                        }
                    }
                    Err(e) => {
                        if is_retryable(&e) {
                            tracing::warn!(
                                candidate = %candidate,
                                error = %e,
                                "failover stream candidate failed, trying next"
                            );
                            failover.record_failure(candidate, e);
                            continue;
                        }
                        break Err(e);
                    }
                }
            }
        };

        if multi {
            match tokio::time::timeout(remaining, run).await {
                Ok(r) => r,
                Err(_elapsed) => Err(failover.deadline()),
            }
        } else {
            run.await
        }
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
    fn validate(&self, requested: &str, api: Api) -> Result<ModelDefinition<'_>> {
        let (provider, model) = self.fetch_model(requested)?;
        self.validate_declared(provider, requested)?;
        Self::validate_api(model, api, provider, requested)?;
        Ok(ModelDefinition {
            provider,
            model,
            auth: self.authenticator(provider)?,
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
    /// `model` and `api` are the check; `provider` and `requested` only build
    /// the message. Split out from [`Client::chat_completions`] so the refusal
    /// can be tested at all: every model the roster ships today serves chat
    /// completions, so the failing branch is otherwise unreachable from the
    /// public entrypoint until one lands that does not.
    fn validate_api(
        model: &ModelSpec,
        api: Api,
        provider: &ProviderSpec,
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

/// Whether this error is worth trying the next candidate on.
///
/// The axis is REQUEST-scoped vs PROVIDER-scoped, not retryable vs fatal.
/// A request-scoped failure reproduces identically on every candidate —
/// changing provider changes nothing, so it stops the chain. A
/// provider-scoped failure belongs to the credentials or account of the
/// candidate that reported it, so the next candidate — especially across
/// providers — has a real chance.
///
/// Provider-scoped, fail over: `Network` (unreachable), `RateLimited`,
/// `Overloaded`, `ServerError` (transient load), `ModelNotFound` (the list
/// may name different models at different hosts), and `Authentication`,
/// `PermissionDenied`, `QuotaExceeded` — a revoked key or emptied quota on
/// provider A is exactly the outage a `[openai/..., anthropic/...]` list
/// exists to survive.
///
/// Request-scoped or otherwise unrecoverable, fatal: `InvalidRequest`,
/// `ContextLengthExceeded`, `ContentFilter` (any candidate rejects the same
/// body), `Decode` (the winner returned 200 and billed — the next candidate
/// only buys a silent second completion, and schema drift is deterministic
/// across a provider's own OpenAI-compatible surface), and `Unknown`
/// (failing over turns one unexplained failure into several billed ones).
/// Anything warpllm itself decided (`Gateway` origin) is also fatal.
fn is_retryable(err: &Error) -> bool {
    matches!(
        err,
        Error::Network { .. }
            | Error::RateLimited(_)
            | Error::Overloaded(_)
            | Error::ServerError(_)
            | Error::ModelNotFound(_)
            | Error::Authentication(_)
            | Error::PermissionDenied(_)
            | Error::QuotaExceeded(_)
    )
}

/// Whether an error read off a stream that has not yet delivered a single
/// chunk justifies trying the next candidate.
///
/// The prefetch boundary grants what the exchange-level table grants, plus the
/// stream-endpoint failures that a still-silent connection can report:
/// [`Error::StreamTruncated`] (the socket closed before any content) and
/// [`Error::StreamStalled`] (it went quiet on the read timeout). Only here,
/// where zero chunks have reached the caller, are those safe to fail over on —
/// after the first chunk there is content on the wire and a truncated or
/// stalled stream must surface to the caller as itself, never as another
/// candidate spliced in.
fn fails_over_before_first_chunk(err: &Error) -> bool {
    is_retryable(err)
        || matches!(
            err,
            Error::StreamTruncated { .. } | Error::StreamStalled { .. }
        )
}

/// Maximum number of failover candidates a request may name.
///
/// A bounded connection budget at admission: one request must not be able to
/// drive unbounded sequential upstream attempts. Small bounds also make the
/// dedup above trivial.
pub const MAX_FAILOVER_CANDIDATES: usize = 8;

/// The ordered candidate chain for a request, decided at admission.
#[derive(Debug)]
struct Candidates {
    /// Model strings, deduplicated, in caller order.
    list: Vec<String>,
    /// Whether the caller used the `models` field rather than the single
    /// `model` one. Everything downstream — failover semantics, exhaustion
    /// error shape — keys off this, so it must be derived by the same code
    /// that builds the list, never re-derived from `request.models.is_some()`.
    requested_models: bool,
}

/// Build the candidate list from a request's `model` / `models` fields.
///
/// Exactly one must be non-empty; both or neither is
/// [`Error::InvalidInput`]. `models: []` is refused outright rather than
/// treated as absent — "empty" must never silently mean "absent", because
/// that is what lets single-model backward compatibility and failover
/// semantics disagree about the same request. Deduplicates (a repeated
/// candidate burns two attempts on the same endpoint and telling a caller
/// they have a four-chain when they have two) and caps the chain at
/// [`MAX_FAILOVER_CANDIDATES`].
fn build_candidates(request: &CreateChatCompletionRequest) -> Result<Candidates> {
    let has_model = !request.model.is_empty();
    let models = request.models.as_deref();
    let has_models = models.is_some_and(|m| !m.is_empty());

    if models == Some(&[]) {
        return Err(Error::InvalidInput("models must not be empty".into()));
    }

    let list = match (has_model, has_models) {
        (true, true) => {
            return Err(Error::InvalidInput(
                "both model and models are set; use exactly one".into(),
            ));
        }
        (false, false) => {
            return Err(Error::InvalidInput(
                "either model or models is required".into(),
            ));
        }
        (true, false) => vec![request.model.clone()],
        (false, true) => models.unwrap().to_vec(),
    };

    let mut deduped: Vec<String> = Vec::with_capacity(list.len());
    for model in list {
        if !deduped.contains(&model) {
            deduped.push(model);
        }
    }
    if deduped.len() > MAX_FAILOVER_CANDIDATES {
        return Err(Error::InvalidInput(format!(
            "models must not exceed {MAX_FAILOVER_CANDIDATES} candidates"
        )));
    }
    Ok(Candidates {
        list: deduped,
        requested_models: has_models,
    })
}

/// Manages the candidate list, failover loop, deadline enforcement, and
/// error collection for a per-request failover chain.
///
/// Shared between [`Client::chat_completions`] and
/// [`Client::chat_completions_stream`].
struct Failover {
    candidates: Vec<String>,
    deadline: Instant,
    idx: usize,
    tried: Vec<(String, Box<Error>)>,
    /// Whether the caller explicitly passed `models`. When false, the
    /// request used the single `model` field and we preserve backward-
    /// compatible error semantics: a retryable failure returns the inner
    /// error directly instead of wrapping it in `CandidatesExhausted`.
    requested_models: bool,
}

impl Failover {
    fn new(candidates: Vec<String>, timeout_secs: Option<u64>, requested_models: bool) -> Self {
        Self {
            deadline: Instant::now()
                + Duration::from_secs(timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS)),
            candidates,
            idx: 0,
            tried: Vec::new(),
            requested_models,
        }
    }

    /// The next candidate, or `None` when the list is exhausted.
    ///
    /// ADVANCES the cursor, so the loop terminates by construction: getting
    /// `Some` on one iteration can never yield the same candidate on the
    /// next, no matter whether the iteration records a failure or not.
    fn next(&mut self) -> Option<&str> {
        let candidate = self.candidates.get(self.idx)?;
        self.idx += 1;
        Some(candidate.as_str())
    }

    /// Time remaining until the overall deadline, or `Duration::ZERO` if
    /// the deadline has already passed.
    fn remaining(&self) -> Duration {
        self.deadline.saturating_duration_since(Instant::now())
    }

    /// Record a failed attempt. Advancing happens in [`Self::next`].
    fn record_failure(&mut self, candidate: String, error: Error) {
        self.tried.push((candidate, Box::new(error)));
    }

    /// Convert accumulated state into [`Error::CandidatesExhausted`].
    ///
    /// When the request used the single `model` field (not `models`), a
    /// single failure returns the inner error directly for backward
    /// compatibility — callers were never expected to match
    /// `CandidatesExhausted` on a single-model request.
    fn exhausted(&mut self) -> Error {
        if !self.requested_models && self.tried.len() == 1 {
            return *self.tried.drain(..).next().unwrap().1;
        }
        Error::CandidatesExhausted {
            models: std::mem::take(&mut self.candidates),
            tried: std::mem::take(&mut self.tried),
        }
    }

    /// Convert accumulated state into [`Error::DeadlineExceeded`] when the
    /// chain's deadline elapsed before any candidate finished.
    fn deadline(&mut self) -> Error {
        Error::DeadlineExceeded {
            tried: std::mem::take(&mut self.tried),
        }
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
    /// The first item, read at commit time by the failover chain and buffered
    /// here so the caller still receives it. `None` means "nothing prefetched"
    /// (a stream committed and handed over without prefetch); `Some(None)` is
    /// a committed but empty stream; `Some(Some(item))` is the buffered first
    /// chunk or the terminal first error. Everything after unreels from
    /// `chunks`, which was already advanced past this item.
    first: Option<Option<Result<CreateChatCompletionStreamResponse>>>,
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
        if let Some(first) = self.first.take() {
            return first;
        }
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

    /// The gate the whole model-level `supported_apis` split exists for. This
    /// model sits under a perfectly ordinary chat-serving provider and does
    /// not serve chat itself, which only the model's own list can say.
    #[test]
    fn a_model_that_does_not_serve_the_api_is_refused() {
        let err = Client::validate_api(
            &demo_model(vec![SupportedApi {
                api: Api::OpenAiCompatResponses,
            }]),
            Api::OpenAiCompatChatCompletions,
            &demo_provider("https://api.demo.test", Credential::EnvVar("DEMO_API_KEY")),
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
            &demo_model(vec![
                SupportedApi {
                    api: Api::OpenAiCompatChatCompletions,
                },
                SupportedApi {
                    api: Api::OpenAiCompatResponses,
                },
            ]),
            Api::OpenAiCompatChatCompletions,
            &demo_provider("https://api.demo.test", Credential::EnvVar("DEMO_API_KEY")),
            "demo/chat",
        )
        .unwrap();
    }

    /// The point of passing the surface in: one model, and the answer depends
    /// on which surface is asked about. A check that ignored its argument
    /// would pass both of the tests above and fail here.
    #[test]
    fn the_answer_depends_on_which_api_is_asked_about() {
        let model = &demo_model(vec![SupportedApi {
            api: Api::OpenAiCompatResponses,
        }]);
        let provider = &demo_provider("https://api.demo.test", Credential::EnvVar("DEMO_API_KEY"));

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
            &demo_model(vec![SupportedApi {
                api: Api::OpenAiCompatChatCompletions,
            }]),
            Api::OpenAiCompatChatCompletionsStream,
            &demo_provider("https://api.demo.test", Credential::EnvVar("DEMO_API_KEY")),
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
            .validate("openai/gpt-5.6", Api::OpenAiCompatChatCompletions)
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

    // ------------------------------------------------------- failover

    /// A request that names its chain through the `models` extension.
    fn models_request(models: &[&str]) -> CreateChatCompletionRequest {
        CreateChatCompletionRequest {
            models: Some(models.iter().map(|m| m.to_string()).collect()),
            messages: vec![ChatCompletionRequestMessage::new("user", "hi")],
            ..Default::default()
        }
    }

    /// The chain is decided from exactly one field: the single `model` string
    /// routes a one-candidate chain, and the `models` list a multi-candidate
    /// one. The flag is what downstream keyed off it, so it must be derived
    /// here, beside the list it describes.
    #[test]
    fn build_candidates_derives_the_chain_from_model_or_models() {
        let single = build_candidates(&CreateChatCompletionRequest {
            model: "openai/gpt-5.6".into(),
            messages: vec![ChatCompletionRequestMessage::new("user", "hi")],
            ..Default::default()
        })
        .unwrap();
        assert_eq!(single.list, ["openai/gpt-5.6"]);
        assert!(
            !single.requested_models,
            "a single model is not a models list"
        );

        let multi = build_candidates(&models_request(&[
            "openai/gpt-5.6",
            "deepseek/deepseek-v4-flash",
        ]))
        .unwrap();
        assert_eq!(multi.list, ["openai/gpt-5.6", "deepseek/deepseek-v4-flash"]);
        assert!(multi.requested_models);
    }

    /// Both spellings at once is ambiguity worth refusing: which is the
    /// intended chain, and which the accident? Neither is ambiguous; a request
    /// with neither has nothing to route. `models: []` is a third spelling of
    /// "neither" that must not silently mean absent, because absent is a
    /// one-candidate backward-compatible chain and empty is nothing at all.
    #[test]
    fn build_candidates_refuses_ambiguity_and_emptiness() {
        let ambiguous = CreateChatCompletionRequest {
            model: "openai/gpt-5.6".into(),
            messages: vec![ChatCompletionRequestMessage::new("user", "hi")],
            ..models_request(&["deepseek/deepseek-v4-flash"]).clone()
        };
        assert!(matches!(
            build_candidates(&ambiguous),
            Err(Error::InvalidInput(_))
        ));

        let neither = CreateChatCompletionRequest {
            messages: vec![ChatCompletionRequestMessage::new("user", "hi")],
            ..Default::default()
        };
        assert!(matches!(
            build_candidates(&neither),
            Err(Error::InvalidInput(_))
        ));

        let empty = models_request(&[]);
        let message = build_candidates(&empty).unwrap_err().to_string();
        assert!(message.contains("models must not be empty"), "{message}");
    }

    /// A repeated candidate is one endpoint tried twice and reported as a
    /// two-chain — no redundancy gained, half the chain's headroom spent. It
    /// is kept ONCE, in the position it first appeared, so the surviving list
    /// still reads as the caller wrote it.
    #[test]
    fn build_candidates_deduplicates_and_keeps_call_order() {
        let deduped = build_candidates(&models_request(&[
            "deepseek/deepseek-v4-flash",
            "openai/gpt-5.6",
            "deepseek/deepseek-v4-flash",
            "openai/gpt-5.6",
        ]))
        .unwrap();
        assert_eq!(
            deduped.list,
            ["deepseek/deepseek-v4-flash", "openai/gpt-5.6"]
        );
    }

    /// One request must not be able to drive unbounded sequential attempts,
    /// so the chain is capped at admission and told apart from a happy chain
    /// by the same error.
    #[test]
    fn build_candidates_caps_the_chain() {
        let big: Vec<String> = (0..MAX_FAILOVER_CANDIDATES + 3)
            .map(|i| format!("openai/gpt-5.6-{i}"))
            .collect();
        let message = build_candidates(&CreateChatCompletionRequest {
            models: Some(big),
            messages: vec![ChatCompletionRequestMessage::new("user", "hi")],
            ..Default::default()
        })
        .unwrap_err()
        .to_string();
        assert!(
            message.contains(&format!(
                "must not exceed {MAX_FAILOVER_CANDIDATES} candidates"
            )),
            "{message}"
        );
    }

    /// The cursor is what makes the loop terminate: each `next` ADVANCES even
    /// when the iteration that read it records no failure, so no accounting
    /// error downstream can revisit a candidate.
    #[test]
    fn failover_next_advances_and_never_repeats() {
        let mut failover = Failover::new(
            vec!["openai/gpt-5.6".into(), "deepseek/deepseek-v4-flash".into()],
            Some(60),
            true,
        );
        assert_eq!(failover.next(), Some("openai/gpt-5.6"));
        // A failure is recorded AFTER `next` advanced — this is the exact
        // sequence the loop uses — yet the next call still moves on.
        failover.record_failure(
            "openai/gpt-5.6".into(),
            Error::RateLimited(Box::new(provider_error())),
        );
        assert_eq!(failover.next(), Some("deepseek/deepseek-v4-flash"));
        assert_eq!(failover.next(), None, "the chain is exhausted");
        assert_eq!(failover.next(), None, "exhaustion is stable");
    }

    /// A single-model request keeps its pre-failover error shape: when the
    /// one candidate fails, the inner error stands on its own. Nobody was
    /// ever told to look for `CandidatesExhausted` on a single-model call,
    /// and retrofitting one on would break matching done against the old
    /// contract.
    #[test]
    fn failover_exhaustion_preserves_single_model_errors() {
        let mut failover = Failover::new(vec!["openai/gpt-5.6".into()], Some(60), false);
        let inner = Error::RateLimited(Box::new(provider_error()));
        failover.next();
        failover.record_failure("openai/gpt-5.6".into(), inner);
        let err = failover.exhausted();
        assert!(
            matches!(err, Error::RateLimited(_)),
            "a single-model failure surfaces as itself, not wrapped: {err:?}"
        );
    }

    /// The `models` chain, by contrast, is exactly the feature that new is a
    /// wrap for: every attempt in order, and the chain it belonged to, so a
    /// caller told "all candidates exhausted" can see which candidates were
    /// tried and what each said.
    #[test]
    fn failover_exhaustion_reports_every_attempt_and_the_chain() {
        let mut failover = Failover::new(
            vec!["openai/gpt-5.6".into(), "deepseek/deepseek-v4-flash".into()],
            Some(60),
            true,
        );
        failover.next();
        failover.record_failure(
            "openai/gpt-5.6".into(),
            Error::RateLimited(Box::new(provider_error())),
        );
        failover.next();
        failover.record_failure(
            "deepseek/deepseek-v4-flash".into(),
            Error::ServerError(Box::new(provider_error())),
        );
        let err = failover.exhausted();
        // The aggregate is warpllm's verdict, not any provider's: there is no
        // single upstream whose status or retry-after belongs in the envelope.
        assert_eq!(err.origin(), crate::error::Origin::Gateway);
        assert!(err.provider_error().is_none());
        match err {
            Error::CandidatesExhausted { models, tried } => {
                assert_eq!(models, ["openai/gpt-5.6", "deepseek/deepseek-v4-flash"]);
                assert_eq!(tried.len(), 2);
                assert!(matches!(tried[0].0.as_str(), "openai/gpt-5.6"));
                assert!(matches!(tried[1].0.as_str(), "deepseek/deepseek-v4-flash"));
            }
            other => panic!("expected CandidatesExhausted, got {other:?}"),
        }
    }

    /// A chain that runs out of TIME reports exactly what it did get: which
    /// candidates were attempted and how each failed, so nobody mistakes a
    /// deadline for a provider's own timeout.
    #[test]
    fn failover_deadline_reports_every_attempt() {
        let mut failover = Failover::new(vec!["openai/gpt-5.6".into()], Some(60), true);
        failover.next();
        failover.record_failure(
            "openai/gpt-5.6".into(),
            Error::Overloaded(Box::new(provider_error())),
        );
        let deadline = failover.deadline();
        assert_eq!(deadline.code(), "deadline_exceeded");
        match deadline {
            Error::DeadlineExceeded { tried } => {
                assert_eq!(tried.len(), 1);
                assert_eq!(tried[0].0, "openai/gpt-5.6");
            }
            other => panic!("expected DeadlineExceeded, got {other:?}"),
        }
    }

    fn provider_error() -> crate::gateway::types::ProviderError {
        crate::gateway::types::ProviderError {
            provider: "demo",
            status: 500,
            message: "m".into(),
            error_type: None,
            provider_code: None,
            retry_after: None,
            request_id: None,
            raw_body: String::new(),
        }
    }

    /// The classification table, as the reviewers were right to ask for it
    /// explicitly. The axis is REQUEST-scoped vs PROVIDER-scoped, not
    /// "retryable vs fatal": the request-scoped half reproduces identically
    /// on every candidate, so a chain over it would be theater.
    #[tokio::test]
    async fn provider_scoped_errors_fail_over_and_request_scoped_errors_stop_the_chain() {
        let p = provider_error;
        let retryable = [
            Error::Network {
                provider: "demo",
                source: refused().await,
            },
            Error::RateLimited(Box::new(p())),
            Error::Overloaded(Box::new(p())),
            Error::ServerError(Box::new(p())),
            Error::ModelNotFound(Box::new(p())),
            Error::Authentication(Box::new(p())),
            Error::PermissionDenied(Box::new(p())),
            Error::QuotaExceeded(Box::new(p())),
        ];
        for error in &retryable {
            assert!(is_retryable(error), "should fail over: {error:?}");
            assert!(
                fails_over_before_first_chunk(error),
                "the prefetch grants the exchange-level table: {error:?}"
            );
        }

        let fatal = [
            Error::InvalidRequest(Box::new(p())),
            Error::ContextLengthExceeded(Box::new(p())),
            Error::ContentFilter(Box::new(p())),
            Error::Decode {
                provider: "demo",
                message: "bad json".into(),
            },
            Error::Unknown(Box::new(p())),
            Error::Internal("setup".into()),
            Error::InvalidInput("payload".into()),
        ];
        for error in &fatal {
            assert!(!is_retryable(error), "must stop the chain: {error:?}");
        }
    }

    /// A stream that has already delivered chunks can still end truncated or
    /// stalled — those failures belong to a committed candidate, not the
    /// chain, so the prefetch's opts-in-to-failover test must not leak them.
    #[tokio::test]
    async fn a_delivered_streams_failures_do_not_fail_over_the_chain() {
        assert!(fails_over_before_first_chunk(&Error::StreamTruncated {
            provider: "demo"
        }));
        assert!(fails_over_before_first_chunk(&Error::StreamStalled {
            provider: "demo",
            timeout: Duration::from_secs(60),
        }));
        assert!(fails_over_before_first_chunk(&Error::Network {
            provider: "demo",
            source: refused().await,
        }));
        assert!(
            !fails_over_before_first_chunk(&Error::Decode {
                provider: "demo",
                message: "bad json".into(),
            }),
            "a billed 200 must never be re-attempted"
        );
    }

    /// A real `reqwest::Error` for the `Network` rows; the only hard-to-build
    /// variant, taken from a connection that is refused by construction.
    async fn refused() -> reqwest::Error {
        reqwest::Client::new()
            .get("http://127.0.0.1:1/never-routes")
            .send()
            .await
            .unwrap_err()
    }

    /// The whole point of the feature, end to end: a provider-scoped failure
    /// on candidate 1 advances the chain, and candidate 2's reply is the
    /// answer — echoing the string the CALLER routed with, not the upstream's
    /// own model name.
    #[tokio::test]
    async fn a_retryable_failure_advances_to_the_next_candidate() {
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let chain = [
            ("gpt-5.6", 429, "rate limit"),
            ("deepseek-v4-flash", 200, "served by deepseek"),
        ];
        for (model, status, message) in chain {
            let body = if status == 200 {
                serde_json::json!({
                    "id": "chatcmpl-2",
                    "object": "chat.completion",
                    "created": 1_700_000_000,
                    "model": "gpt-5.6",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": message},
                        "finish_reason": "stop"
                    }]
                })
            } else {
                serde_json::json!({
                    "error": {"message": message, "type": "rate_limit_error", "code": "rate_limit_exceeded"}
                })
            };
            Mock::given(method("POST"))
                .and(path("/chat/completions"))
                .and(body_partial_json(serde_json::json!({"model": model})))
                .respond_with(ResponseTemplate::new(status).set_body_json(body))
                .mount(&server)
                .await;
        }

        let client = with_env(
            &[
                ("OPENAI_API_KEY", Some("sk-openai")),
                ("DEEPSEEK_API_KEY", Some("sk-deepseek")),
            ],
            || {
                Client::new(ClientConfig {
                    base_url: Some(server.uri()),
                    ..Default::default()
                })
                .unwrap()
            },
        );
        let completion = client
            .chat_completions(models_request(&[
                "openai/gpt-5.6",
                "deepseek/deepseek-v4-flash",
            ]))
            .await
            .unwrap();
        assert_eq!(completion.model, "deepseek/deepseek-v4-flash");
        assert_eq!(
            completion.choices[0].message.content,
            Some("served by deepseek".into())
        );
        // Both candidates were tried, in order: the mock fan-out happened.
        let requests = server.received_requests().await.unwrap();
        let bodies: Vec<_> = requests
            .iter()
            .map(|r| {
                serde_json::from_slice::<serde_json::Value>(&r.body).unwrap()["model"].to_string()
            })
            .collect();
        assert_eq!(bodies, ["\"gpt-5.6\"", "\"deepseek-v4-flash\""]);
    }

    /// A request-scoped failure is deterministic across candidates — changing
    /// provider changes nothing about the payload — so the chain stops at
    /// the candidate that reported it rather than making the caller pay for a
    /// second doomed attempt.
    #[tokio::test]
    async fn a_request_scoped_failure_stops_the_chain() {
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(serde_json::json!({"model": "gpt-5.6"})))
            .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "error": {"message": "no such feature", "type": "invalid_request_error", "code": "bad_request"}
            })))
            .mount(&server)
            .await;

        let client = with_env(
            &[
                ("OPENAI_API_KEY", Some("sk-openai")),
                ("DEEPSEEK_API_KEY", Some("sk-deepseek")),
            ],
            || {
                Client::new(ClientConfig {
                    base_url: Some(server.uri()),
                    ..Default::default()
                })
                .unwrap()
            },
        );
        let err = client
            .chat_completions(models_request(&[
                "openai/gpt-5.6",
                "deepseek/deepseek-v4-flash",
            ]))
            .await
            .unwrap_err();
        assert!(
            matches!(err, Error::InvalidRequest(_)),
            "the fatal error must surface, not wrap: {err:?}"
        );
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            1,
            "the chain must not attempt a request that would fail identically"
        );
    }

    /// The last mile of `models`: when every candidate fails retryably, the
    /// exhausted verdict names the whole chain and every attempt — not some
    /// single provider's status.
    #[tokio::test]
    async fn every_candidate_failing_exhausts_with_the_chain_attached() {
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        for model in ["gpt-5.6", "deepseek-v4-flash"] {
            Mock::given(method("POST"))
                .and(path("/chat/completions"))
                .and(body_partial_json(serde_json::json!({"model": model})))
                .respond_with(ResponseTemplate::new(503).set_body_json(serde_json::json!({
                    "error": {"message": "load", "type": "server_error", "code": "server_error"}
                })))
                .mount(&server)
                .await;
        }

        let client = with_env(
            &[
                ("OPENAI_API_KEY", Some("sk-openai")),
                ("DEEPSEEK_API_KEY", Some("sk-deepseek")),
            ],
            || {
                Client::new(ClientConfig {
                    base_url: Some(server.uri()),
                    ..Default::default()
                })
                .unwrap()
            },
        );
        let err = client
            .chat_completions(models_request(&[
                "openai/gpt-5.6",
                "deepseek/deepseek-v4-flash",
            ]))
            .await
            .unwrap_err();
        match err {
            Error::CandidatesExhausted { models, tried } => {
                assert_eq!(models, ["openai/gpt-5.6", "deepseek/deepseek-v4-flash"]);
                assert_eq!(tried.len(), 2, "{tried:?}");
            }
            other => panic!("expected CandidatesExhausted, got {other:?}"),
        }
    }

    fn sse_chunk() -> String {
        format!(
            "data: {}\n\n",
            serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1_700_000_000,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": null}]
            })
        )
    }

    /// The streaming commit boundary, pinned at the moment the first chunk
    /// lands: after that chunk nothing is re-routed. The candidate that
    /// yielded it is serving — its middle truncation surfaces to the caller
    /// as itself, and the supposedly backup candidate is never contacted.
    #[tokio::test]
    async fn a_stream_locks_in_at_its_first_chunk() {
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // One chunk and then the socket closes: no sentinel, no blank line —
        // the exact body a connection dying mid-answer produces.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(serde_json::json!({"model": "gpt-5.6"})))
            .respond_with(ResponseTemplate::new(200).set_body_string(sse_chunk()))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(
                serde_json::json!({"model": "deepseek-v4-flash"}),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_string(format!(
                "{}{}",
                sse_chunk(),
                "data: [DONE]\n\n"
            )))
            .mount(&server)
            .await;

        let client = with_env(
            &[
                ("OPENAI_API_KEY", Some("sk-openai")),
                ("DEEPSEEK_API_KEY", Some("sk-deepseek")),
            ],
            || {
                Client::new(ClientConfig {
                    base_url: Some(server.uri()),
                    ..Default::default()
                })
                .unwrap()
            },
        );
        let mut stream = client
            .chat_completions_stream(models_request(&[
                "openai/gpt-5.6",
                "deepseek/deepseek-v4-flash",
            ]))
            .await
            .unwrap();

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(
            first.choices[0]
                .delta
                .content
                .as_ref()
                .unwrap()
                .as_ref()
                .unwrap(),
            "hi"
        );
        assert!(
            matches!(
                stream.next().await,
                Some(Err(Error::StreamTruncated { provider: "openai" }))
            ),
            "a committed stream reports its own truncation"
        );
        assert!(stream.next().await.is_none());
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            1,
            "committed at the first chunk: the backup candidate is never contacted"
        );
    }

    /// ...and before that first chunk, the candidate is NOT committed: an
    /// opened stream that goes silent without a single chunk is a failed
    /// attempt like any other, and the next candidate serves instead.
    #[tokio::test]
    async fn a_stream_that_fails_before_its_first_chunk_fails_over() {
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // A 200 with nothing in it: the socket closes before any chunk, which
        // the transport reports as StreamTruncated on the first read.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(serde_json::json!({"model": "gpt-5.6"})))
            .respond_with(ResponseTemplate::new(200).set_body_string(""))
            .mount(&server)
            .await;
        let done = "data: [DONE]\n\n".to_string();
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(
                serde_json::json!({"model": "deepseek-v4-flash"}),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_string(format!(
                "{}{}",
                sse_chunk(),
                done
            )))
            .mount(&server)
            .await;

        let client = with_env(
            &[
                ("OPENAI_API_KEY", Some("sk-openai")),
                ("DEEPSEEK_API_KEY", Some("sk-deepseek")),
            ],
            || {
                Client::new(ClientConfig {
                    base_url: Some(server.uri()),
                    ..Default::default()
                })
                .unwrap()
            },
        );
        let mut stream = client
            .chat_completions_stream(models_request(&[
                "openai/gpt-5.6",
                "deepseek/deepseek-v4-flash",
            ]))
            .await
            .unwrap();

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(
            first.choices[0]
                .delta
                .content
                .as_ref()
                .unwrap()
                .as_ref()
                .unwrap(),
            "hi",
            "the second candidate served"
        );
        assert_eq!("deepseek/deepseek-v4-flash", first.model);
        assert!(stream.next().await.is_none());
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            2,
            "the truncated candidate was tried and its backup served"
        );
    }

    /// A clean-but-empty stream — a candidate that 200s and sends only
    /// `[DONE]`, no chunks — is a COMPLETE reply, not a failure: the exact
    /// outcome a single candidate delivers, so the chain hands it over intact.
    #[tokio::test]
    async fn a_stream_that_ends_clean_without_chunks_is_a_complete_reply() {
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(serde_json::json!({"model": "gpt-5.6"})))
            .respond_with(ResponseTemplate::new(200).set_body_string("data: [DONE]\n\n"))
            .mount(&server)
            .await;

        let client = with_env(
            &[
                ("OPENAI_API_KEY", Some("sk-openai")),
                ("DEEPSEEK_API_KEY", Some("sk-deepseek")),
            ],
            || {
                Client::new(ClientConfig {
                    base_url: Some(server.uri()),
                    ..Default::default()
                })
                .unwrap()
            },
        );
        let mut stream = client
            .chat_completions_stream(models_request(&[
                "openai/gpt-5.6",
                "deepseek/deepseek-v4-flash",
            ]))
            .await
            .unwrap();
        assert!(
            stream.next().await.is_none(),
            "a clean empty stream ends as an empty stream"
        );
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            1,
            "an empty-but-complete reply commits, never fails over"
        );
    }
}
