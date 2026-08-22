//! What the registry holds, and how a caller reads it.
//!
//! Two levels, two types, because they answer two questions. A
//! [`ProviderSpec`] is how an API is reached: one host and one credential. A
//! [`ModelSpec`] is one routable name under that provider, carrying what
//! differs between models of the same one — its limits, and which API surfaces
//! it serves. [`crate::fetch_model`] hands back one of each and merges nothing.
//!
//! `supported_apis` is the MODEL's, and only the model's. A provider is a
//! host, not a capability: one host commonly serves chat completions,
//! embeddings, and moderation from three disjoint sets of models, so a
//! provider-level list could only ever be their union — true of the host and
//! false of every model under it.
//!
//! Which WIRE FORMAT is spoken is the model's too, by the same argument and
//! one step further: an [`Api`] names its own protocol, so a provider-level
//! field could only restate what every surface under it already says, or
//! contradict it. A host is free to serve one model over one protocol and its
//! neighbour over another, and nothing here has to be taught that.
//!
//! These are READ SURFACES, not the YAML schema: `load` next door owns the
//! schema and does the settling, which is why nothing here is an `Option`
//! meaning "not answered yet". A roster that leaves a required field unset
//! fails to load, so a spec that exists is a spec that is complete, and the
//! accessors below can hand back values rather than possibilities.
//!
//! How a provider authenticates is the one field with more than two states,
//! and [`Credential`] spells all three out rather than leaving one of them as
//! the absence of the others: a variable to read, nothing to send, or no way to
//! authenticate at all. [`ProviderSpec::env_api_key`] answers only the first,
//! so it stays an `Option` and keeps meaning what it always did. The three
//! [`Capabilities`] limits are the other genuine `Option`s, and there `None`
//! means undocumented — never unlimited.
//!
//! Fields are `pub(crate)` so the loader can build these. They are private to
//! everyone else: outside this crate a spec is read-only.

use std::collections::HashMap;

use crate::types::Api;

/// The resolved roster: providers, and every routable `model_str` under them.
///
/// Two `HashMap`s rather than one merged table. They are keyed at different
/// levels and looked up in sequence — the model row names its provider, and
/// the provider row is fetched by that name — so a provider's transport is
/// stored once no matter how many models it serves.
#[derive(Debug, Default)]
pub(crate) struct Registry {
    /// Keyed by provider name, the first segment of a `model_str`.
    pub(crate) providers: HashMap<String, ProviderSpec>,
    /// Keyed by the whole `model_str`, prefix included.
    pub(crate) models: HashMap<String, ModelSpec>,
}

/// One provider: where its API is, and how to authenticate.
///
/// Transport, and nothing else. Everything here is true of every model the
/// provider serves, which is what keeps it stated exactly once — and it is why
/// what a model can DO, and which wire format that is spoken in, are not here.
#[derive(Debug, Clone)]
pub struct ProviderSpec {
    /// Interned, and the only field here that is. A name reaches the public
    /// error surface, which holds it as `&'static str`; `intern` next door
    /// argues why that is worth a bounded leak.
    pub(crate) name: &'static str,
    pub(crate) base_url: String,
    pub(crate) credential: Credential,
}

/// How a provider authenticates, with all three states named.
///
/// Three, not two, because "the roster names a variable" and "the roster says
/// this host wants nothing" and "the roster has no answer" are different
/// situations with different remedies — and the third has to stay reachable,
/// since it is what an entry means today when it simply says nothing. Folding
/// it into the second would make a forgotten `env_api_key:` line silently send
/// a prompt to a paid host with no credential.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Credential {
    /// `env_api_key: FOO` — read `FOO` from the environment.
    EnvVar(&'static str),
    /// `auth: none` — the host takes no credential, so no `Authorization`
    /// header is sent at all. What a self-hosted box on a private network
    /// declares.
    NotRequired,
    /// Neither field. The provider cannot be authenticated, and a request
    /// routed to it says so rather than naming a variable nothing reads.
    Unavailable,
}

impl ProviderSpec {
    /// The provider's name — its key in the roster, and the first segment of
    /// every `model_str` it serves.
    ///
    /// `&'static str` because a name outlives the roster it was read from:
    /// it is what the error types carry, and they are handed to callers who
    /// may well have dropped the client by then.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The provider's API root, version prefix included and no trailing
    /// slash; an endpoint appends its own path.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The environment variable warpllm reads this provider's API key from, if
    /// the roster names one. The default key source, and the only one a client
    /// that declares nothing has.
    ///
    /// `None` covers the two cases where there is no variable to name, which
    /// this question cannot tell apart and does not try to: the provider takes
    /// no credential at all, or the roster records no way to authenticate it.
    /// [`unauthenticated`](Self::unauthenticated) is the one that separates
    /// them, and it is the one a request has to ask.
    ///
    /// Either way this is only the ROSTER's answer. A client that supplies the
    /// key itself through
    /// [`ProviderConfig::api_key`](crate::ProviderConfig::api_key) can
    /// authenticate a provider named no variable here at all.
    pub fn env_api_key(&self) -> Option<&'static str> {
        match self.credential {
            Credential::EnvVar(var) => Some(var),
            Credential::NotRequired | Credential::Unavailable => None,
        }
    }

    /// Whether this provider is served with NO credential — the roster's
    /// `auth: none`, which a self-hosted host on a private network declares.
    ///
    /// The distinction that matters is against a provider whose entry simply
    /// names nothing: that one cannot be authenticated, and a request to it
    /// fails saying so. This one is complete, and a request to it sends no
    /// `Authorization` header. Only a deliberate line in the roster produces
    /// it, so nobody reaches an unauthenticated endpoint by forgetting one.
    ///
    /// It says what the ROSTER declares, not what happens: a client that
    /// declares an inline key for this provider is making the more specific
    /// statement and that key still goes out. Somebody who put a token in front
    /// of their own box has said something the roster file could not.
    pub fn unauthenticated(&self) -> bool {
        self.credential == Credential::NotRequired
    }
}

/// One routable model: the name it ships upstream, the surfaces it serves,
/// and its published limits.
///
/// Deliberately thin. The transport lives in [`ProviderSpec`], so an entry
/// here is only what makes this model different from its siblings — which for
/// most models is nothing at all.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// The provider serving this model — the key its [`ProviderSpec`] is
    /// filed under, and the first segment of this model's own key.
    pub(crate) provider: String,
    /// Upstream model name — what ships on the wire. Defaults to the key's
    /// last segment, so it differs only when warpllm's routing alias differs
    /// from the provider's own model name.
    pub(crate) model: String,
    /// Every surface this model serves, written out in the roster. Required
    /// and never inherited: a model that says nothing serves nothing, which is
    /// a load failure rather than a silent claim on everything its host does.
    pub(crate) supported_apis: Vec<SupportedApi>,
    pub(crate) capabilities: Capabilities,
    /// The day the provider stops serving this model, `YYYY-MM-DD`, when one
    /// has been announced.
    ///
    /// Where a provider publishes two dates — the day it announces the
    /// deprecation and the later day access actually ends — this holds the
    /// second. A deprecated model still answers, so the date worth recording
    /// is the one after which routing to it breaks. `None` is the ordinary
    /// case and means nothing is scheduled, never that the model is permanent.
    pub(crate) deprecation_date: Option<String>,
}

impl ModelSpec {
    /// The model name as it ships upstream, which differs from the
    /// `model_str` whenever warpllm's routing alias differs from the
    /// provider's own name for it.
    ///
    /// Always a real name: the roster registers every routable model by name,
    /// so there is no entry that serves many and pins none.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Every surface this model serves, with what the roster records about
    /// each, in the order the file lists them. Never empty.
    ///
    /// Each names its protocol as well as its surface, so a model answering
    /// the same idea in two wire formats lists both and nothing has to decide
    /// they are the same thing. It is also the only place the roster records a
    /// wire format at all.
    pub fn supported_apis(&self) -> &[SupportedApi] {
        &self.supported_apis
    }

    /// The day the provider stops serving this model, `YYYY-MM-DD`, or `None`
    /// when the roster records no scheduled retirement.
    ///
    /// Nothing acts on it: routing does not consult it, and a model whose date
    /// has passed still resolves. The string is whatever the roster wrote —
    /// the loader does not check it against a calendar — so a caller reading
    /// it parses it.
    pub fn deprecation_date(&self) -> Option<&str> {
        self.deprecation_date.as_deref()
    }

    /// This model's entry for `api`, or `None` if it does not serve it. The
    /// gate a request passes before it is routed anywhere.
    ///
    /// Hands back the ENTRY rather than a `bool` so that a caller asking
    /// whether the model serves something already holds what the roster
    /// records about serving it — the shape that survives
    /// [`SupportedApi`] gaining fields.
    ///
    /// One method for every surface, rather than one method each: [`Api`]
    /// carries no payload, so it can simply be passed in.
    pub fn supported_api(&self, api: Api) -> Option<&SupportedApi> {
        self.supported_apis.iter().find(|entry| entry.api == api)
    }

    /// Whether this model serves `api` at all.
    pub fn supports_api(&self, api: Api) -> bool {
        self.supported_api(api).is_some()
    }

    /// What this model's published limits are.
    pub fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }
}

/// One entry in a model's `supported_apis`: a surface it serves, and what the
/// roster records about serving it.
///
/// Written `- {api: openai_compat_chat_completions}`. The surface is a field rather
/// than the entry's own shape, and that is the whole design: a field added here
/// belongs to EVERY surface at once. `input_modalities` recorded per-surface
/// would otherwise mean three payload types to add it to and keep in step, with
/// nothing to catch the one that was missed.
///
/// Carries only `api` today. An entry is therefore one key wide, which is why
/// the roster writes it on one line.
///
/// A YAML schema in its own right, like [`Capabilities`] and for the same
/// reason: it maps one-to-one onto what the file writes, with nothing to
/// settle, so a second struct to deserialize into would be a copy to keep in
/// step. `deny_unknown_fields` is what turns a contributor's `apis:` typo into
/// an error rather than an entry that quietly names no surface.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct SupportedApi {
    pub(crate) api: Api,
}

impl SupportedApi {
    /// Which surface this entry is about.
    pub fn api(&self) -> Api {
        self.api
    }
}

/// A model's published limits. Deliberately NOT coupled to the request shape:
/// parameter support is passthrough — the provider is the authority and
/// rejects what it doesn't accept. A field is added here only when a real
/// consumer need arrives with it.
///
/// The one type that IS its own YAML schema: it maps one-to-one onto a
/// `capabilities:` block with nothing to settle, so a second struct to
/// deserialize into would be a copy to keep in step. `deny_unknown_fields` is
/// what turns a contributor's `max_input_token:` typo into an error instead of
/// a silently ignored line.
///
/// No `Default`, deliberately. A derived `Default` on a public struct is
/// public too, and the loader's blank starting point is not something a caller
/// should be able to conjure — so it gets `Capabilities::blank`, which is
/// `pub(crate)` and therefore unreachable from outside.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capabilities {
    pub(crate) max_input_tokens: Option<u32>,
    pub(crate) max_output_tokens: Option<u32>,
    /// Requests this model will serve at once. Unset means undocumented,
    /// NOT unlimited. Account tier can move these, so treat the roster
    /// value as the default a config surface would later override.
    pub(crate) max_concurrent_requests: Option<u32>,
}

impl Capabilities {
    /// Nothing recorded — what an entry with no `capabilities:` block
    /// deserializes to.
    pub(crate) const fn blank() -> Self {
        Self {
            max_input_tokens: None,
            max_output_tokens: None,
            max_concurrent_requests: None,
        }
    }

    /// Largest documented input context, in tokens.
    ///
    /// `None` means the registry has no published figure for this model — it
    /// never means unlimited. These three stay `Option` precisely because
    /// undocumented and unbounded are different claims, and the registry
    /// refuses to guess between them.
    pub fn max_input_tokens(&self) -> Option<u32> {
        self.max_input_tokens
    }

    /// Largest documented output length, in tokens. `None` means
    /// undocumented, not unlimited.
    pub fn max_output_tokens(&self) -> Option<u32> {
        self.max_output_tokens
    }

    /// Documented ceiling on requests served at once. `None` means
    /// undocumented, not unlimited. Account tier can move this, so treat it
    /// as a default rather than a hard limit.
    pub fn max_concurrent_requests(&self) -> Option<u32> {
        self.max_concurrent_requests
    }
}
