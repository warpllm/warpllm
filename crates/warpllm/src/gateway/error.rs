//! What an upstream failure MEANS, and the seam that lets one provider
//! differ from the protocol it speaks.
//!
//! Provider-agnostic, and deliberately so: this module names no protocol and
//! no provider. It owns the VOCABULARY LOOKUP as a trait and the ORDER those
//! lookups run in, and nothing else. Extracting `type`/`code`/`message` out
//! of a body is a protocol's job, because only a protocol knows the shape of
//! its error envelope.
//!
//! Three layers meet here, and each knows strictly less than the one below:
//!
//! 1. [`Error`] and [`ProviderError`] — the canonical failures. No protocol,
//!    no provider.
//! 2. A PROTOCOL's vocabulary — the spellings every backend speaking it
//!    shares. `openai_compat` next door.
//! 3. A PROVIDER's vocabulary — only where that provider genuinely diverges
//!    from its protocol.
//!
//! A provider with nothing to add implements nothing, and inherits its
//! protocol whole. That is what keeps a new roster entry a YAML edit: adding
//! a backend that speaks `openai_compat` faithfully needs no Rust at all, and
//! cannot silently lose the taxonomy by forgetting to write some.

use crate::error::Error;
use crate::gateway::types::ProviderError;

/// The classifier's answer: the [`Error`] variant this failure IS.
///
/// A variant constructor rather than a separate kind enum — the flat
/// [`Error`] already names every failure, so a parallel vocabulary here would
/// be a second level reintroduced one module over.
pub(crate) type Classified = fn(Box<ProviderError>) -> Error;

/// What a set of error spellings MEANS, as two lookups.
///
/// Implemented twice over for any given exchange: once by the protocol, for
/// the vocabulary its whole ecosystem shares, and once by the provider, for
/// what only it does. [`classify`] interleaves them.
///
/// TWO METHODS, NOT ONE, and the split that remains is the one [`classify`]
/// can enforce without knowing any vocabulary. A `code` is the upstream NAMING
/// its own failure, and no reading of the envelope around it — from the
/// provider or from anyone — should outrank that. A single `classify`-shaped
/// hook would lose exactly that guarantee: a provider mapping a bare status
/// would beat the protocol reading an explicit slug.
///
/// TWO METHODS, NOT THREE, and that is the other half. Status and `type` were
/// once separate hooks ranked status-over-type, which made the ranking
/// [`classify`]'s to impose on every protocol at once — and made a provider's
/// `type` structurally unable to outrank its protocol's status, however
/// decisive that `type` was. OpenCode Zen is where that broke: it reports
/// credit exhaustion as `CreditsError` at HTTP 401, the same status it uses
/// for a genuinely bad key, so neither signal decides alone and no rule
/// written on one of them could ever have been right. Which of the two is
/// stronger is a property of a particular vocabulary, not of HTTP, so it
/// belongs to the mapper holding both — where the arms of one `match` state it
/// in reading order.
///
/// Both methods default to `None`, so an implementor writes only the lookups
/// it actually has.
// `from_*` taking `&self` trips `clippy::wrong_self_convention`, which reads
// the prefix as a constructor. These are lookups on a mapper — "what does this
// mapper make of this code" — and the name states the SIGNAL each one reads,
// which is what `classify`'s ordering is built around. Renaming to `by_*`
// would lose that reading for a lint about constructors that do not exist here.
#[allow(clippy::wrong_self_convention)]
pub(crate) trait ErrorMapper: Sync {
    /// The provider's `code` — the failure naming itself. Strongest.
    fn from_code(&self, _code: &str) -> Option<Classified> {
        None
    }

    /// The rest of the envelope: the HTTP status, and the `type` family the
    /// body named beside it. Weaker than a `code`, which is why it is asked
    /// second — but weaker only as a pair, because within it the two signals
    /// have no fixed order to enforce.
    ///
    /// `error_type` is `Option` because plenty of backends send a bare status
    /// with no envelope at all; the status is always there to read.
    fn from_status_and_type(&self, _status: u16, _error_type: Option<&str>) -> Option<Classified> {
        None
    }
}

/// Nothing at all — the vocabulary of a provider that matches its protocol.
pub(crate) struct MatchesProtocol;

impl ErrorMapper for MatchesProtocol {}

/// Resolves one failure against a provider and the protocol it speaks,
/// strongest signal first.
///
/// The tier order lives HERE, once, rather than in each protocol: that a
/// named `code` outranks the envelope around it is a property of HTTP error
/// envelopes in general, not of any one wire format. Within a tier the
/// provider is asked first, since it is the more specific authority on its own
/// spellings — but a tier is never skipped, so a provider's reading of a
/// status or a family cannot outrank the protocol's reading of an explicit
/// `code`.
///
/// Four lookups, and nothing finer. Ranking a status against a `type` needs to
/// know what a particular vocabulary spells them with, which is precisely what
/// this function refuses to know — so that call sits inside each
/// [`ErrorMapper`] instead.
///
/// The residual is pure HTTP semantics and belongs to no vocabulary: a 4xx
/// nobody recognized is still a client error, and a 5xx is still the server's.
/// Anything else stays [`Error::Unknown`] rather than becoming a guess.
pub(crate) fn classify(
    provider: &dyn ErrorMapper,
    protocol: &dyn ErrorMapper,
    status: u16,
    error_type: Option<&str>,
    code: Option<&str>,
) -> Classified {
    code.and_then(|code| provider.from_code(code))
        .or_else(|| code.and_then(|code| protocol.from_code(code)))
        .or_else(|| provider.from_status_and_type(status, error_type))
        .or_else(|| protocol.from_status_and_type(status, error_type))
        .unwrap_or(match status {
            400 | 422 => Error::InvalidRequest,
            500..=599 => Error::ServerError,
            _ => Error::Unknown,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A protocol that answers on both tiers, and on each signal within the
    /// second, so a provider rule can be shown to win or lose against each one
    /// independently.
    struct Protocol;

    impl ErrorMapper for Protocol {
        fn from_code(&self, code: &str) -> Option<Classified> {
            (code == "known_code").then_some(Error::ContentFilter as Classified)
        }
        fn from_status_and_type(
            &self,
            status: u16,
            error_type: Option<&str>,
        ) -> Option<Classified> {
            Some(match (status, error_type) {
                (429, _) => Error::RateLimited,
                (_, Some("known_type")) => Error::ServerError,
                _ => return None,
            })
        }
    }

    /// A provider that reads the envelope both ways: a status the protocol
    /// says nothing about, and a family the protocol would otherwise never be
    /// asked for because its own status rule fires first.
    struct ProviderEnvelope;

    impl ErrorMapper for ProviderEnvelope {
        fn from_status_and_type(
            &self,
            status: u16,
            error_type: Option<&str>,
        ) -> Option<Classified> {
            Some(match (status, error_type) {
                (402, _) => Error::QuotaExceeded,
                (_, Some("provider_type")) => Error::ContextLengthExceeded,
                _ => return None,
            })
        }
    }

    fn code_of(
        provider: &dyn ErrorMapper,
        status: u16,
        error_type: Option<&str>,
        code: Option<&str>,
    ) -> &'static str {
        classify(provider, &Protocol, status, error_type, code)(Box::new(ProviderError {
            provider: "demo",
            status,
            message: String::new(),
            error_type: None,
            provider_code: None,
            retry_after: None,
            request_id: None,
            raw_body: String::new(),
        }))
        .code()
    }

    /// A provider that adds nothing inherits its protocol whole — the case
    /// every faithful backend is in, and the reason adding one needs no Rust.
    #[test]
    fn a_provider_with_no_vocabulary_inherits_the_protocol() {
        let none = &MatchesProtocol;
        assert_eq!(code_of(none, 429, None, None), "rate_limited");
        assert_eq!(
            code_of(none, 200, None, Some("known_code")),
            "content_filter"
        );
        assert_eq!(
            code_of(none, 200, Some("known_type"), None),
            "provider_server_error"
        );
    }

    /// Within a tier the provider is the more specific authority and wins.
    #[test]
    fn a_provider_rule_beats_the_protocol_at_the_same_tier() {
        assert_eq!(
            code_of(&ProviderEnvelope, 402, None, None),
            "quota_exceeded"
        );
    }

    /// THE case the remaining split exists for. A provider's reading of the
    /// envelope must NOT outrank the protocol reading an explicit `code`: the
    /// upstream named its own failure, and a rule about status numbers or
    /// families is a weaker signal than that. A single `classify` hook would
    /// get this backwards.
    #[test]
    fn a_provider_envelope_rule_does_not_outrank_a_protocol_code() {
        assert_eq!(
            code_of(&ProviderEnvelope, 402, None, Some("known_code")),
            "content_filter",
            "a status guess outranked the failure the provider named"
        );
    }

    /// What merging the two envelope lookups BOUGHT, and the shape issue #66
    /// reported: a provider names a failure with a `type`, under a status its
    /// protocol already claims. While status and family were separate tiers
    /// this was unreachable — every status outranked every family, so the
    /// protocol's reading of 429 answered and the provider's rule could not be
    /// consulted no matter what it said.
    #[test]
    fn a_provider_family_outranks_a_protocol_status() {
        assert_eq!(
            code_of(&ProviderEnvelope, 429, Some("provider_type"), None),
            "context_length_exceeded",
            "the protocol's status answered a failure the provider had named"
        );
        // ...and with nothing from the provider to read, that same status is
        // still the protocol's to answer.
        assert_eq!(code_of(&ProviderEnvelope, 429, None, None), "rate_limited");
    }

    /// Nothing recognized anywhere: HTTP semantics answer, and only where
    /// they actually say something.
    #[test]
    fn the_residual_is_http_semantics_and_nothing_more() {
        let none = &MatchesProtocol;
        assert_eq!(code_of(none, 400, None, None), "provider_invalid_request");
        assert_eq!(code_of(none, 422, None, None), "provider_invalid_request");
        assert_eq!(code_of(none, 500, None, None), "provider_server_error");
        assert_eq!(code_of(none, 502, None, None), "provider_server_error");
        // A status that means nothing in particular must not be guessed at.
        assert_eq!(code_of(none, 418, None, None), "provider_unknown");
    }
}
