//! Where a provider on this protocol diverges from the protocol itself.
//!
//! An entry here is a claim that one backend genuinely differs from
//! `openai_compat` — not a place to restate what the protocol already says.
//! Most providers need nothing: [`error_mapper`] hands back
//! [`MatchesProtocol`] and they inherit the baseline whole, which is why
//! adding a faithful backend to the roster is a YAML edit and no Rust at all.
//!
//! One directory per provider, one module inside it per thing that provider
//! does differently — `deepseek/error.rs` maps its error codes, statuses and
//! types, and a second divergence would land beside it rather than widening
//! it.
//!
//! There is deliberately no `openai/`. OpenAI's spellings ARE the
//! `openai_compat` baseline, so an empty impl for symmetry would be a
//! directory that says nothing and a second place to look for what
//! [`super::error`] already holds.
//!
//! A provider whose error ENVELOPE is shaped differently does not belong
//! here. That is a different protocol, and it gets a sibling of
//! [`super`](crate::gateway::openai_compat) — these overrides answer "what
//! does this spelling mean", never "where do I find it".

mod deepseek;

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::gateway::error::{ErrorMapper, MatchesProtocol};

/// Every provider on this protocol with an error mapping of its own, keyed by
/// its roster name.
///
/// A map rather than a `match` so the entries are ENUMERABLE: the keys are
/// `specs.yaml` provider names, and a rename there that orphans one here
/// would silently stop applying that provider's overrides — a wrong
/// classification, not a crash. The test below is what catches it.
///
/// `LazyLock` because a `HashMap` cannot be `const`, the same arrangement
/// [`registry`](crate::registry) uses for the roster itself: built once on
/// first use, read-only forever after.
static OVERRIDES: LazyLock<HashMap<&'static str, &'static dyn ErrorMapper>> = LazyLock::new(|| {
    HashMap::from([(
        "deepseek",
        &deepseek::error::DeepSeek as &'static dyn ErrorMapper,
    )])
});

/// The provider's own error mapping, or [`MatchesProtocol`] if it has none.
///
/// A miss is the NORMAL answer, not a failure: most providers speak their
/// protocol faithfully and inherit it whole, which is what keeps adding one a
/// roster edit and no Rust.
pub(super) fn error_mapper(provider: &str) -> &'static dyn ErrorMapper {
    OVERRIDES.get(provider).copied().unwrap_or(&MatchesProtocol)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The keys here are roster keys, matched by string. Rename a provider in
    /// `specs.yaml` without renaming it here and its overrides quietly stop
    /// applying — the failure keeps classifying, just wrongly, which is the
    /// kind of bug no request ever surfaces.
    #[test]
    fn every_override_names_a_provider_the_roster_has() {
        for name in OVERRIDES.keys() {
            assert!(
                crate::registry::provider(name).is_some(),
                "`{name}` has error overrides but is not in specs.yaml — a roster \
                 rename would leave them silently unused"
            );
        }
    }

    /// A provider nobody has written an override for inherits the protocol,
    /// rather than falling into some emptier default. This is the path every
    /// faithful backend takes.
    #[test]
    fn an_unlisted_provider_gets_nothing_of_its_own() {
        let mapper = error_mapper("openai");
        assert!(mapper.from_code("insufficient_quota").is_none());
        assert!(mapper.from_status(402).is_none());
        assert!(mapper.from_type("api_error").is_none());
    }
}
