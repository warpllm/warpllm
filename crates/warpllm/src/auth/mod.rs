//! How a provider's credential is presented on the wire.
//!
//! One [`Authenticator`] per provider, resolved by [`crate::credentials`] and
//! handed to a transport, which applies it to a request it has otherwise
//! finished building. That split is the point: the PATH and the ENVELOPE differ
//! per wire format and belong to a protocol's transport, while the CREDENTIAL
//! differs per provider and belongs here. Vertex (#25) is the case that settles
//! it — one OAuth token serves both the Anthropic Messages and the Google
//! `generateContent` surfaces under a single provider entry, so a credential
//! filed under either protocol would have to be built twice.
//!
//! At the crate root rather than under [`crate::protocol`], and for the same
//! reason [`crate::http`] is: this is transport machinery every protocol module
//! uses and none of them owns. `protocol` is indexed by wire format; a
//! credential is not.
//!
//! # A whole request, not a header
//!
//! [`Authenticator::authenticate`] takes a built [`reqwest::Request`] and hands
//! one back. A `headers() -> (name, value)` seam would be smaller and could not
//! express SigV4 (#24), which signs the method, the URL, the headers AND a hash
//! of the body — there is no value to compute without them. Building first and
//! authenticating second is what makes the body available to sign, and it costs
//! the simple case nothing.
//!
//! It is `async` and fallible though nothing here is either yet. Both arrive
//! with the first cloud credential — SigV4 can fail on a body it cannot read,
//! and an OAuth token can await a refresh — and a call site written
//! `auth.authenticate(request).await?` today does not change when they do.
//!
//! # Secrets and `Debug`
//!
//! Every variant's payload HAND-WRITES its own [`std::fmt::Debug`]. A derived
//! one prints the secret the moment the value lands in a `tracing` field or a
//! panic message, which is the same reasoning behind [`crate::credentials`]'s.
//! The header value is additionally marked sensitive, so the secret does not
//! survive into [`reqwest::Request`]'s own `Debug` either.

mod header;
mod oauth;

#[cfg(test)]
pub(crate) mod testing;

use crate::error::Result;

pub(crate) use header::Header;
pub(crate) use oauth::OauthBearer;

/// One provider's resolved credential.
///
/// An enum rather than a trait: the crate's only trait is `ErrorMapper`, and it
/// is one because the provider-override table is genuinely open-ended. The set
/// of ways to authenticate is not — it grows by an issue at a time, and a
/// closed match is what makes a new one fail to compile until every site
/// handles it. `Protocol` in [`crate::types`] is kept as a one-variant enum on
/// the same reasoning.
///
/// A variant holds a RESOLVED value, never a resolver. `SigV4 { keys, region }`
/// rather than `SigV4 { chain }`, so a credential built in a test is literals
/// and cannot reach `~/.aws/config`, honour `AWS_PROFILE`, or call out to
/// instance metadata. `credentials::with_env` exists to keep the ambient
/// environment out of tests; a resolver parked in here would walk straight
/// around it.
///
/// `Debug` is derived HERE and hand-written on each payload — see the module
/// docs. A variant added without doing that leaks its secret.
#[derive(Debug)]
pub(crate) enum Authenticator {
    /// A secret sent in a named header. `Authorization: Bearer …` and
    /// Anthropic's `x-api-key: …` differ only in the header name and the
    /// prefix, so they are one variant rather than two.
    Header(Header),
    // SigV4 { .. }        <- #24, in `sigv4.rs`
    /// Reads as dead outside tests until token minting from ADC exists —
    /// tracked as a follow-up to #25. The variant and its `apply` path are
    /// real and tested; only the constructor that would reach them from a
    /// live request is missing.
    #[allow(dead_code)]
    OauthBearer(OauthBearer),
}

impl Authenticator {
    /// `Authorization: Bearer <secret>` — what every provider on the roster
    /// today uses, and what an OpenAI-compatible endpoint expects.
    pub(crate) fn bearer(secret: String) -> Self {
        Self::Header(Header::bearer(secret))
    }

    /// `x-api-key: <secret>` — Anthropic's spelling, which deliberately does
    /// NOT also set `Authorization`.
    ///
    /// Reads as dead outside tests, and is: nothing routes to Anthropic yet, so
    /// [`crate::credentials`] never picks this scheme and the whole
    /// `gateway::anthropic` tree that would reach it is staged behind the same
    /// attribute. Both allows come off on the change that adds the roster entry
    /// and the surface — the same change that drops the one at
    /// `gateway/mod.rs`.
    #[allow(dead_code)]
    pub(crate) fn anthropic_api_key(secret: String) -> Self {
        Self::Header(Header::anthropic_api_key(secret))
    }

    /// Applies this credential to a request that is otherwise complete.
    ///
    /// Takes the request by value and hands it back, rather than mutating
    /// through `&mut`: a signing scheme has to read the whole request —
    /// method, URL, headers, body — before it can add anything, and a borrow
    /// that is simultaneously handing out the pieces and taking the header map
    /// is a shape the simple case would be paying for.
    pub(crate) async fn authenticate(&self, request: reqwest::Request) -> Result<reqwest::Request> {
        match self {
            Authenticator::Header(header) => header.apply(request),
            Authenticator::OauthBearer(oauth) => oauth.apply(request),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::applied;
    use super::*;

    #[tokio::test]
    async fn bearer_sets_an_authorization_header() {
        let auth = Authenticator::bearer("sk-openai".into());
        assert_eq!(
            applied(&auth, "authorization").await.as_deref(),
            Some("Bearer sk-openai")
        );
        assert_eq!(applied(&auth, "x-api-key").await, None);
    }

    /// Anthropic reads `x-api-key` and nothing else. An `Authorization` header
    /// alongside it is not merely redundant — a proxy in front of Anthropic may
    /// read one and be answered by the other.
    #[tokio::test]
    async fn an_anthropic_key_never_lands_in_an_authorization_header() {
        let auth = Authenticator::anthropic_api_key("sk-ant-demo".into());
        assert_eq!(
            applied(&auth, "x-api-key").await.as_deref(),
            Some("sk-ant-demo")
        );
        assert_eq!(applied(&auth, "authorization").await, None);
    }

    /// `Debug` is what a panic or a `tracing` field prints. It must name the
    /// header and none of the secret — the enum's derive is only as safe as the
    /// payload's own impl.
    #[test]
    fn debug_redacts_the_secret() {
        let rendered = format!("{:?}", Authenticator::bearer("sk-secret-value".into()));
        assert!(rendered.contains("authorization"), "{rendered}");
        assert!(!rendered.contains("sk-secret-value"), "{rendered}");
    }
}
