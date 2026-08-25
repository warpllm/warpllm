//! A source of OAuth access tokens, abstracted over how they're minted.
//!
//! Distinct from [`super::Header`]'s closed set of static schemes: OAuth
//! genuinely has more than one source — a GCP service account key, workload
//! identity, `gcloud` ADC, and eventually Azure/Entra — and each mints and
//! refreshes a token differently. A trait, not another closed enum, is what
//! keeps that openness from leaking into [`super::Authenticator`] itself:
//! the enum stays closed (`Header`, `OAuth`, eventually `SigV4`), and only
//! the `OAuth` variant's payload is where per-source variation lives.
//!
//! `async-trait` is used here specifically because this is the one place
//! dynamic dispatch is required: [`super::OAuth`] needs to hold whichever
//! concrete provider a client is configured for, chosen at runtime, and
//! native `async fn` in traits is not yet `dyn`-compatible. Every other
//! credential in this module keeps its zero-cost static path; this is a
//! deliberate, scoped exception — see the design discussion on #25.

use std::time::{Duration, SystemTime};

use crate::error::Result;

/// The default ceiling on how long before expiry a token is refreshed.
/// [`Token::new`] scales this down for a token whose own lifetime is
/// shorter, so a short-lived token is not treated as stale the instant
/// it is minted.
#[allow(dead_code)]
const DEFAULT_REFRESH_MARGIN: Duration = Duration::from_secs(300);

/// A resolved token, when it stops being valid, and when it should be
/// refreshed. `refresh_at` is deliberately its own field rather than a flat
/// margin subtracted from `expires_at` at the call site: a source that
/// knows its own refresh semantics (a metadata server's own cache lifetime,
/// a service-account JWT's issued-at) can state `refresh_at` directly
/// instead of leaving [`super::OAuth`] to infer it.
pub(crate) struct Token {
    pub(crate) value: String,
    pub(crate) expires_at: SystemTime,
    pub(crate) refresh_at: SystemTime,
}

impl Token {
    /// Builds a token with `refresh_at` scaled to its own lifetime: up to
    /// [`DEFAULT_REFRESH_MARGIN`] before expiry, but never past the token's
    /// own halfway point. A provider whose token lives 60 seconds should
    /// not be treated as needing refresh from the instant it is minted,
    /// which a flat 300-second margin would otherwise force.
    ///
    /// A provider that already knows its own refresh semantics (a
    /// metadata server's cache lifetime, a JWT's issued-at) should build
    /// [`Token`] directly with an explicit `refresh_at` instead.
    #[allow(dead_code)]
    pub(crate) fn new(value: String, expires_at: SystemTime) -> Self {
        let now = SystemTime::now();
        let lifetime = expires_at.duration_since(now).unwrap_or_default();
        let margin = DEFAULT_REFRESH_MARGIN.min(lifetime / 2);
        let refresh_at = expires_at.checked_sub(margin).unwrap_or(now);
        Self {
            value,
            expires_at,
            refresh_at,
        }
    }
}

/// Mints or refreshes an OAuth access token for one credential source.
///
/// Implementors do the actual minting — reading a service account key,
/// talking to a metadata server, whatever a given source's flow requires.
/// This trait only asks for the result: a valid token, or an error naming
/// what went wrong. Caching, refresh timing, and the request-facing
/// `apply()` step live in [`super::OAuth`], not inside an implementor.
///
/// Requires `Debug`, but implementors must hand-write it — same discipline
/// as [`super::Header`] and the old `OauthBearer` before it — so a cached
/// or in-flight token never leaks into a panic message or a `tracing` field.
#[async_trait::async_trait]
pub(crate) trait TokenProvider: std::fmt::Debug + Send + Sync {
    async fn token(&self) -> Result<Token>;
}
