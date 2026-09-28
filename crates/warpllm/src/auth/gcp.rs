//! Application Default Credentials, wrapped as a [`TokenProvider`].
//!
//! `gcp_auth` walks the ADC chain -- a service account key file, workload
//! identity, `gcloud`'s own cached login, or the GCE/Cloud Run metadata
//! server -- and hands back whichever source applies. This module only
//! adapts its result into warpllm's own [`Token`] shape; the chain-walking
//! and the refresh-token exchange are `gcp_auth`'s job, not this crate's --
//! see [`super::token_provider`]'s module docs for why that division holds.
//!
//! Fully qualified (`gcp_auth::TokenProvider`, `gcp_auth::Token`)
//! throughout, never `use`d unqualified: `gcp_auth` ships its own
//! `TokenProvider` trait, and colliding the two by name in one file would
//! make every call site ambiguous about which one is meant.

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use tokio::sync::OnceCell;

use super::token_provider::{Token, TokenProvider};
use crate::error::{Error, Result};

/// The one OAuth scope every Vertex surface (Anthropic Messages and
/// Google's own `generateContent`) is reached under.
const CLOUD_PLATFORM_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

/// Resolves Application Default Credentials lazily: the chain is only
/// walked on the first real [`TokenProvider::token`] call, not at
/// construction. This keeps [`crate::credentials::Credentials::resolve`]
/// synchronous -- a Vertex entry in a caller's roster must not force
/// `Client::new` to block on network I/O or filesystem probing that no
/// other provider needs.
pub(crate) struct GcpTokenProvider {
    inner: OnceCell<Arc<dyn gcp_auth::TokenProvider>>,
}

impl GcpTokenProvider {
    pub(crate) fn new() -> Self {
        Self {
            inner: OnceCell::new(),
        }
    }

    /// Bypasses ADC resolution entirely by pre-filling the cell with a
    /// caller-supplied provider — for tests that need to exercise a mint's
    /// success or failure without a real ADC chain. `inner()` never calls
    /// `gcp_auth::provider()` when the cell is already populated, so this
    /// only reaches the mint path, never the resolution-itself-failing path.
    #[cfg(test)]
    pub(crate) fn with_provider(provider: Arc<dyn gcp_auth::TokenProvider>) -> Self {
        Self {
            inner: OnceCell::new_with(Some(provider)),
        }
    }

    /// Resolves the ADC chain on first call, memoized after. A failure
    /// here is never cached as a failure -- the next call tries again,
    /// since a transient cause (metadata server briefly unreachable) should
    /// not wedge every subsequent request behind the first one's bad luck.
    async fn inner(&self) -> Result<&Arc<dyn gcp_auth::TokenProvider>> {
        self.inner
            .get_or_try_init(|| async {
                gcp_auth::provider()
                    .await
                    .map_err(|err| Error::CredentialResolutionFailed {
                        provider: "vertex",
                        message: err.to_string(),
                    })
            })
            .await
    }
}

#[async_trait::async_trait]
impl TokenProvider for GcpTokenProvider {
    async fn token(&self) -> Result<Token> {
        let inner = self.inner().await?;
        let token = inner.token(&[CLOUD_PLATFORM_SCOPE]).await.map_err(|err| {
            Error::CredentialResolutionFailed {
                provider: "vertex",
                message: err.to_string(),
            }
        })?;
        Ok(Token::new(token.as_str().to_string(), gcp_expiry(&token)))
    }
}

/// `gcp_auth`'s current `Token::expires_at()` returns a concrete
/// `DateTime<Utc>`, not an `Option` -- every ADC source it uses genuinely
/// expires, so this never actually returns `None` in practice. Still
/// converted defensively rather than trusted blindly: a timestamp this
/// platform's `SystemTime` cannot represent is treated as no expiry, the
/// same policy [`Token`] gives a source that reports none at all, rather
/// than panicking on an edge value neither side controls.
fn gcp_expiry(token: &gcp_auth::Token) -> Option<SystemTime> {
    let seconds: u64 = token.expires_at().timestamp().try_into().ok()?;
    SystemTime::UNIX_EPOCH.checked_add(Duration::from_secs(seconds))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc as StdArc;

    use super::*;

    /// A [`gcp_auth::TokenProvider`] that hands back a fixed, well-formed
    /// token built the same way gcp_auth's own tests build one --
    /// deserialized from the wire shape, since `gcp_auth::Token` has no
    /// public constructor.
    struct FakeGcpProvider {
        access_token: &'static str,
        expires_in_secs: u64,
    }

    #[async_trait::async_trait]
    impl gcp_auth::TokenProvider for FakeGcpProvider {
        async fn token(
            &self,
            _scopes: &[&str],
        ) -> std::result::Result<StdArc<gcp_auth::Token>, gcp_auth::Error> {
            let json = format!(
                r#"{{"access_token":"{}","expires_in":{}}}"#,
                self.access_token, self.expires_in_secs
            );
            let token: gcp_auth::Token =
                serde_json::from_str(&json).expect("well-formed test token JSON must deserialize");
            Ok(StdArc::new(token))
        }

        async fn project_id(&self) -> std::result::Result<StdArc<str>, gcp_auth::Error> {
            Ok(StdArc::from("test-project"))
        }
    }

    /// A [`gcp_auth::TokenProvider`] whose mint always fails -- for
    /// exercising the mint-failure path without a real ADC dependency.
    struct FailingGcpProvider;

    #[async_trait::async_trait]
    impl gcp_auth::TokenProvider for FailingGcpProvider {
        async fn token(
            &self,
            _scopes: &[&str],
        ) -> std::result::Result<StdArc<gcp_auth::Token>, gcp_auth::Error> {
            Err(gcp_auth::Error::Str(
                "no available authentication method found",
            ))
        }

        async fn project_id(&self) -> std::result::Result<StdArc<str>, gcp_auth::Error> {
            Err(gcp_auth::Error::Str(
                "no available authentication method found",
            ))
        }
    }

    /// A successful mint through an injected provider produces a
    /// [`Token`] carrying the minted value and a `Some` expiry derived
    /// from `gcp_auth`'s own `expires_at()`.
    #[tokio::test]
    async fn a_successful_mint_produces_a_token_with_the_minted_value() {
        let provider = GcpTokenProvider::with_provider(StdArc::new(FakeGcpProvider {
            access_token: "ya29.fake-vertex-token",
            expires_in_secs: 3600,
        }));
        let token = provider.token().await.expect("a fake mint must succeed");
        assert_eq!(token.value, "ya29.fake-vertex-token");
        assert!(
            token.expires_at.is_some(),
            "a reported expiry must survive as Some"
        );
    }

    /// A mint failure from the underlying provider surfaces as
    /// [`Error::CredentialResolutionFailed`], naming "vertex" -- the same
    /// failure path a bad service account file or unreachable metadata
    /// server would take in production.
    #[tokio::test]
    async fn a_failed_mint_surfaces_as_credential_resolution_failed() {
        let provider = GcpTokenProvider::with_provider(StdArc::new(FailingGcpProvider));
        // Not expect_err/unwrap_err: both require the Ok type (Token) to
        // implement Debug, which it deliberately does not -- Token holds
        // the secret itself, and this crate never gives it a Debug impl
        // to print. Matching directly avoids needing one just for a test.
        match provider.token().await {
            Ok(_) => panic!("a failing underlying provider must not produce a token"),
            Err(Error::CredentialResolutionFailed { provider, .. }) => {
                assert_eq!(provider, "vertex");
            }
            Err(other) => panic!("expected CredentialResolutionFailed, got {other}"),
        }
    }
}
