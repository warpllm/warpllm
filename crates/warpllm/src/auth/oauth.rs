//! An OAuth bearer credential that caches and refreshes its own token.
//!
//! `OauthBearer` held an already-resolved token and refused to send it
//! past expiry (#72) — deliberately narrow, with minting out of scope.
//! [`OAuth`] is the follow-up: it owns a [`TokenProvider`] and a cache, so
//! a transport calling [`OAuth::apply`] never sees a resolver, only ever a
//! valid header, the same shape [`super::Header`] already has.
//!
//! Refresh happens at the token's own `refresh_at`, not at expiry, so a
//! request built while the cached token is still nominally valid does not
//! race a Vertex-side clock skew or the request's own in-flight time. The
//! cache is held behind a `tokio::sync::Mutex` for the duration of a
//! refresh (not just the read), so concurrent callers on a due-for-refresh
//! token await one mint rather than each firing their own — a thundering
//! herd against the token endpoint on every simultaneous expiry.
//!
//! A failed mint does not discard a cached token that is still valid: the
//! refresh margin exists to buy slack for exactly this, and a token-endpoint
//! blip must not fail a request the still-cached token could authenticate.
//! Only a mint with nothing usable to fall back on propagates its error.

use std::sync::Arc;
use std::time::SystemTime;

use reqwest::header::{AUTHORIZATION, HeaderValue};
use tokio::sync::Mutex;

use super::token_provider::{Token, TokenProvider};
use crate::error::{Error, Result};

/// An OAuth credential backed by a [`TokenProvider`], with its own cache.
pub(crate) struct OAuth {
    provider: Arc<dyn TokenProvider>,
    cached: Mutex<Option<Token>>,
}

impl OAuth {
    #[allow(dead_code)]
    pub(crate) fn new(provider: Arc<dyn TokenProvider>) -> Self {
        Self {
            provider,
            cached: Mutex::new(None),
        }
    }

    /// Applies a valid `Authorization: Bearer …` header, minting or
    /// reusing a token as needed.
    ///
    /// Holds the lock across the mint, not just the cache check: two
    /// requests racing a due-for-refresh cache must not both call the
    /// provider — the second should find the first's fresh token already
    /// there when it acquires the lock, not mint a redundant one of its own.
    pub(crate) async fn apply(&self, mut request: reqwest::Request) -> Result<reqwest::Request> {
        let mut cached = self.cached.lock().await;

        let needs_refresh = match &*cached {
            Some(token) => SystemTime::now() >= token.refresh_at,
            None => true,
        };

        if needs_refresh {
            match self.provider.token().await {
                Ok(fresh) => {
                    if SystemTime::now() >= fresh.expires_at {
                        return Err(Error::Internal(
                            "the token provider returned a token that was already expired".into(),
                        ));
                    }
                    *cached = Some(fresh);
                }
                // Due for refresh is not the same as invalid. A cached
                // token still short of its own expiry can authenticate
                // this request even though the mint that would have
                // replaced it failed — the margin exists to buy exactly
                // this slack.
                Err(mint_error) => {
                    let still_usable = matches!(
                        &*cached,
                        Some(token) if SystemTime::now() < token.expires_at
                    );
                    if !still_usable {
                        return Err(mint_error);
                    }
                }
            }
        }

        let token = cached
            .as_ref()
            .ok_or_else(|| Error::Internal("cached token cannot be empty".into()))?;

        let mut value =
            HeaderValue::from_str(&format!("Bearer {}", token.value)).map_err(|_| {
                Error::Internal(
                    "the OAuth token holds a byte that cannot be sent in a header".into(),
                )
            })?;
        value.set_sensitive(true);
        request.headers_mut().insert(AUTHORIZATION, value);
        Ok(request)
    }
}

/// The provider only, never the cached token — same hazard [`super::Header`]
/// and the old `OauthBearer` hand-write their own `Debug` for. `Token`
/// itself has no `Debug`, so a derive here could not compile even by
/// accident; this impl exists to name that deliberately rather than leave
/// the type un-`Debug`-able with no explanation.
impl std::fmt::Debug for OAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuth")
            .field("provider", &self.provider)
            .field("cached", &"<redacted>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    fn request() -> reqwest::Request {
        reqwest::Client::new()
            .post("https://example.invalid/v1/models/gemini:generateContent")
            .body("{}")
            .build()
            .expect("a well-formed request")
    }

    fn future() -> SystemTime {
        SystemTime::now() + Duration::from_secs(3600)
    }

    fn past() -> SystemTime {
        SystemTime::now() - Duration::from_secs(1)
    }

    /// Far enough out that a token carrying it as `refresh_at` reads as
    /// fresh — the zone a live cache hit should stay in.
    fn far_refresh() -> SystemTime {
        SystemTime::now() + Duration::from_secs(3300)
    }

    /// Already due for refresh, independent of when the token actually
    /// expires — lets a test drive the refresh path without waiting on a
    /// real margin.
    fn due_refresh() -> SystemTime {
        SystemTime::now() - Duration::from_secs(1)
    }

    /// A [`TokenProvider`] that hands back a fixed token on an explicit
    /// `expires_at`/`refresh_at`, and counts how many times it was asked to
    /// mint one — the only way to observe caching and thundering-herd
    /// behaviour from outside [`OAuth`] itself. Yields once before
    /// returning, since a real network mint always does, and it is that
    /// await point that gives concurrent callers something to interleave
    /// around at all.
    #[derive(Debug)]
    struct CountingProvider {
        calls: AtomicUsize,
        expires_at: SystemTime,
        refresh_at: SystemTime,
    }

    impl CountingProvider {
        fn new(expires_at: SystemTime, refresh_at: SystemTime) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                expires_at,
                refresh_at,
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl TokenProvider for CountingProvider {
        async fn token(&self) -> Result<Token> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            tokio::task::yield_now().await;
            Ok(Token {
                value: "ya29.demo-token".into(),
                expires_at: self.expires_at,
                refresh_at: self.refresh_at,
            })
        }
    }

    /// A provider whose mint always fails — for exercising the fallback
    /// and error paths without a real network dependency.
    #[derive(Debug)]
    struct FailingProvider;

    #[async_trait::async_trait]
    impl TokenProvider for FailingProvider {
        async fn token(&self) -> Result<Token> {
            Err(Error::Internal("token endpoint unavailable".into()))
        }
    }

    /// A provider whose first `fail_after` mints succeed and every one
    /// after that fails — for exercising a refresh that fails only once
    /// there is already something cached to fall back on.
    #[derive(Debug)]
    struct FlakyProvider {
        calls: AtomicUsize,
        expires_at: SystemTime,
        refresh_at: SystemTime,
        fail_after: usize,
    }

    #[async_trait::async_trait]
    impl TokenProvider for FlakyProvider {
        async fn token(&self) -> Result<Token> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            tokio::task::yield_now().await;
            if call >= self.fail_after {
                return Err(Error::Internal("token endpoint unavailable".into()));
            }
            Ok(Token {
                value: "ya29.demo-token".into(),
                expires_at: self.expires_at,
                refresh_at: self.refresh_at,
            })
        }
    }

    /// A fresh mint is applied as a standard bearer header — the same wire
    /// shape [`super::super::Header`] and the old `OauthBearer` produced.
    #[tokio::test]
    async fn a_minted_token_sets_an_authorization_header() {
        let provider = Arc::new(CountingProvider::new(future(), far_refresh()));
        let auth = OAuth::new(provider);
        let signed = auth.apply(request()).await.unwrap();
        assert_eq!(signed.headers()[AUTHORIZATION], "Bearer ya29.demo-token");
    }

    /// A cached token whose `refresh_at` is still in the future is reused,
    /// not re-minted — the whole point of holding a cache instead of
    /// calling through on every request.
    #[tokio::test]
    async fn a_fresh_cached_token_is_not_refreshed() {
        let provider = Arc::new(CountingProvider::new(future(), far_refresh()));
        let auth = OAuth::new(provider.clone());
        auth.apply(request()).await.unwrap();
        auth.apply(request()).await.unwrap();
        assert_eq!(provider.calls(), 1, "a live cache hit must not re-mint");
    }

    /// A token past its own `refresh_at` is re-minted before it is
    /// applied — proactive, not reactive to actual expiry.
    #[tokio::test]
    async fn a_token_past_its_refresh_point_is_refreshed() {
        let provider = Arc::new(CountingProvider::new(future(), due_refresh()));
        let auth = OAuth::new(provider.clone());
        auth.apply(request()).await.unwrap();
        auth.apply(request()).await.unwrap();
        assert_eq!(
            provider.calls(),
            2,
            "a token past its own refresh_at must be re-minted on every call"
        );
    }

    /// [`Token::new`] scales the margin to the token's own lifetime, so a
    /// short-lived token is not treated as due for refresh the instant it
    /// is minted — the failure mode a flat margin would otherwise cause.
    #[tokio::test]
    async fn token_new_scales_the_margin_to_a_short_lifetime() {
        let token = Token::new(
            "ya29.demo-token".into(),
            SystemTime::now() + Duration::from_secs(60),
        );
        assert!(
            token.refresh_at > SystemTime::now(),
            "a 60s token must not already be due for refresh"
        );
    }

    /// Two callers racing a cache with nothing usable in it must not both
    /// mint. The lock is held across the mint itself, not just the read,
    /// so the second caller finds the first one's fresh token already
    /// there.
    #[tokio::test]
    async fn concurrent_callers_on_an_empty_cache_mint_only_once() {
        let provider = Arc::new(CountingProvider::new(future(), far_refresh()));
        let auth = Arc::new(OAuth::new(provider.clone()));

        let (a, b) = tokio::join!(
            {
                let auth = auth.clone();
                async move { auth.apply(request()).await }
            },
            {
                let auth = auth.clone();
                async move { auth.apply(request()).await }
            },
        );
        a.unwrap();
        b.unwrap();

        assert_eq!(provider.calls(), 1, "a thundering herd must mint once");
    }

    /// A refresh that fails does not discard a cached token that is still
    /// short of its own expiry — the request still succeeds on the token
    /// already held, rather than failing on a transient endpoint blip.
    #[tokio::test]
    async fn a_failed_refresh_keeps_a_still_valid_cached_token() {
        let provider = Arc::new(FlakyProvider {
            calls: AtomicUsize::new(0),
            expires_at: future(),
            refresh_at: due_refresh(),
            fail_after: 1,
        });
        let auth = OAuth::new(provider);

        auth.apply(request()).await.unwrap();
        let signed = auth
            .apply(request())
            .await
            .expect("a still-valid cached token must be applied despite the failed refresh");
        assert_eq!(signed.headers()[AUTHORIZATION], "Bearer ya29.demo-token");
    }

    /// A refresh that fails with nothing cached to fall back on propagates
    /// its error rather than manufacturing a token.
    #[tokio::test]
    async fn a_failed_refresh_with_no_cache_returns_the_error() {
        let auth = OAuth::new(Arc::new(FailingProvider));
        let error = auth
            .apply(request())
            .await
            .expect_err("nothing to fall back on must surface the mint error");
        assert!(error.to_string().contains("unavailable"), "{error}");
    }

    /// A provider that hands back a token already past its own expiry is
    /// refused, not sent — the same guarantee `OauthBearer` gave (#72),
    /// now covering a bad mint rather than just a stale cache.
    #[tokio::test]
    async fn an_already_expired_minted_token_is_rejected() {
        let provider = Arc::new(CountingProvider::new(past(), past()));
        let auth = OAuth::new(provider);
        let error = auth
            .apply(request())
            .await
            .expect_err("an expired mint must not be applied");
        assert!(error.to_string().contains("expired"), "{error}");
    }

    /// `Debug` on the credential itself: never the cached token, and no
    /// need to have minted one first — the same guarantee `OauthBearer`
    /// gave before it.
    #[tokio::test]
    async fn debug_redacts_the_token_after_a_mint() {
        let provider = Arc::new(CountingProvider::new(future(), far_refresh()));
        let auth = OAuth::new(provider);
        auth.apply(request()).await.unwrap();
        let rendered = format!("{auth:?}");
        assert!(!rendered.contains("ya29.demo-token"), "{rendered}");
    }

    /// A request is a natural thing to log when one fails, and its `Debug`
    /// prints every header. The sensitive flag is what stops that being a
    /// leak, the same guarantee `Header::apply` and `OauthBearer` gave.
    #[tokio::test]
    async fn the_token_does_not_survive_into_the_requests_own_debug() {
        let provider = Arc::new(CountingProvider::new(future(), far_refresh()));
        let auth = OAuth::new(provider);
        let signed = auth.apply(request()).await.unwrap();
        let rendered = format!("{signed:?}");
        assert!(!rendered.contains("ya29.demo-token"), "{rendered}");
    }
}
