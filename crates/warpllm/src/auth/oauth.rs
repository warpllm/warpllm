//! An OAuth bearer credential that caches and refreshes its own token.
//!
//! `OauthBearer` held an already-resolved token and refused to send it
//! past expiry (#72) — deliberately narrow, with minting out of scope.
//! [`OAuth`] is the follow-up: it owns a [`TokenProvider`] and a cache, so
//! a transport calling [`OAuth::apply`] never sees a resolver, only ever a
//! valid header, the same shape [`super::Header`] already has.
//!
//! # Accepting a token
//!
//! A [`Token`] arrives on the wall clock — `expires_at`/`refresh_at` are
//! genuinely wall-clock instants, since that is what the token's issuer
//! means by them. But the question [`OAuth`] actually asks on every call —
//! "has enough time passed that this needs refreshing?" — is about elapsed
//! duration, and `SystemTime` only answers that correctly as long as the
//! system clock never steps backward. [`CachedToken::accept`] converts a
//! [`Token`]'s wall-clock deadlines into `Instant`-based ones once, at the
//! moment it is accepted, so a later NTP correction or a VM resume cannot
//! retroactively extend how long a cached token is treated as valid.
//!
//! `accept` also rejects a [`Token`] whose `refresh_at` is after its own
//! `expires_at` — a contract violation, checked rather than trusted, since
//! [`Token`]'s fields are constructible directly by any provider. A
//! rejected token is treated exactly like a failed mint (see below).
//!
//! # A failed or unusable mint
//!
//! A mint can fail outright, return a token already past its own expiry,
//! violate its own contract, or return a value that cannot become a
//! header. All four are handled identically through [`OAuth::mint`]: none
//! of them touch the cache, and [`OAuth::apply`] falls back to a
//! still-valid cached token if there is one, only propagating the error
//! when there is nothing left to fall back on. This also means a
//! malformed mint can never evict a good cached token — the cache is only
//! ever replaced with a [`CachedToken`] that has already cleared every
//! check, header validation included.
//!
//! # Concurrency
//!
//! The cache is held behind a `tokio::sync::Mutex` for the duration of a
//! refresh (not just the read), so concurrent callers on a due-for-refresh
//! token await one mint rather than each firing their own — a thundering
//! herd against the token endpoint on every simultaneous expiry.

use std::sync::Arc;
use std::time::{Instant, SystemTime};

use reqwest::header::{AUTHORIZATION, HeaderValue};
use tokio::sync::Mutex;

use super::token_provider::{Token, TokenProvider};
use crate::error::{Error, Result};

/// A token whose deadlines have been converted to a monotonic clock at the
/// moment it was accepted. See the module docs for why.
struct CachedToken {
    value: String,
    refresh_at: Instant,
    expires_at: Instant,
}

impl CachedToken {
    /// Converts a wire [`Token`]'s wall-clock deadlines into monotonic ones
    /// anchored to the instant it is accepted. Returns `None` if the token
    /// violates its own contract (`refresh_at` after `expires_at`) rather
    /// than accepting it and letting the violation surface later as a
    /// token sent past its expiry.
    fn accept(token: Token, wall_now: SystemTime) -> Option<Self> {
        if token.refresh_at > token.expires_at {
            return None;
        }
        let anchor = Instant::now();
        let until_refresh = token
            .refresh_at
            .duration_since(wall_now)
            .unwrap_or_default();
        let until_expiry = token
            .expires_at
            .duration_since(wall_now)
            .unwrap_or_default();
        Some(Self {
            value: token.value,
            refresh_at: anchor + until_refresh,
            expires_at: anchor + until_expiry,
        })
    }
}

/// Builds a `Bearer` header value from a token's value, marked sensitive so
/// it never survives into a request's own `Debug`.
fn build_header(value: &str) -> Result<HeaderValue> {
    let mut header = HeaderValue::from_str(&format!("Bearer {value}")).map_err(|_| {
        Error::Internal("the OAuth token holds a byte that cannot be sent in a header".into())
    })?;
    header.set_sensitive(true);
    Ok(header)
}

/// An OAuth credential backed by a [`TokenProvider`], with its own cache.
pub(crate) struct OAuth {
    provider: Arc<dyn TokenProvider>,
    cached: Mutex<Option<CachedToken>>,
}

impl OAuth {
    #[allow(dead_code)]
    pub(crate) fn new(provider: Arc<dyn TokenProvider>) -> Self {
        Self {
            provider,
            cached: Mutex::new(None),
        }
    }

    /// Mints a fresh token and prepares everything needed to accept it,
    /// without touching the cache. Kept separate from `apply` so a bad
    /// mint — a provider error, a contract-violating token, an
    /// already-expired token, or a value that cannot become a header —
    /// never partially commits: the cache is only ever replaced with a
    /// [`CachedToken`] that has cleared every one of these checks.
    async fn mint(&self) -> Result<(CachedToken, HeaderValue)> {
        let token = self.provider.token().await?;
        let wall_now = SystemTime::now();
        let accepted = CachedToken::accept(token, wall_now).ok_or_else(|| {
            Error::Internal(
                "the token provider returned a token whose refresh_at is after its own \
                 expires_at"
                    .into(),
            )
        })?;
        if Instant::now() >= accepted.expires_at {
            return Err(Error::Internal(
                "the token provider returned a token that was already expired".into(),
            ));
        }
        let header = build_header(&accepted.value)?;
        Ok((accepted, header))
    }

    /// Applies a valid `Authorization: Bearer …` header, minting or
    /// reusing a token as needed.
    ///
    /// Holds the lock across the mint, not just the cache check: two
    /// requests racing a due-for-refresh cache must not both call the
    /// provider — the second should find the first's fresh token already
    /// there when it acquires the lock, not mint a redundant one of its
    /// own.
    pub(crate) async fn apply(&self, mut request: reqwest::Request) -> Result<reqwest::Request> {
        let mut cached = self.cached.lock().await;

        let needs_refresh = match &*cached {
            Some(token) => Instant::now() >= token.refresh_at || Instant::now() >= token.expires_at,
            None => true,
        };

        let header = if needs_refresh {
            match self.mint().await {
                Ok((fresh, header)) => {
                    *cached = Some(fresh);
                    header
                }
                // Nothing usable came out of the mint — a network error, a
                // malformed token, or one already dead on arrival. A
                // still-valid cached token can authenticate this request
                // anyway; only propagate when there is nothing left to
                // fall back on.
                Err(mint_error) => {
                    let still_usable = matches!(
                        &*cached,
                        Some(token) if Instant::now() < token.expires_at
                    );
                    if !still_usable {
                        return Err(mint_error);
                    }
                    tracing::warn!(
                        error = %mint_error,
                        "token refresh failed; serving the cached token until it expires"
                    );
                    let token = cached
                        .as_ref()
                        .ok_or_else(|| Error::Internal("cached token cannot be empty".into()))?;
                    build_header(&token.value)?
                }
            }
        } else {
            let token = cached
                .as_ref()
                .ok_or_else(|| Error::Internal("cached token cannot be empty".into()))?;
            build_header(&token.value)?
        };

        request.headers_mut().insert(AUTHORIZATION, header);
        Ok(request)
    }
}

/// The provider only, and only as a fixed placeholder — never delegates to
/// the provider's own `Debug`, which might not exist (`TokenProvider` does
/// not require it) or might not redact what it holds. Never the cached
/// token either, for the same reason [`super::Header`] and the old
/// `OauthBearer` hand-write their own `Debug`.
impl std::fmt::Debug for OAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuth")
            .field("provider", &"<provider>")
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
    /// mint one. Yields once before returning, since a real network mint
    /// always does, and it is that await point that gives concurrent
    /// callers something to interleave around at all.
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

    /// A provider whose first mint is good and due for refresh, and whose
    /// every mint after that returns an already-expired (but otherwise
    /// well-formed) token — for exercising the fallback when a mint
    /// succeeds but produces nothing usable.
    struct ExpiredAfterProvider {
        calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl TokenProvider for ExpiredAfterProvider {
        async fn token(&self) -> Result<Token> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            tokio::task::yield_now().await;
            if call == 0 {
                Ok(Token {
                    value: "ya29.demo-token".into(),
                    expires_at: future(),
                    refresh_at: due_refresh(),
                })
            } else {
                let expired = past();
                Ok(Token {
                    value: "ya29.stale-token".into(),
                    expires_at: expired,
                    refresh_at: expired,
                })
            }
        }
    }

    /// A provider whose first mint is good and due for refresh, and whose
    /// every mint after that returns a value that cannot become a header —
    /// for exercising that a malformed mint neither evicts a still-valid
    /// cache nor wedges the provider out of future retries.
    struct MalformedAfterProvider {
        calls: AtomicUsize,
    }

    impl MalformedAfterProvider {
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl TokenProvider for MalformedAfterProvider {
        async fn token(&self) -> Result<Token> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            tokio::task::yield_now().await;
            if call == 0 {
                Ok(Token {
                    value: "ya29.demo-token".into(),
                    expires_at: future(),
                    refresh_at: due_refresh(),
                })
            } else {
                Ok(Token {
                    value: "ya29.bad\ntoken".into(),
                    expires_at: future(),
                    refresh_at: far_refresh(),
                })
            }
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
    #[test]
    fn token_new_scales_the_margin_to_a_short_lifetime() {
        let token = Token::new(
            "ya29.demo-token".into(),
            SystemTime::now() + Duration::from_secs(60),
        );
        assert!(
            token.refresh_at > SystemTime::now(),
            "a 60s token must not already be due for refresh"
        );
    }

    /// A token whose `expires_at` is already in the past gets a
    /// `refresh_at` no later than that expiry — never past it — rather
    /// than an accidentally negative margin from an unchecked subtraction.
    #[test]
    fn token_new_on_an_already_past_expiry_never_sets_refresh_at_past_expiry() {
        let past = SystemTime::now() - Duration::from_secs(5);
        let token = Token::new("ya29.demo-token".into(), past);
        assert!(token.refresh_at <= token.expires_at);
    }

    /// A provider returning `refresh_at` after `expires_at` violates the
    /// contract [`Token`] documents. The token must be rejected outright
    /// rather than accepted and later sent past its own expiry.
    #[tokio::test]
    async fn a_token_with_refresh_at_after_expires_at_is_rejected() {
        let provider = Arc::new(CountingProvider::new(past(), future()));
        let auth = OAuth::new(provider);
        let error = auth
            .apply(request())
            .await
            .expect_err("an inverted refresh_at/expires_at pair must be rejected");
        assert!(error.to_string().contains("refresh_at"), "{error}");
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
    /// refused, not sent — the same guarantee `OauthBearer` gave (#72), now
    /// covering a bad mint rather than just a stale cache.
    #[tokio::test]
    async fn an_already_expired_minted_token_is_rejected() {
        let expired = past();
        let provider = Arc::new(CountingProvider::new(expired, expired));
        let auth = OAuth::new(provider);
        let error = auth
            .apply(request())
            .await
            .expect_err("an expired mint must not be applied");
        assert!(error.to_string().contains("expired"), "{error}");
    }

    /// The same case as above, but with a still-valid cached token
    /// present: the expired mint must fall back to it rather than fail the
    /// request outright, exactly like a mint that errors.
    #[tokio::test]
    async fn an_expired_mint_falls_back_to_a_still_valid_cache() {
        let provider = Arc::new(ExpiredAfterProvider {
            calls: AtomicUsize::new(0),
        });
        let auth = OAuth::new(provider);

        auth.apply(request()).await.unwrap();
        let signed = auth
            .apply(request())
            .await
            .expect("a still-valid cached token must be applied despite an expired mint");
        assert_eq!(signed.headers()[AUTHORIZATION], "Bearer ya29.demo-token");
    }

    /// A malformed mint must not evict a still-valid cached token, and
    /// must not stop the provider from being asked again on the next call
    /// — the cache is only ever replaced once a candidate has cleared
    /// every check, header validation included.
    #[tokio::test]
    async fn a_malformed_mint_does_not_evict_the_cache_or_wedge_the_provider() {
        let provider = Arc::new(MalformedAfterProvider {
            calls: AtomicUsize::new(0),
        });
        let auth = OAuth::new(provider.clone());

        auth.apply(request()).await.unwrap();
        assert_eq!(provider.calls(), 1);

        let signed = auth
            .apply(request())
            .await
            .expect("a still-valid cached token must be applied despite the malformed mint");
        assert_eq!(signed.headers()[AUTHORIZATION], "Bearer ya29.demo-token");
        assert_eq!(provider.calls(), 2);

        let signed = auth.apply(request()).await.expect(
            "still malformed, but the provider must still be retried, not permanently wedged",
        );
        assert_eq!(signed.headers()[AUTHORIZATION], "Bearer ya29.demo-token");
        assert_eq!(
            provider.calls(),
            3,
            "a malformed mint must not wedge the provider out of future retries"
        );
    }

    /// `Debug` on the credential itself never touches the provider's own
    /// `Debug` — `TokenProvider` does not require the bound, and even a
    /// provider that derives it must not have that derive reach the
    /// caller through `OAuth`.
    #[tokio::test]
    async fn debug_never_delegates_to_the_providers_own_debug() {
        let provider = Arc::new(CountingProvider::new(future(), far_refresh()));
        let auth = OAuth::new(provider);
        auth.apply(request()).await.unwrap();
        let rendered = format!("{auth:?}");
        assert!(rendered.contains("<provider>"), "{rendered}");
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
