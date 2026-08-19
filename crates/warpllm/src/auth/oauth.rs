//! An OAuth bearer token from Google's Application Default Credentials.
//!
//! Distinct from [`super::Header`]'s static secret: the token this holds
//! expires, and applying it after expiry would send a request Vertex is
//! guaranteed to reject with a 401. [`OauthBearer::apply`] checks the expiry
//! itself rather than leaving that to the caller, since a transport has no
//! way to know a token's lifetime from the outside.
//!
//! Holds the RESOLVED token, never a resolver — see the module docs on
//! [`super::Authenticator`]. Minting and refreshing a token from ADC (a
//! service account key, workload identity, or `gcloud` login) is deliberately
//! out of scope here; that is its own change, tracked against #25. A
//! [`OauthBearer`] built in a test is two literals and cannot reach
//! `~/.config/gcloud`, an instance metadata server, or the network at all.

use std::time::SystemTime;

use reqwest::header::{AUTHORIZATION, HeaderValue};

use crate::error::{Error, Result};

/// A resolved OAuth access token and the instant it stops being valid.
pub(crate) struct OauthBearer {
    token: String,
    expires_at: SystemTime,
}

impl OauthBearer {
    /// Reads as dead outside tests until ADC token-minting exists to call
    /// this — the same reason the `OauthBearer` variant itself is marked,
    /// see [`super::Authenticator::OauthBearer`].
    #[allow(dead_code)]
    pub(crate) fn new(token: String, expires_at: SystemTime) -> Self {
        Self { token, expires_at }
    }

    /// Applies the token if it is still valid; refuses to send an expired one.
    ///
    /// Sending an expired token is not a network problem — it just trades a
    /// clear local error for the same 401 Vertex would return anyway, with
    /// less context about what actually needs fixing. Checking here names
    /// the token as the cause, not the request.
    pub(crate) fn apply(&self, mut request: reqwest::Request) -> Result<reqwest::Request> {
        if SystemTime::now() >= self.expires_at {
            return Err(Error::Internal(
                "OAuth token expired before the request could be sent; refresh is not \
                 yet implemented (#25)"
                    .into(),
            ));
        }
        let mut value = HeaderValue::from_str(&format!("Bearer {}", self.token)).map_err(|_| {
            Error::Internal("the OAuth token holds a byte that cannot be sent in a header".into())
        })?;
        value.set_sensitive(true);
        request.headers_mut().insert(AUTHORIZATION, value);
        Ok(request)
    }
}

/// The expiry only, never the token. A derived `Debug` prints the secret the
/// moment an [`OauthBearer`] reaches a `tracing` field or a panic message —
/// the same hazard [`super::Header`] hand-writes its own `Debug` for.
impl std::fmt::Debug for OauthBearer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OauthBearer")
            .field("expires_at", &self.expires_at)
            .field("token", &"<redacted>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
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

    /// A token that has not yet expired is applied as a standard bearer
    /// header — Vertex reads `Authorization` the same way an OpenAI-compatible
    /// host does.
    #[test]
    fn a_valid_token_sets_an_authorization_header() {
        let auth = OauthBearer::new("ya29.demo-token".into(), future());
        let signed = auth.apply(request()).unwrap();
        assert_eq!(signed.headers()[AUTHORIZATION], "Bearer ya29.demo-token");
    }

    /// An expired token is refused rather than sent. The alternative is a
    /// 401 from Vertex that looks identical to a bad token, with none of the
    /// context this error carries.
    #[test]
    fn an_expired_token_is_rejected_before_it_is_sent() {
        let auth = OauthBearer::new("ya29.demo-token".into(), past());
        let error = auth
            .apply(request())
            .expect_err("an expired token must not be applied");
        let rendered = error.to_string();
        assert!(rendered.contains("expired"), "{rendered}");
        assert!(!rendered.contains("ya29.demo-token"), "{rendered}");
    }

    /// A token expiring at exactly `now` is treated as already expired, not
    /// as valid for one more instant — `>=`, not `>`, so there is no window
    /// where a request can be built and sent on a token that just lapsed.
    #[test]
    fn a_token_expiring_at_now_is_treated_as_expired() {
        let auth = OauthBearer::new("ya29.demo-token".into(), SystemTime::now());
        assert!(auth.apply(request()).is_err());
    }

    /// A request is a natural thing to log when one fails, and its `Debug`
    /// prints every header. The sensitive flag is what stops that being a
    /// leak, the same guarantee `Header::apply` gives.
    #[test]
    fn the_token_does_not_survive_into_the_requests_own_debug() {
        let signed = OauthBearer::new("ya29.demo-token".into(), future())
            .apply(request())
            .unwrap();
        let rendered = format!("{signed:?}");
        assert!(!rendered.contains("ya29.demo-token"), "{rendered}");
    }

    /// `Debug` on the credential itself: the expiry, none of the token.
    #[test]
    fn debug_redacts_the_token() {
        let rendered = format!("{:?}", OauthBearer::new("ya29.demo-token".into(), future()));
        assert!(rendered.contains("expires_at"), "{rendered}");
        assert!(!rendered.contains("ya29.demo-token"), "{rendered}");
    }
}
