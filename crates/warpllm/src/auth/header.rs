//! A secret sent in a named header — the whole of authentication for every
//! provider on the roster today.
//!
//! One type for two spellings. `Authorization: Bearer sk-…` and Anthropic's
//! `x-api-key: sk-ant-…` are the same act with a different header name and a
//! different prefix, so they are two constructors rather than two variants; an
//! Azure-style `api-key:` would be a third and add no code. What they are NOT
//! is interchangeable: Anthropic reads `x-api-key` and OpenAI-compatible hosts
//! read `Authorization`, so the constructor a provider gets is a fact about
//! that provider, decided by [`crate::credentials`].

use reqwest::header::{AUTHORIZATION, HeaderName, HeaderValue};

use crate::error::{Error, Result};

/// A secret presented as `{prefix}{secret}` under one header name.
pub(crate) struct Header {
    name: HeaderName,
    /// What precedes the secret in the header value. `"Bearer "` for
    /// [`Header::bearer`], empty everywhere else — a scheme token belongs to
    /// the header's grammar, not to the key.
    prefix: &'static str,
    secret: String,
}

impl Header {
    /// `Authorization: Bearer <secret>`.
    pub(crate) fn bearer(secret: String) -> Self {
        Self {
            name: AUTHORIZATION,
            prefix: "Bearer ",
            secret,
        }
    }

    /// `x-api-key: <secret>`, and deliberately no `Authorization` alongside it.
    ///
    /// Unreachable outside tests until a provider routes to Anthropic — see
    /// [`Authenticator::anthropic_api_key`](super::Authenticator::anthropic_api_key).
    #[allow(dead_code)]
    pub(crate) fn anthropic_api_key(secret: String) -> Self {
        Self {
            name: HeaderName::from_static("x-api-key"),
            prefix: "",
            secret,
        }
    }

    /// Writes the header onto a built request.
    ///
    /// `insert` rather than `append`: a credential replaces whatever was there,
    /// so a transport that set one by hand cannot end up sending two.
    ///
    /// The value is marked SENSITIVE, which is what keeps the secret out of
    /// [`reqwest::Request`]'s own `Debug` — a request is a natural thing to log
    /// when one fails — and out of the HPACK index on an HTTP/2 connection.
    /// `reqwest::RequestBuilder::bearer_auth`, which this replaces, did the
    /// same; the Anthropic transport's hand-written `.header("x-api-key", …)`
    /// did not, and now does.
    pub(crate) fn apply(&self, mut request: reqwest::Request) -> Result<reqwest::Request> {
        let mut value = HeaderValue::from_str(&format!("{}{}", self.prefix, self.secret))
            // The secret itself is NOT in the message. A key with a stray
            // newline is the usual cause, and naming the header is enough to
            // find it.
            .map_err(|_| {
                Error::Internal(format!(
                    "the credential for `{}` holds a byte that cannot be sent in a header",
                    self.name
                ))
            })?;
        value.set_sensitive(true);
        request.headers_mut().insert(self.name.clone(), value);
        Ok(request)
    }
}

/// The header name, never its value. A derived `Debug` prints the API key the
/// moment a [`Header`] reaches a `tracing` field or a panic message — the same
/// hazard [`crate::credentials::Credentials`] hand-writes its own `Debug` for.
impl std::fmt::Debug for Header {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Header")
            .field("name", &self.name)
            .field("secret", &"<redacted>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> reqwest::Request {
        reqwest::Client::new()
            .post("https://example.invalid/v1/chat/completions")
            .body("{}")
            .build()
            .expect("a well-formed request")
    }

    /// The prefix belongs to the header's grammar, so it is present for bearer
    /// and absent for a bare key — with no space left behind by the empty one.
    #[test]
    fn the_prefix_is_part_of_the_value_and_only_where_it_belongs() {
        let signed = Header::bearer("sk-openai".into()).apply(request()).unwrap();
        assert_eq!(signed.headers()[AUTHORIZATION], "Bearer sk-openai");

        let signed = Header::anthropic_api_key("sk-ant-demo".into())
            .apply(request())
            .unwrap();
        assert_eq!(signed.headers()["x-api-key"], "sk-ant-demo");
    }

    /// A second application replaces the first. Nothing does this today, and
    /// the alternative — two `Authorization` headers, one stale — is the kind
    /// of failure a provider reports as a plain 401.
    #[test]
    fn applying_twice_leaves_one_header() {
        let once = Header::bearer("first".into()).apply(request()).unwrap();
        let twice = Header::bearer("second".into()).apply(once).unwrap();
        assert_eq!(twice.headers().get_all(AUTHORIZATION).iter().count(), 1);
        assert_eq!(twice.headers()[AUTHORIZATION], "Bearer second");
    }

    /// A key with a newline in it is a configuration mistake, and it must fail
    /// as one rather than as a 401 from the provider. The message names the
    /// header and never the value.
    #[test]
    fn a_secret_that_cannot_be_a_header_value_is_rejected() {
        let error = Header::bearer("sk-with-a\nnewline".into())
            .apply(request())
            .expect_err("a newline cannot travel in a header");
        let rendered = error.to_string();
        assert!(rendered.contains("authorization"), "{rendered}");
        assert!(!rendered.contains("sk-with-a"), "{rendered}");
    }

    /// A request is a natural thing to log when one fails, and its `Debug`
    /// prints every header. The sensitive flag is what stops that being a leak.
    #[test]
    fn the_secret_does_not_survive_into_the_requests_own_debug() {
        let signed = Header::bearer("sk-secret-value".into())
            .apply(request())
            .unwrap();
        let rendered = format!("{signed:?}");
        assert!(!rendered.contains("sk-secret-value"), "{rendered}");
    }

    /// `Debug` on the credential itself: the header, none of the secret.
    #[test]
    fn debug_redacts_the_secret() {
        let rendered = format!("{:?}", Header::bearer("sk-secret-value".into()));
        assert!(rendered.contains("authorization"), "{rendered}");
        assert!(!rendered.contains("sk-secret-value"), "{rendered}");
    }
}
