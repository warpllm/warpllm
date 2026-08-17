//! Client configuration.

use std::collections::BTreeMap;

use serde::Deserialize;

/// Matches the OpenAI SDK's default request timeout.
pub(crate) const DEFAULT_TIMEOUT_SECS: u64 = 600;

/// Asks for no API key, and takes one where a caller has nowhere else to put
/// it. Left alone, a client reads its keys from the environment when it is
/// built, one variable per provider the roster names — so a client is never
/// asked up front for keys it can find, nor for keys a given request will not
/// use.
///
/// [`providers`](Self::providers) is the other half, for the embedder this
/// struct was always going to have to serve: one holding keys somewhere the
/// process environment cannot reach, and one that knows which providers it
/// means to talk to.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClientConfig {
    /// Overrides the provider's default base URL (proxies, tests). Absent
    /// means each provider talks to its own API.
    pub base_url: Option<String>,
    pub timeout_secs: Option<u64>,
    /// How long a stream may go without a single byte before warpllm gives up
    /// on it. Absent means never, which is the default.
    ///
    /// [`timeout_secs`](Self::timeout_secs) is a TOTAL deadline — it cannot
    /// tell a stream that is alive and slow from one that is wedged, so it
    /// bounds a stall only by outliving it. This bounds the GAP instead, and
    /// resets on every byte, which is the shape that fits a response whose
    /// length nobody knows in advance.
    ///
    /// Opt-in because there is no value that is right for everyone, and a
    /// wrong one fails silently in the worst direction: the gap before the
    /// FIRST chunk is a gap like any other, and a reasoning model can think
    /// for minutes before it emits a token. Set this above the slowest
    /// time-to-first-token you expect, not merely above the gap between
    /// chunks — several providers also send `:` keepalive comments during a
    /// long think, and those count as bytes.
    pub stream_read_timeout_secs: Option<u64>,
    /// The providers this client serves, keyed by roster name (`openai`,
    /// `deepseek`, …). ABSENT means every provider warpllm's roster holds.
    ///
    /// Absent and empty are different claims, which is the whole reason this
    /// is an [`Option`] rather than a map defaulting to empty. `None` is "I
    /// did not say", and it is what every client did before this field
    /// existed: the roster is routable end to end, and each provider's
    /// variable is read. `Some({})` is "I said none", and nothing routes.
    ///
    /// Declaring narrows ROUTING as well as keys. A request for a model under
    /// a provider not listed here is refused at admission with
    /// [`Error::ProviderNotDeclared`](crate::Error::ProviderNotDeclared),
    /// before any socket opens — so a client serving one provider cannot be
    /// talked into billing another by a model string. It also narrows what is
    /// READ: the environment is consulted for the declared providers and no
    /// others, so a key exported for something else is not quietly adopted.
    ///
    /// A declaration SELECTS from the roster; it does not extend it. A name
    /// the roster does not hold fails at [`Client::new`](crate::Client::new),
    /// and there is no `base_url` or `models` to write here — adding a
    /// provider warpllm does not ship is a roster change, not a config one.
    ///
    /// A [`BTreeMap`], so the same declaration always logs and reports in the
    /// same order, and so a provider cannot be named twice.
    pub providers: Option<BTreeMap<String, ProviderConfig>>,
}

/// One declared provider.
///
/// An empty entry is the ordinary case: serve this provider, and read its key
/// from the environment variable its roster entry names, exactly as an
/// undeclared client does for the whole roster.
#[derive(Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    /// This provider's API key, in place of the variable the roster names.
    ///
    /// For an embedder that holds keys somewhere the process environment
    /// cannot reach — a secret manager, a per-tenant record in a database.
    /// Absent means the environment still supplies it, so declaring a provider
    /// costs nothing for the callers who never needed this.
    ///
    /// Wins over the environment when both hold a key: this is the more
    /// specific statement, written for this client in this process, and a
    /// caller who put a key here has no other way to make the ambient
    /// environment yield.
    ///
    /// An EMPTY string is treated as unstated rather than as a stated nothing,
    /// so the variable still gets its turn. That is the same judgement `""`
    /// already gets from the environment, and it matters most at the bindings'
    /// boundary: a caller writing `os.environ.get("OPENAI_API_KEY", "")` into
    /// their config would otherwise disable a provider whose key is sitting
    /// right there.
    pub api_key: Option<String>,
}

/// PRESENCE only, never the value. A derived `Debug` would print an inline key
/// the moment anything formats a [`ClientConfig`] — a `tracing::debug!(?config)`
/// in a caller's own code is enough, and the config is held for the life of the
/// client. Whether a key was supplied is the diagnostically useful half, and
/// the only half that is safe to say.
impl std::fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> ClientConfig {
        serde_json::from_str(json).expect("valid configuration")
    }

    /// The redaction claim, on the shape a caller is most likely to print:
    /// the whole config, through the derived `Debug` that `ClientConfig`
    /// keeps.
    #[test]
    fn an_inline_key_never_appears_in_debug_output() {
        let rendered = format!(
            "{:?}",
            parse(r#"{"providers":{"openai":{"api_key":"sk-secret-value"}}}"#)
        );
        assert!(rendered.contains("openai"), "{rendered}");
        assert!(rendered.contains("<redacted>"), "{rendered}");
        assert!(!rendered.contains("sk-secret-value"), "{rendered}");
    }

    /// Saying nothing and saying none are different claims, and the whole
    /// behaviour of the field hangs on the difference — one serves the roster,
    /// the other serves nothing. A `#[serde(default)]` map would collapse them.
    #[test]
    fn an_absent_declaration_is_not_an_empty_one() {
        assert!(parse("{}").providers.is_none());
        assert_eq!(
            parse(r#"{"providers":{}}"#).providers.map(|p| p.len()),
            Some(0)
        );
    }

    /// The ordinary case: serve this provider, key from the environment. An
    /// entry needs nothing in it, and a caller who wants only selection should
    /// not have to write a field to get it.
    #[test]
    fn a_declared_provider_needs_no_entry_of_its_own() {
        let config = parse(r#"{"providers":{"openai":{},"deepseek":{"api_key":"sk-x"}}}"#);
        let providers = config.providers.expect("declared");
        assert_eq!(providers["openai"].api_key, None);
        assert_eq!(providers["deepseek"].api_key.as_deref(), Some("sk-x"));
    }

    /// `deny_unknown_fields` reaches inside an entry too. This is the guard on
    /// the Node binding's hand-written camelCase mapping: a missed `apiKey` ->
    /// `api_key` translation is a key silently ignored and a request that
    /// fails saying the credential is missing, which is a long way from the
    /// mistake.
    #[test]
    fn an_entrys_unknown_field_is_refused() {
        assert!(
            serde_json::from_str::<ClientConfig>(r#"{"providers":{"openai":{"apiKey":"sk-x"}}}"#)
                .is_err()
        );
    }
}
