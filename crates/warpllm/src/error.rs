use crate::gateway::types::ProviderError;
use crate::protocol::openai_compat::error::OpenAiError;

/// Every way a warpllm call can fail, in ONE flat enum.
///
/// Flat on purpose. A provider failure is not a special kind of error that
/// needs unwrapping before it can be read — it is an error, and it sits
/// beside warpllm's own. That means one `match` reaches every case
/// (`Error::RateLimited(e)`), rather than a match on `Error` followed by a
/// second match on a nested kind. This enum is Rust's alone — the bindings
/// raise one class carrying OpenAI's error object — so a nested kind would
/// have been a shape only Rust callers ever saw.
///
/// The trade the flattening makes is that WHO failed is no longer visible in
/// the type — so [`Error::origin`] states it explicitly, and
/// [`Error::provider_error`] hands back the upstream's evidence for the
/// variants that have any. Both are total functions over the enum, which is
/// what stops the grouping from rotting as variants are added.
///
/// `non_exhaustive` because this exists to grow: providers invent failure
/// modes, and adding one must not break every downstream `match`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    // ---------------------------------------------- warpllm's own failures
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid model string '{given}': no registered model spec")]
    InvalidModel { given: String },
    #[error("{}", missing_api_key_message(provider, *env_var))]
    MissingApiKey {
        provider: &'static str,
        /// The variable to set, when the registry names one for this model.
        ///
        /// `None` means the entry declares no `env_api_key`, and since the
        /// environment is the only key source, such an entry cannot be
        /// authenticated at all — the remedy is a roster edit, not a shell one.
        /// One `missing_api_key` code covers both: the failure is the same,
        /// only the remedy differs, and the remedy rides in the message —
        /// which is the only place it can, since the OpenAI envelope the
        /// bindings receive has no field to put a variable name in.
        env_var: Option<&'static str>,
    },
    /// The routed provider is not one this client declared, so nothing routes
    /// to it here — whatever the roster says about it.
    ///
    /// Distinct from [`InvalidModel`](Self::InvalidModel), which means warpllm
    /// serves no such model anywhere, and from
    /// [`MissingApiKey`](Self::MissingApiKey), which means the provider IS
    /// served here and has no credential. Both of those remedies would be
    /// wrong advice: there is nothing to spell differently and nothing to
    /// export. What is wrong is the client's declaration, or the model string
    /// aimed at it.
    ///
    /// Only a client that declared
    /// [`ClientConfig::providers`](crate::ClientConfig::providers) can produce
    /// this. Leaving that field absent leaves the whole roster routable, which
    /// is what every client did before it existed.
    #[error(
        "{provider} is not declared on this client: add it to `providers` in the client \
         configuration, or route {requested} to a provider that is"
    )]
    ProviderNotDeclared {
        provider: &'static str,
        /// The caller's own routing string, quoted back so the message names
        /// what was typed rather than only the provider it resolved to.
        requested: String,
    },
    /// The provider was never reached, so it never had a chance to answer.
    #[error("network error talking to {provider}: {source}")]
    Network {
        provider: &'static str,
        #[source]
        source: reqwest::Error,
    },
    /// The provider answered with a success status and a body warpllm could
    /// not read — a gap in warpllm's shapes, not a failure the provider
    /// reported.
    #[error("could not decode {provider} response: {message}")]
    Decode {
        provider: &'static str,
        message: String,
    },
    /// The provider's stream ended before the sentinel that says it finished.
    ///
    /// A stream has two ways to stop and only one of them means the answer is
    /// whole. Without this, they are the same `None` — and a caller told a
    /// truncated reply was complete has no way left to find out otherwise.
    ///
    /// Nothing is attached because nothing was said: the socket closed. That
    /// makes it warpllm's report of an upstream's silence rather than an
    /// upstream's own failure, which is why it sits here beside
    /// [`Network`](Self::Network) and [`Decode`](Self::Decode).
    #[error("{provider} ended the stream before it was complete")]
    StreamTruncated { provider: &'static str },
    /// The provider's stream went quiet for longer than
    /// [`ClientConfig::stream_read_timeout_secs`](crate::ClientConfig::stream_read_timeout_secs),
    /// so warpllm stopped waiting.
    ///
    /// Separate from [`StreamTruncated`](Self::StreamTruncated) because the
    /// socket never closed and the remedy is different: an upstream really is
    /// wedged, or the limit is tighter than the model's slowest pause. Only
    /// the caller knows which, and only if told which one happened.
    #[error("{provider} sent nothing for {}s", timeout.as_secs())]
    StreamStalled {
        provider: &'static str,
        timeout: std::time::Duration,
    },
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
    /// warpllm could not set itself up — building the HTTP client failed, for
    /// instance, which happens when the platform's TLS backend will not
    /// initialize.
    ///
    /// Nothing the caller passed is wrong, so this is deliberately NOT
    /// [`InvalidInput`](Self::InvalidInput): that spelling would tell someone
    /// to go fix a payload that was fine, and the server would answer 400 for
    /// what is squarely a 500.
    #[error("internal error: {0}")]
    Internal(String),

    // ----------------------------------------- the provider's own failures
    //
    // One variant per normalized failure, each carrying the same evidence.
    // Boxed so that eleven wide variants do not widen every `Result` in the
    // crate. What a caller should DO about one is deliberately absent:
    // warpllm has no retry loop and no fallback chain yet, so an opinion
    // about either would be an untested guess shipped as a public contract.
    /// Requests arrived faster than the account may send them.
    #[error("{0}")]
    RateLimited(Box<ProviderError>),
    /// The account is out of credit or past a spend cap. Distinct from
    /// [`RateLimited`](Self::RateLimited) because waiting does not fix it —
    /// somebody has to pay — and providers report both under a 429.
    #[error("{0}")]
    QuotaExceeded(Box<ProviderError>),
    /// The provider is up but shedding load (503, Anthropic-style 529).
    #[error("{0}")]
    Overloaded(Box<ProviderError>),
    /// An unattributed 5xx from the provider.
    #[error("{0}")]
    ServerError(Box<ProviderError>),
    /// The prompt (or prompt + completion) exceeds the model's window.
    #[error("{0}")]
    ContextLengthExceeded(Box<ProviderError>),
    /// The provider refused on safety grounds.
    #[error("{0}")]
    ContentFilter(Box<ProviderError>),
    /// The credential is missing or invalid. Distinct from
    /// [`MissingApiKey`](Self::MissingApiKey), which fires before any
    /// request goes out.
    #[error("{0}")]
    Authentication(Box<ProviderError>),
    /// The credential is valid but not entitled to what was asked for.
    /// Split from [`Authentication`](Self::Authentication) because the
    /// remedy differs: a wrong key is replaced, an unentitled one is
    /// granted access.
    #[error("{0}")]
    PermissionDenied(Box<ProviderError>),
    /// The provider does not serve the model that was requested. Distinct
    /// from [`InvalidModel`](Self::InvalidModel), which fires before any
    /// request goes out.
    #[error("{0}")]
    ModelNotFound(Box<ProviderError>),
    /// The provider rejected the request as malformed, or rejected a
    /// parameter it does not accept.
    #[error("{0}")]
    InvalidRequest(Box<ProviderError>),
    /// The provider failed and nothing in the envelope identified how.
    /// Carries no claim either way — read the [`ProviderError`] and decide.
    #[error("{0}")]
    Unknown(Box<ProviderError>),
}

/// Who a failure came from.
///
/// The distinction the flat enum gives up in its shape, handed back as a
/// value. It is the first question worth asking about any failure: an
/// upstream answered and said no, or the call never got that far.
///
/// Deliberately NOT `non_exhaustive`, unlike [`Error`] itself. Two arms is
/// the whole design — every failure is one side or the other — so a third
/// would be a change every consumer must see, not one to wave through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// warpllm, its configuration, or the call into it. The provider either
    /// was never reached or was never asked. Retrying the identical call
    /// changes nothing.
    Gateway,
    /// The provider answered, and its answer was a failure. A
    /// [`ProviderError`] is attached with everything it said.
    Provider,
}

impl Origin {
    /// The wire spelling, written once so every surface that reports an
    /// origin agrees. Today that is the gateway's HTTP envelope; the FFI
    /// form deliberately omits it, origin being warpllm's own vocabulary.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gateway => "gateway",
            Self::Provider => "provider",
        }
    }
}

/// A function rather than two `#[error]` strings because the REMEDY differs
/// while the failure does not. Naming a variable that no spec declares would
/// send someone off to set an environment variable nothing ever reads.
fn missing_api_key_message(provider: &str, env_var: Option<&str>) -> String {
    match env_var {
        Some(var) => format!("missing API key for {provider}: set the {var} environment variable"),
        None => format!(
            "missing API key for {provider}: its registry entry names no environment variable \
             to read one from, so declare it under `providers` with an `api_key` instead"
        ),
    }
}

impl Error {
    /// Whether the provider said no, or the call failed before it could.
    ///
    /// Exhaustive by construction: a new variant does not compile until it
    /// is placed on one side or the other, which is what keeps this honest
    /// as the enum grows.
    pub fn origin(&self) -> Origin {
        match self {
            Error::InvalidInput(_)
            | Error::InvalidModel { .. }
            | Error::MissingApiKey { .. }
            | Error::ProviderNotDeclared { .. }
            // Network and Decode name a provider but are NOT its failures:
            // one never reached it, the other means it answered with a
            // success warpllm could not read. Neither is something the
            // upstream reported.
            | Error::Network { .. }
            | Error::Decode { .. }
            | Error::StreamTruncated { .. }
            | Error::StreamStalled { .. }
            | Error::NotImplemented(_)
            | Error::Internal(_) => Origin::Gateway,
            Error::RateLimited(_)
            | Error::QuotaExceeded(_)
            | Error::Overloaded(_)
            | Error::ServerError(_)
            | Error::ContextLengthExceeded(_)
            | Error::ContentFilter(_)
            | Error::Authentication(_)
            | Error::PermissionDenied(_)
            | Error::ModelNotFound(_)
            | Error::InvalidRequest(_)
            | Error::Unknown(_) => Origin::Provider,
        }
    }

    /// Everything the upstream said, for the variants an upstream produced.
    ///
    /// `Some` exactly when [`origin`](Self::origin) is
    /// [`Origin::Provider`] — the two answer the same question, one as a
    /// tag and one as the evidence.
    pub fn provider_error(&self) -> Option<&ProviderError> {
        match self {
            Error::RateLimited(e)
            | Error::QuotaExceeded(e)
            | Error::Overloaded(e)
            | Error::ServerError(e)
            | Error::ContextLengthExceeded(e)
            | Error::ContentFilter(e)
            | Error::Authentication(e)
            | Error::PermissionDenied(e)
            | Error::ModelNotFound(e)
            | Error::InvalidRequest(e)
            | Error::Unknown(e) => Some(e),
            _ => None,
        }
    }

    /// The stable machine-readable slug for this variant, and a public
    /// contract: these strings cross the FFI boundary as `code` and the
    /// bindings dispatch on them.
    ///
    /// Flat, like the enum — one slug per variant, no second field to
    /// disambiguate. `invalid_request` is warpllm's own rejection and
    /// `provider_invalid_request` is the upstream's, because a caller
    /// fixing the first edits their call while the second may just need a
    /// different model.
    pub fn code(&self) -> &'static str {
        match self {
            Error::InvalidInput(_) | Error::InvalidModel { .. } => "invalid_request",
            Error::MissingApiKey { .. } => "missing_api_key",
            Error::ProviderNotDeclared { .. } => "provider_not_declared",
            Error::Network { .. } => "connection_error",
            Error::Decode { .. } => "decode_error",
            Error::StreamTruncated { .. } => "stream_truncated",
            Error::StreamStalled { .. } => "stream_stalled",
            Error::NotImplemented(_) => "not_implemented",
            Error::Internal(_) => "internal_error",
            Error::RateLimited(_) => "rate_limited",
            Error::QuotaExceeded(_) => "quota_exceeded",
            Error::Overloaded(_) => "overloaded",
            Error::ServerError(_) => "provider_server_error",
            Error::ContextLengthExceeded(_) => "context_length_exceeded",
            Error::ContentFilter(_) => "content_filter",
            Error::Authentication(_) => "authentication",
            Error::PermissionDenied(_) => "permission_denied",
            Error::ModelNotFound(_) => "model_not_found",
            Error::InvalidRequest(_) => "provider_invalid_request",
            Error::Unknown(_) => "provider_unknown",
        }
    }

    /// This failure as an OpenAI-compatible surface reports it.
    ///
    /// ONE conversion, two renderings: the HTTP gateway turns this into a
    /// status line and a JSON body, a binding turns the status into the
    /// exception class its language's OpenAI SDK would have raised. Neither
    /// decides anything, so neither can disagree with the other.
    ///
    /// Normalizing, not forwarding. A failure warpllm CLASSIFIED is reported
    /// the way OpenAI reports it — DeepSeek's 402 for an exhausted balance
    /// comes out as 429 `insufficient_quota` — because a caller handling a
    /// billing failure should not have to know which provider served the
    /// request. A failure warpllm did NOT classify keeps the upstream's own
    /// answer, since inventing one would be a guess in normalization's
    /// clothing.
    ///
    /// warpllm's own taxonomy is absent by design. [`code`](Self::code) and
    /// [`origin`](Self::origin) stay Rust-side, where changing them costs a
    /// recompile instead of a major version.
    pub fn to_openai(&self) -> OpenAiError {
        crate::gateway::openai_compat::error::to_openai(self)
    }

    /// [`to_openai`](Self::to_openai), serialized for the FFI boundary.
    ///
    /// The JSON contains the status, headers, OpenAI error object, and the
    /// language-level error class already selected by Rust. A binding
    /// reconstructs its local exception without interpreting the status.
    pub fn to_openai_json(&self) -> String {
        serde_json::to_string(&self.to_openai())
            .expect("an OpenAiError is plain data and always serializes")
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn wire(err: &Error) -> serde_json::Value {
        serde_json::from_str(&err.to_openai_json()).unwrap()
    }

    fn rate_limit() -> Error {
        Error::RateLimited(Box::new(ProviderError {
            provider: "openai",
            status: 429,
            message: "slow down".into(),
            error_type: Some("rate_limit_error".into()),
            provider_code: Some("rate_limit_exceeded".into()),
            retry_after: Some(Duration::from_secs(30)),
            request_id: Some("req-1".into()),
            raw_body: r#"{"error":{"message":"slow down"}}"#.into(),
        }))
    }

    #[test]
    fn a_provider_failure_renders_as_an_openai_error() {
        let v = wire(&rate_limit());
        assert_eq!(v["status"], 429);
        assert_eq!(v["error"]["type"], "rate_limit_error");
        assert!(v["error"]["param"].is_null());
        assert!(v["error"]["message"].as_str().unwrap().contains("HTTP 429"));
        assert_eq!(v["error"]["code"], "rate_limit_exceeded");
        // Both live only in the response headers, so they travel as headers
        // here too rather than as body fields OpenAI has no place for.
        assert_eq!(v["headers"]["x-request-id"], "req-1");
        assert_eq!(v["headers"]["retry-after"], "30");
    }

    /// A quota exhaustion and a rate limit are both 429s and read alike, but
    /// no amount of backing off buys credit. OpenAI already spells the
    /// difference — `insufficient_quota` — so the distinction survives a
    /// pure-OpenAI envelope without warpllm's taxonomy crossing to carry it.
    #[test]
    fn quota_exhaustion_stays_distinguishable_from_a_rate_limit() {
        let quota = Error::QuotaExceeded(Box::new(ProviderError {
            provider: "openai",
            status: 429,
            message: "quota".into(),
            error_type: Some("insufficient_quota".into()),
            provider_code: Some("insufficient_quota".into()),
            retry_after: None,
            request_id: None,
            raw_body: String::new(),
        }));
        let (quota, limit) = (wire(&quota), wire(&rate_limit()));

        assert_eq!(quota["status"], limit["status"], "both are 429s");
        assert_ne!(quota["error"]["code"], limit["error"]["code"]);
        assert_eq!(quota["error"]["code"], "insufficient_quota");
    }

    /// NO OPINION, so nothing is invented. warpllm did not classify this
    /// failure, so the upstream's own status stands and `code` is left null
    /// rather than filled with a warpllm slug an OpenAI client cannot read.
    #[test]
    fn an_unclassified_failure_keeps_the_upstreams_own_answer() {
        let v = wire(&Error::Unknown(Box::new(ProviderError {
            provider: "demo",
            status: 402,
            message: "balance".into(),
            error_type: None,
            provider_code: None,
            retry_after: None,
            request_id: None,
            raw_body: "Insufficient Balance".into(),
        })));
        assert_eq!(v["status"], 402, "the upstream's own, untouched");
        assert_eq!(v["error"]["type"], "api_error");
        assert!(v["error"]["code"].is_null());
        assert!(v["headers"].as_object().unwrap().is_empty());
    }

    /// The other half of the same rule: a failure warpllm DID classify is
    /// reported as OpenAI reports it, whatever the provider called it.
    /// DeepSeek answers an exhausted balance with 402, which OpenAI never
    /// sends and its SDKs have no arm for — so a caller branching on a
    /// billing failure would simply miss it.
    #[test]
    fn a_classified_failure_is_normalized_away_from_the_provider_spelling() {
        let v = wire(&Error::QuotaExceeded(Box::new(ProviderError {
            provider: "deepseek",
            status: 402,
            message: "Insufficient Balance".into(),
            error_type: None,
            provider_code: None,
            retry_after: None,
            request_id: None,
            raw_body: "Insufficient Balance".into(),
        })));
        assert_eq!(v["status"], 429, "402 is not a status OpenAI sends");
        assert_eq!(v["error"]["type"], "insufficient_quota");
        assert_eq!(v["error"]["code"], "insufficient_quota");
    }

    /// The envelope is OpenAI's and only OpenAI's. warpllm's taxonomy is
    /// reachable in Rust and deliberately absent from the wire, so shipping
    /// a binding does not promise compatibility on a name still in flux.
    #[test]
    fn warpllms_own_vocabulary_never_crosses_the_boundary() {
        for error in [rate_limit(), Error::NotImplemented("streaming")] {
            let v = wire(&error);
            let envelope = v["error"].as_object().unwrap();
            let mut keys: Vec<_> = envelope.keys().map(String::as_str).collect();
            keys.sort_unstable();
            assert_eq!(keys, ["code", "message", "param", "type"], "{v}");
            for warpllm_only in ["origin", "provider", "provider_code", "provider_raw"] {
                assert!(v.get(warpllm_only).is_none(), "{warpllm_only} in {v}");
            }
        }
    }

    /// A gateway-side failure has no upstream to borrow a status from, so
    /// the mapping has to supply one an OpenAI client can act on.
    #[test]
    fn gateway_failures_map_onto_openai_statuses() {
        let cases = [
            (
                Error::InvalidInput("x".into()),
                400,
                "invalid_request_error",
            ),
            (
                Error::MissingApiKey {
                    provider: "openai",
                    env_var: Some("OPENAI_API_KEY"),
                },
                401,
                "invalid_request_error",
            ),
            (
                Error::ProviderNotDeclared {
                    provider: "deepseek",
                    requested: "deepseek/deepseek-v4-flash".into(),
                },
                400,
                "invalid_request_error",
            ),
            (Error::NotImplemented("streaming"), 501, "api_error"),
            (Error::Internal("tls".into()), 500, "server_error"),
        ];
        for (error, status, error_type) in cases {
            let rendered = error.to_openai();
            assert_eq!(rendered.status, Some(status), "{error:?}");
            assert_eq!(rendered.error.error_type, error_type, "{error:?}");
        }
    }

    /// An undeclared provider and a missing key are the two failures most
    /// easily mistaken for each other, and the wire has to keep them apart:
    /// one is answered by editing the client's configuration, the other by
    /// exporting a variable. A caller sent after a credential they already
    /// hold and deliberately withheld has been told the wrong thing.
    ///
    /// This is also the only guard on the mapping itself. `opinion` ends in a
    /// catch-all arm, so a variant left out of it does not fail to compile —
    /// it renders with a null status and nobody hears about it until a caller
    /// does.
    #[test]
    fn an_undeclared_provider_is_not_a_missing_key_on_the_wire() {
        let undeclared = wire(&Error::ProviderNotDeclared {
            provider: "deepseek",
            requested: "deepseek/deepseek-v4-flash".into(),
        });
        let unauthenticated = wire(&Error::MissingApiKey {
            provider: "deepseek",
            env_var: Some("DEEPSEEK_API_KEY"),
        });

        assert_eq!(undeclared["status"], 400, "the configuration is wrong");
        assert_eq!(unauthenticated["status"], 401, "the credential is missing");
        assert_ne!(
            undeclared["error"]["code"],
            unauthenticated["error"]["code"]
        );
        // The routing string the caller typed, so the message names what they
        // wrote rather than only the provider it resolved to.
        assert!(
            undeclared["error"]["message"]
                .as_str()
                .unwrap()
                .contains("deepseek/deepseek-v4-flash"),
            "{undeclared}"
        );
    }

    /// The two ways a stream dies badly, told apart on the wire.
    ///
    /// Both are 5xx and neither is the provider's reported failure, so the
    /// temptation is one code for "the stream broke". They earn separate ones
    /// because the remedies diverge: a truncation is an upstream that hung up,
    /// while a stall may be nothing worse than `stream_read_timeout_secs` set
    /// tighter than the model's slowest pause — and the message has to name
    /// the limit for that to be actionable.
    #[test]
    fn a_dead_stream_and_a_quiet_one_are_distinguishable() {
        let truncated = wire(&Error::StreamTruncated { provider: "openai" });
        let stalled = wire(&Error::StreamStalled {
            provider: "openai",
            timeout: Duration::from_secs(45),
        });

        assert_eq!(truncated["status"], 502, "the upstream answered badly");
        assert_eq!(stalled["status"], 504, "the upstream stopped answering");
        assert_ne!(truncated["error"]["code"], stalled["error"]["code"]);
        assert!(
            stalled["error"]["message"].as_str().unwrap().contains("45"),
            "the limit that fired is the whole remedy: {stalled}"
        );
    }

    /// A body did arrive — it was just unreadable — so this is a bad
    /// gateway, not a connection that never happened.
    #[test]
    fn an_unreadable_body_is_a_bad_gateway() {
        let rendered = Error::Decode {
            provider: "openai",
            message: "bad json".into(),
        }
        .to_openai();
        assert_eq!(rendered.status, Some(502));
    }

    /// The whole point of the flat enum: a caller matches one level.
    #[test]
    fn a_provider_failure_matches_without_unwrapping_a_kind() {
        assert!(matches!(rate_limit(), Error::RateLimited(_)));
    }

    /// `origin` and `provider_error` answer the same question two ways, so
    /// they must never disagree — an evidence-free provider error, or an
    /// upstream payload on a gateway failure, would both be bugs.
    #[test]
    fn origin_and_provider_error_agree_on_every_variant() {
        let all = [
            Error::InvalidInput("x".into()),
            Error::InvalidModel { given: "x".into() },
            Error::MissingApiKey {
                provider: "openai",
                env_var: None,
            },
            Error::ProviderNotDeclared {
                provider: "openai",
                requested: "openai/gpt-5.6".into(),
            },
            Error::Decode {
                provider: "openai",
                message: "x".into(),
            },
            Error::StreamTruncated { provider: "openai" },
            Error::StreamStalled {
                provider: "openai",
                timeout: Duration::from_secs(60),
            },
            Error::NotImplemented("x"),
            Error::Internal("x".into()),
            rate_limit(),
            Error::QuotaExceeded(Box::new(provider_error())),
            Error::Overloaded(Box::new(provider_error())),
            Error::ServerError(Box::new(provider_error())),
            Error::ContextLengthExceeded(Box::new(provider_error())),
            Error::ContentFilter(Box::new(provider_error())),
            Error::Authentication(Box::new(provider_error())),
            Error::PermissionDenied(Box::new(provider_error())),
            Error::ModelNotFound(Box::new(provider_error())),
            Error::InvalidRequest(Box::new(provider_error())),
            Error::Unknown(Box::new(provider_error())),
        ];
        for error in &all {
            assert_eq!(
                error.origin() == Origin::Provider,
                error.provider_error().is_some(),
                "{error:?}"
            );
        }
    }

    fn provider_error() -> ProviderError {
        ProviderError {
            provider: "demo",
            status: 500,
            message: "m".into(),
            error_type: None,
            provider_code: None,
            retry_after: None,
            request_id: None,
            raw_body: String::new(),
        }
    }

    /// The slugs are a public contract the bindings dispatch on, and they
    /// must be UNIQUE now that one flat space holds warpllm's failures and
    /// the provider's: `invalid_request` is warpllm rejecting the call,
    /// `provider_invalid_request` is the upstream rejecting it.
    #[test]
    fn every_code_is_distinct() {
        let codes = [
            Error::InvalidInput("x".into()).code(),
            Error::MissingApiKey {
                provider: "openai",
                env_var: None,
            }
            .code(),
            Error::ProviderNotDeclared {
                provider: "openai",
                requested: "openai/gpt-5.6".into(),
            }
            .code(),
            Error::Decode {
                provider: "openai",
                message: "x".into(),
            }
            .code(),
            Error::StreamTruncated { provider: "openai" }.code(),
            Error::StreamStalled {
                provider: "openai",
                timeout: Duration::from_secs(60),
            }
            .code(),
            Error::NotImplemented("x").code(),
            Error::Internal("x".into()).code(),
            Error::RateLimited(Box::new(provider_error())).code(),
            Error::QuotaExceeded(Box::new(provider_error())).code(),
            Error::Overloaded(Box::new(provider_error())).code(),
            Error::ServerError(Box::new(provider_error())).code(),
            Error::ContextLengthExceeded(Box::new(provider_error())).code(),
            Error::ContentFilter(Box::new(provider_error())).code(),
            Error::Authentication(Box::new(provider_error())).code(),
            Error::PermissionDenied(Box::new(provider_error())).code(),
            Error::ModelNotFound(Box::new(provider_error())).code(),
            Error::InvalidRequest(Box::new(provider_error())).code(),
            Error::Unknown(Box::new(provider_error())).code(),
        ];
        let unique: std::collections::BTreeSet<_> = codes.iter().collect();
        assert_eq!(unique.len(), codes.len(), "{codes:?}");
        // The pair that forced the naming: both mean "bad request", and a
        // caller fixes them in different places.
        assert_eq!(Error::InvalidInput("x".into()).code(), "invalid_request");
        assert_eq!(
            Error::InvalidRequest(Box::new(provider_error())).code(),
            "provider_invalid_request"
        );
    }

    /// A quota exhaustion is not a rate limit. Both arrive as a 429 and
    /// read alike, but no amount of backing off buys credit — so they are
    /// separate variants with separate slugs.
    #[test]
    fn quota_exhaustion_is_a_distinct_variant_from_a_rate_limit() {
        let quota = Error::QuotaExceeded(Box::new(provider_error()));
        assert!(!matches!(quota, Error::RateLimited(_)));
        assert_ne!(quota.code(), rate_limit().code());
    }

    /// The variable to set is the whole remedy, and the envelope has no
    /// field for it — so it has to survive in the message instead.
    #[test]
    fn missing_key_wire_format() {
        let v = wire(&Error::MissingApiKey {
            provider: "openai",
            env_var: Some("OPENAI_API_KEY"),
        });
        assert_eq!(v["status"], 401);
        assert_eq!(v["error"]["code"], "invalid_api_key");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("OPENAI_API_KEY")
        );
    }

    /// Network names a provider but is warpllm-side: the upstream was never
    /// reached, so there is nothing it said to report — no status of its
    /// own, and no `request_id`.
    #[test]
    fn a_network_failure_is_a_gateway_failure() {
        let error = Error::Decode {
            provider: "openai",
            message: "bad json".into(),
        };
        assert_eq!(error.origin(), Origin::Gateway);
        assert!(error.provider_error().is_none());
        let v = wire(&error);
        assert_eq!(v["status"], 502);
        assert_eq!(v["error"]["type"], "api_error");
        assert!(v["request_id"].is_null());
    }

    /// A setup failure is warpllm's, not the caller's. Spelled out because
    /// the tempting variant is `InvalidInput` — which reads as "your request
    /// was wrong" and lands a 500-class failure on a 400.
    #[test]
    fn a_setup_failure_blames_the_gateway_and_not_the_caller() {
        let error = Error::Internal("tls backend unavailable".into());
        assert_eq!(error.origin(), Origin::Gateway);
        assert!(error.provider_error().is_none());
        assert_ne!(error.code(), Error::InvalidInput("x".into()).code());
        let v = wire(&error);
        assert_eq!(v["error"]["code"], "internal_error");
        assert_eq!(v["status"], 500, "a setup failure is not a 400");
    }

    #[test]
    fn not_implemented_wire_format() {
        let v = wire(&Error::NotImplemented("streaming"));
        assert_eq!(v["status"], 501);
        assert_eq!(v["error"]["code"], "not_implemented");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("streaming")
        );
    }
}
