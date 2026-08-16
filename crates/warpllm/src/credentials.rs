//! Which providers this process can authenticate, worked out once.
//!
//! The roster says which providers exist and which environment variable each
//! reads its key from. This turns that into the shorter list that matters to a
//! running client: the providers a key was found for. A model may be registered
//! and still be unreachable, because the key for its provider is not in this
//! environment — and the two are different failures with different remedies.
//!
//! Two sources, in that order of specificity: a key written into
//! [`ClientConfig::providers`](crate::ClientConfig::providers), and the
//! environment variable the roster names. The declaration also bounds WHICH
//! providers are looked at — a client that named two of them reads two
//! variables, not the roster's worth.
//!
//! What each provider gets is an [`Authenticator`] — the resolved credential
//! AND how it goes on the wire — rather than the bare string it used to be.
//! The two halves are separate on purpose: a SOURCE is where a secret comes
//! from, which is what the paragraph above is about, and a SCHEME is how it
//! goes on the request. Only the two together make a provider reachable, and
//! they vary independently — Bedrock (#24) will take an AWS chain under a
//! signature, Vertex (#25) an OAuth token under a bearer header.
//!
//! Read once, at construction. A snapshot rather than a live read, so the set of
//! providers a client can reach cannot change under it mid-flight, and so the
//! answer can be logged at the one moment a caller is set up to see it. The cost
//! is the other side of that coin: a key exported after the client was built is
//! not picked up, and a rotated key needs a new client.

use std::collections::{BTreeMap, HashMap};

use crate::auth::Authenticator;
use crate::config::ProviderConfig;
use crate::registry::{self, ProviderSpec};

/// The credentials this process holds, keyed by provider name.
///
/// A `HashMap`, matching the registry's own tables: this answers one question,
/// "the credential for this provider", and answers it on every request.
/// Iteration order is unspecified and varies run to run, so everything that
/// RENDERS the set goes through [`Credentials::names`] instead.
pub(crate) struct Credentials {
    keys: HashMap<&'static str, Authenticator>,
}

impl Credentials {
    /// The keys this client holds, from the declaration and the environment.
    ///
    /// `declared` absent means no opinion: every roster provider, each from its
    /// own variable, which is what this did before a declaration was possible.
    /// `Some` means THIS AND NO MORE — and iterating the DECLARATION rather
    /// than the roster is what makes that literal. A version that swept the
    /// whole environment and filtered afterwards would produce the same map
    /// while still reading variables the caller withheld, which is the opposite
    /// of what declaring one provider asks for.
    ///
    /// A provider is included only if some source held a NON-EMPTY key. The
    /// judgement is the same for both sources: warpllm has no key for this
    /// provider, so a request routed to it should say so rather than send
    /// `Authorization: Bearer ` upstream and report back whatever the provider
    /// makes of it.
    ///
    /// The scheme is chosen HERE, and it is [`Authenticator::bearer`] for
    /// everything, because every provider on the roster is an OpenAI-compatible
    /// host that reads `Authorization`. [`Self::key_for`] owns the source and
    /// says nothing about presentation; this is the seam where the table that
    /// makes the scheme a real choice goes, with the first provider that is not
    /// bearer.
    pub(crate) fn resolve(declared: Option<&BTreeMap<String, ProviderConfig>>) -> Self {
        let keys = match declared {
            None => registry::providers()
                .filter_map(|provider| {
                    Some((
                        provider.name(),
                        Authenticator::bearer(Self::key_for(provider, None)?),
                    ))
                })
                .collect(),
            Some(declared) => declared
                .iter()
                .filter_map(|(name, entry)| {
                    let provider = registry::provider(name)
                        .expect("Client::new refuses a declaration the roster does not hold");
                    Some((
                        provider.name(),
                        Authenticator::bearer(Self::key_for(provider, Some(entry))?),
                    ))
                })
                .collect(),
        };
        let credentials = Self { keys };

        // The names only. A key never reaches a log, in full or in part.
        if credentials.keys.is_empty() {
            tracing::warn!(
                "no provider API keys found; every request will fail until one is \
                 set in the environment or declared under `providers`"
            );
        } else {
            tracing::info!(providers = ?credentials.names(), "providers available");
        }
        credentials
    }

    /// One provider's key, or `None` when no source held a non-empty one.
    ///
    /// Precedence is inline over environment: a key written into the config is
    /// the more specific statement, and a caller who has one cannot make the
    /// ambient environment yield.
    ///
    /// An EMPTY inline key is treated as UNSTATED rather than as a stated
    /// nothing, so the environment still gets its turn — see
    /// [`ProviderConfig::api_key`] for why that boundary case is the one worth
    /// designing for.
    ///
    /// Takes the SPEC rather than a name so it can be tested against providers
    /// the shipped roster does not have — one naming no `env_api_key` at all
    /// most of all, which is the case an inline key exists to serve and which
    /// nothing in `specs.yaml` expresses today.
    fn key_for(provider: &ProviderSpec, declared: Option<&ProviderConfig>) -> Option<String> {
        if let Some(inline) = declared
            .and_then(|entry| entry.api_key.as_deref())
            .filter(|key| !key.is_empty())
        {
            return Some(inline.to_string());
        }
        let key = std::env::var(provider.env_api_key()?).ok()?;
        (!key.is_empty()).then_some(key)
    }

    /// This provider's credential, or `None` when no source held a key at
    /// construction — which is what makes "warpllm cannot reach this provider"
    /// answerable without a request.
    pub(crate) fn get(&self, provider: &str) -> Option<&Authenticator> {
        self.keys.get(provider)
    }

    /// The providers a request can be routed to, in name order.
    ///
    /// Sorted here rather than held sorted, because the map's order is
    /// unspecified: this is the only thing standing between a hash-seeded
    /// iteration and a log line whose provider list reshuffles between runs.
    /// Called once per client at construction, never on the request path.
    fn names(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = self.keys.keys().copied().collect();
        names.sort_unstable();
        names
    }
}

/// Names only, never values. Derived `Debug` would print every API key the
/// process holds the moment anything upstream formats a [`crate::Client`].
impl std::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credentials")
            .field("providers", &self.names())
            .finish()
    }
}

/// Runs `body` holding temp-env's global lock, with EVERY roster variable
/// unset and then `vars` applied over the top.
///
/// EVERY test in this binary that builds a [`Credentials`] — which now means
/// every test that builds a [`crate::Client`] — has to go through this, even
/// the ones setting nothing, and even the ones supplying every key inline. Env
/// mutation is process-global and `unsafe` since edition 2024 precisely because
/// one test writing a variable while another reads it is a data race, and only
/// a shared lock rules that out. An empty `vars` is the "reads the environment,
/// sets nothing" case; an all-inline declaration still reads it for any
/// provider whose entry supplied no key.
///
/// Clearing the whole roster first is what keeps the AMBIENT environment out of
/// the result. Naming only the variables a test cares about would leave a
/// contributor who exports a third provider's key failing an assertion about
/// which providers are available — and would quietly add every new provider to
/// every snapshot as the roster grows.
///
/// The closure is SYNCHRONOUS, and the lock is released when it returns. A test
/// asserting what a credential puts on the wire therefore builds its
/// [`Credentials`] inside and awaits outside, rather than holding a
/// process-global lock across an await.
#[cfg(test)]
pub(crate) fn with_env<T>(vars: &[(&str, Option<&str>)], body: impl FnOnce() -> T) -> T {
    let mut settings: std::collections::BTreeMap<String, Option<String>> = registry::providers()
        .filter_map(|provider| provider.env_api_key())
        .map(|var| (var.to_string(), None))
        .collect();
    for (var, value) in vars {
        settings.insert((*var).to_string(), value.map(str::to_string));
    }
    temp_env::with_vars(settings.into_iter().collect::<Vec<_>>(), body)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPENAI: &str = "OPENAI_API_KEY";
    const DEEPSEEK: &str = "DEEPSEEK_API_KEY";

    /// The credential this provider would send, read off a real request.
    ///
    /// Asserted through the wire rather than against a stored string, because
    /// holding a key and presenting it are now two claims and only one of them
    /// is a `String` anyone can compare. Which source the key came from is what
    /// most of these tests are about; that it arrives as `Bearer …` is the
    /// other half, and this asserts both at once.
    async fn sent(credentials: &Credentials, provider: &str) -> Option<String> {
        crate::auth::testing::applied(credentials.get(provider)?, "authorization").await
    }

    /// A key for one provider makes that provider available and says nothing
    /// about any other — the roster is not the availability list.
    #[tokio::test]
    async fn one_key_admits_one_provider() {
        let credentials = with_env(&[(OPENAI, Some("sk-openai")), (DEEPSEEK, None)], || {
            Credentials::resolve(None)
        });
        assert_eq!(credentials.names(), vec!["openai"]);
        assert_eq!(
            sent(&credentials, "openai").await.as_deref(),
            Some("Bearer sk-openai")
        );
        assert!(credentials.get("deepseek").is_none());
    }

    /// Every provider whose variable is set is available at once, in name
    /// order, each carrying its OWN key.
    #[tokio::test]
    async fn every_set_key_admits_its_provider() {
        let credentials = with_env(
            &[(OPENAI, Some("sk-openai")), (DEEPSEEK, Some("sk-deepseek"))],
            || Credentials::resolve(None),
        );
        assert_eq!(credentials.names(), vec!["deepseek", "openai"]);
        assert_eq!(
            sent(&credentials, "deepseek").await.as_deref(),
            Some("Bearer sk-deepseek")
        );
    }

    /// An empty environment is not an error. Construction succeeds with nothing
    /// available, and the failure lands on the request that needed a key —
    /// where the message can name the provider and the variable to set.
    #[test]
    fn an_empty_environment_yields_no_providers() {
        with_env(&[(OPENAI, None), (DEEPSEEK, None)], || {
            let credentials = Credentials::resolve(None);
            assert!(credentials.names().is_empty());
            assert!(credentials.get("openai").is_none());
        });
    }

    /// `OPENAI_API_KEY=` is a variable that is set and a key that is not.
    /// Treating it as present would send `Bearer ` upstream and turn a local
    /// configuration mistake into a remote authentication failure.
    #[test]
    fn an_empty_value_is_not_a_key() {
        with_env(&[(OPENAI, Some("")), (DEEPSEEK, None)], || {
            assert!(Credentials::resolve(None).names().is_empty());
        });
    }

    /// A provider warpllm does not serve has no key here, whatever the
    /// environment holds.
    #[test]
    fn an_unknown_provider_is_never_available() {
        with_env(&[(OPENAI, Some("sk-openai"))], || {
            assert!(Credentials::resolve(None).get("mistral").is_none());
        });
    }

    /// Debug is what a panic or a `tracing` field would print. It must name the
    /// providers and none of their keys, whichever source they came from.
    #[test]
    fn debug_redacts_the_keys() {
        with_env(
            &[(OPENAI, Some("sk-secret-value")), (DEEPSEEK, None)],
            || {
                let rendered = format!("{:?}", Credentials::resolve(None));
                assert!(rendered.contains("openai"), "{rendered}");
                assert!(!rendered.contains("sk-secret-value"), "{rendered}");

                let inline = format!(
                    "{:?}",
                    Credentials::resolve(Some(&declare(&[
                        ("deepseek", Some("sk-inline-secret"),)
                    ])))
                );
                assert!(inline.contains("deepseek"), "{inline}");
                assert!(!inline.contains("sk-inline-secret"), "{inline}");
            },
        );
    }

    // ------------------------------------------------------- declared clients

    /// Builds a declaration. `None` is an entry with no key of its own, which
    /// is the ordinary case: serve this provider, read its variable.
    fn declare(entries: &[(&str, Option<&str>)]) -> BTreeMap<String, ProviderConfig> {
        entries
            .iter()
            .map(|(name, key)| {
                (
                    (*name).to_string(),
                    ProviderConfig {
                        api_key: key.map(str::to_string),
                    },
                )
            })
            .collect()
    }

    /// The claim the whole field rests on: a provider the caller did not name
    /// is not consulted, even when its variable is sitting right there. A key
    /// exported for something else is not quietly adopted.
    #[test]
    fn an_undeclared_providers_variable_is_never_read() {
        with_env(
            &[(OPENAI, Some("sk-openai")), (DEEPSEEK, Some("sk-deepseek"))],
            || {
                let credentials = Credentials::resolve(Some(&declare(&[("openai", None)])));
                assert_eq!(credentials.names(), vec!["openai"]);
                assert!(credentials.get("deepseek").is_none());
            },
        );
    }

    /// Declaring a provider without a key of its own leaves the environment
    /// doing exactly what it did before — declaring is not opting out of it.
    #[tokio::test]
    async fn a_declaration_with_no_key_still_reads_the_environment() {
        let credentials = with_env(&[(OPENAI, Some("sk-openai")), (DEEPSEEK, None)], || {
            Credentials::resolve(Some(&declare(&[("openai", None)])))
        });
        assert_eq!(
            sent(&credentials, "openai").await.as_deref(),
            Some("Bearer sk-openai")
        );
    }

    /// The more specific statement wins. A caller who wrote a key into the
    /// config has no other way to make the ambient environment yield.
    #[tokio::test]
    async fn an_inline_key_wins_over_the_environment() {
        let credentials = with_env(&[(OPENAI, Some("sk-from-the-environment"))], || {
            Credentials::resolve(Some(&declare(&[("openai", Some("sk-inline"))])))
        });
        assert_eq!(
            sent(&credentials, "openai").await.as_deref(),
            Some("Bearer sk-inline")
        );
    }

    /// `""` is unstated, not a stated nothing — the same judgement the
    /// environment's own empty value gets. A caller writing
    /// `os.environ.get("OPENAI_API_KEY", "")` into their config must not
    /// thereby disable a provider whose key is right there.
    #[tokio::test]
    async fn an_empty_inline_key_falls_back_to_the_environment() {
        let credentials = with_env(&[(OPENAI, Some("sk-openai"))], || {
            Credentials::resolve(Some(&declare(&[("openai", Some(""))])))
        });
        assert_eq!(
            sent(&credentials, "openai").await.as_deref(),
            Some("Bearer sk-openai")
        );
    }

    /// Falling back to nothing is still nothing. Neither source held a key, so
    /// the provider is unavailable rather than authenticated with `""`.
    #[test]
    fn an_empty_inline_key_with_no_variable_set_is_no_key() {
        with_env(&[(OPENAI, None)], || {
            assert!(
                Credentials::resolve(Some(&declare(&[("openai", Some(""))])))
                    .names()
                    .is_empty()
            );
        });
    }

    /// Declaring nothing reaches nothing. Legal, and distinct from declaring
    /// nothing at all — which is the next test.
    #[test]
    fn an_empty_declaration_authenticates_nothing() {
        with_env(&[(OPENAI, Some("sk-openai"))], || {
            assert!(
                Credentials::resolve(Some(&BTreeMap::new()))
                    .names()
                    .is_empty()
            );
        });
    }

    /// The compatibility claim: an absent declaration behaves exactly as this
    /// did before the field existed.
    #[test]
    fn an_absent_declaration_reads_the_whole_roster() {
        with_env(
            &[(OPENAI, Some("sk-openai")), (DEEPSEEK, Some("sk-deepseek"))],
            || {
                assert_eq!(
                    Credentials::resolve(None).names(),
                    vec!["deepseek", "openai"]
                );
            },
        );
    }

    /// The capability an inline key adds rather than narrows: a provider whose
    /// roster entry names no variable had no key source at all, and now has
    /// one. Nothing in `specs.yaml` expresses this today, which is why
    /// `key_for` takes a spec — the case has to be built to be tested.
    #[test]
    fn an_inline_key_authenticates_a_provider_the_roster_gives_no_variable() {
        let spec = ProviderSpec {
            name: "keyless".into(),
            base_url: "https://api.keyless.test/v1".into(),
            env_api_key: None,
        };
        with_env(&[], || {
            assert_eq!(
                Credentials::key_for(
                    &spec,
                    Some(&ProviderConfig {
                        api_key: Some("sk-inline".into()),
                    }),
                )
                .as_deref(),
                Some("sk-inline")
            );
            // Without one it is still unauthenticatable, and the request-time
            // error names the roster rather than a variable nothing reads.
            assert_eq!(Credentials::key_for(&spec, None), None);
        });
    }
}
