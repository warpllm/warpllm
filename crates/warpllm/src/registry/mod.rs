//! The model roster and the lookup over it.
//!
//! The roster is pure data, edited by the community in `specs.yaml` next
//! door. Nothing here is hand-written per provider: adding a provider or a
//! model is a YAML edit, never a Rust edit.
//!
//! The file is compiled into the binary with `include_str!` and loaded once,
//! lazily, on the first lookup — a few kilobytes, turned into two `HashMap`s
//! that then answer every request without walking anything. It used to be
//! compiled to Rust source by a build script; that bought a build-time gate on
//! a bad roster and cost a code generator writing Rust by string
//! concatenation, a types file `include!`d into two crates, and leaked
//! `&'static` strings to make the result a `const`. The gate now lives in
//! `the_shipped_registry_loads_and_lints`, which CI runs on every PR.
//!
//! Three modules and no more: `types` is what a spec IS, `load` turns the
//! file into the tables, `lint` holds what is merely true of a tidy roster.
//! Each keeps its own tests.

use std::sync::LazyLock;

use crate::error::{Error, Result};

mod load;
mod types;

#[cfg(test)]
mod lint;
#[cfg(test)]
mod testing;

use types::Registry;
pub use types::{Capabilities, ModelSpec, ProviderSpec, SupportedApi};

/// The shipped roster, loaded on first use.
///
/// The panic is deliberate and stays out of [`fetch_model`]'s signature: the
/// input is compiled into this binary, cannot vary at runtime, and is held to
/// both gates by `the_shipped_registry_loads_and_lints`. Threading "warpllm's
/// own roster is malformed" through the public API would put an arm at every
/// call site that no caller could ever act on.
static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    load::load(include_str!("specs.yaml")).unwrap_or_else(|e| panic!("specs.yaml: {e}"))
});

/// What warpllm knows about a `model_str` such as `"openai/gpt-5.6"`: the
/// provider that serves it, and the model itself.
///
/// Two rows rather than one merged spec, because the roster keeps them at two
/// levels — how the API is reached is the provider's; the upstream name, the
/// surfaces served, and the published limits are the model's. The transport is
/// stated once no matter how many models a provider serves.
///
/// Which WIRE FORMAT is spoken is the model's, not the provider's: an
/// [`crate::Api`] names its own protocol, so the surfaces a model lists say it
/// already.
///
/// What a request can ASK FOR is the model's alone. A provider is a host, and
/// one host commonly serves chat completions, embeddings, and moderation from
/// disjoint sets of models, so there is nothing at that level to route on.
///
/// The key matches exactly or not at all. Nothing routes on a guess — no
/// pattern, no catch-all, no fallback — so a name no entry claims is an error,
/// and a typo cannot reach a provider as a live, billed request. That includes
/// a bare name with no `/`, since silently assuming OpenAI becomes a footgun
/// once many providers exist.
///
/// # Errors
///
/// [`Error::InvalidModel`] if nothing is registered for `model_str`.
///
/// # Examples
///
/// ```
/// let (provider, model) = warpllm::fetch_model("openai/gpt-5.6")?;
/// assert_eq!(provider.name(), "openai");
/// assert_eq!(model.model(), "gpt-5.6");
/// // The model's own list is what a request is routed on.
/// assert!(model.supports_api(warpllm::Api::OpenAiCompatChatCompletions));
///
/// // A name nobody registered is an error, never a guess.
/// assert!(warpllm::fetch_model("openai/nonexistent").is_err());
/// # Ok::<(), warpllm::Error>(())
/// ```
pub fn fetch_model(model_str: &str) -> Result<(&'static ProviderSpec, &'static ModelSpec)> {
    resolve(&REGISTRY, model_str).ok_or_else(|| Error::InvalidModel {
        given: model_str.to_string(),
    })
}

/// Every provider on the roster, in no particular order.
///
/// Unlike [`fetch_model`] this answers a question about the ROSTER rather than
/// about one request: which providers exist at all, so that something can be
/// worked out per provider before any model is named. The client uses it once,
/// at construction, to find which providers the environment can authenticate.
///
/// Not public. A caller reaches a provider through the model it serves, and
/// handing out the whole roster would make the shipped list an API that could
/// not gain an entry without a semver argument.
pub(crate) fn providers() -> impl Iterator<Item = &'static ProviderSpec> {
    REGISTRY.providers.values()
}

/// The roster's row for this provider, by bare name.
///
/// The one lookup that starts from a provider rather than from a model.
/// [`fetch_model`] is still how a REQUEST reaches a provider — nothing routes
/// on a bare name, and no guess is made from one. This answers the other
/// question: a client declaring which providers it serves names them directly,
/// and the declaration has to be checked against the roster that will serve it.
///
/// Not public, for the reason [`providers`] is not: the shipped list would
/// become an API that could not gain an entry without a semver argument. A
/// caller states a name and hears whether it worked.
pub(crate) fn provider(name: &str) -> Option<&'static ProviderSpec> {
    REGISTRY.providers.get(name)
}

/// The model row filed under `model_str`, and the provider row it names.
///
/// One hash lookup, then a second. There is no second chance at the first: a
/// key the table does not hold is a model warpllm does not serve, and the
/// caller hears that rather than a provider hearing a guess.
///
/// Split out from [`fetch_model`] so the matching can be tested against a
/// fixture roster rather than only the shipped one.
fn resolve<'a>(
    registry: &'a Registry,
    model_str: &str,
) -> Option<(&'a ProviderSpec, &'a ModelSpec)> {
    let model = registry.models.get(model_str)?;
    let provider = registry
        .providers
        .get(&model.provider)
        .expect("load registers a provider for every model it holds");
    Some((provider, model))
}

#[cfg(test)]
mod tests {
    use super::testing::{CHAT, clean, keys, model, models, providers, with};
    use super::*;
    use crate::types::Api;

    // ------------------------------------------------------------- matching

    /// The rule the roster documents: a key matches its own entry or nothing
    /// at all, and what ships upstream is that entry's name.
    #[test]
    fn a_key_matches_its_own_entry_or_nothing() {
        let registry = clean(&with(&format!(
            "      demo/pinned:\n{CHAT}        model: pinned-2024\n{}",
            model("demo/plain")
        )));
        assert_eq!(
            resolve(&registry, "demo/pinned").unwrap().1.model(),
            "pinned-2024"
        );
        assert_eq!(resolve(&registry, "demo/plain").unwrap().1.model(), "plain");
        // One character off, and there is nothing to fall back to.
        assert!(resolve(&registry, "demo/pinnedd").is_none());
        assert!(resolve(&registry, "demo/PLAIN").is_none());
    }

    /// Whichever entry matched, the provider row comes back with it — that is
    /// the whole point of handing back both halves.
    #[test]
    fn a_match_carries_its_provider() {
        let registry = clean(&with(&model("demo/plain")));
        let (provider, _) = resolve(&registry, "demo/plain").unwrap();
        assert_eq!(provider.name(), "demo");
        assert_eq!(provider.base_url(), "https://api.demo.test/v1");
    }

    /// The registry is closed, so an unlisted name is an error — the whole
    /// reason a typo cannot become a billed upstream request.
    ///
    /// A pattern is just another unlisted name. Nothing reads `*` as anything
    /// but a character, so asking for one matches exactly as much as asking
    /// for a misspelling: nothing.
    #[test]
    fn an_unlisted_name_is_rejected() {
        let registry = clean(&with(&model("demo/plain")));
        assert!(resolve(&registry, "demo/plain").is_some());
        for unlisted in ["demo/unlisted", "demo/*", "demo/pl*", "*"] {
            assert!(resolve(&registry, unlisted).is_none(), "`{unlisted}`");
        }
    }

    /// A name grouped under a prefix is registered like any other, and the
    /// grouping buys it no fallback.
    #[test]
    fn a_grouped_name_matches_only_its_own_entry() {
        let registry = clean(&with(&models(&["demo/org/one", "demo/plain"])));
        assert!(resolve(&registry, "demo/org/one").is_some());
        assert!(resolve(&registry, "demo/org/two").is_none());
        assert!(resolve(&registry, "demo/org").is_none());
    }

    /// A provider is not a model. Its key names transport, and no request
    /// routes to it.
    #[test]
    fn a_provider_name_is_not_routable() {
        let registry = clean(&with(&model("demo/plain")));
        assert!(resolve(&registry, "demo/").is_none());
        assert!(resolve(&registry, "demo").is_none());
    }

    // -------------------------------------------------------- shipped roster

    /// The shipped roster is the one case that has to keep working, and the
    /// only place every policy is enforced against real content. This test is
    /// what replaced the build-time gate: it is why [`REGISTRY`] can panic on a
    /// malformed file without that ever reaching anyone.
    #[test]
    fn the_shipped_registry_loads_and_lints() {
        let yaml = include_str!("specs.yaml");
        lint::check(yaml).unwrap_or_else(|e| panic!("specs.yaml: {e}"));
        let registry = load::load(yaml).unwrap();
        assert_eq!(
            providers(&registry),
            vec![
                "deepseek",
                "kimi",
                "mistral",
                "openai",
                "opencode",
                "openrouter"
            ]
        );
        assert_eq!(
            keys(&registry),
            vec![
                "deepseek/deepseek-v4-flash",
                "deepseek/deepseek-v4-pro",
                "kimi/kimi-k2.6",
                "kimi/kimi-k2.7-code",
                "kimi/kimi-k2.7-code-highspeed",
                "kimi/kimi-k3",
                "mistral/codestral-2501",
                "mistral/ministral-8b-2410",
                "mistral/mistral-large-2411",
                "mistral/pixtral-large-2411",
                "openai/gpt-4.1",
                "openai/gpt-4.1-mini",
                "openai/gpt-4o",
                "openai/gpt-4o-mini",
                "openai/gpt-5",
                "openai/gpt-5-mini",
                "openai/gpt-5-nano",
                "openai/gpt-5.1",
                "openai/gpt-5.2",
                "openai/gpt-5.4",
                "openai/gpt-5.4-mini",
                "openai/gpt-5.4-nano",
                "openai/gpt-5.5",
                "openai/gpt-5.6",
                "openai/gpt-5.6-luna",
                "openai/gpt-5.6-sol",
                "openai/gpt-5.6-terra",
                "openai/o3",
                "opencode/big-pickle",
                "opencode/deepseek-v4-flash",
                "opencode/deepseek-v4-flash-free",
                "opencode/deepseek-v4-pro",
                "opencode/glm-5.1",
                "opencode/glm-5.2",
                "opencode/hy3-free",
                "opencode/kimi-k2.6",
                "opencode/kimi-k2.7-code",
                "opencode/kimi-k3",
                "opencode/laguna-s-2.1-free",
                "opencode/mimo-v2.5-free",
                "opencode/minimax-m2.7",
                "opencode/minimax-m3",
                "opencode/nemotron-3-ultra-free",
                "opencode/nemotron-3.5-lightning-free",
                "openrouter/anthropic/claude-opus-4",
                "openrouter/anthropic/claude-sonnet-4",
                "openrouter/auto",
                "openrouter/google/gemini-2.5-pro",
                "openrouter/moonshotai/kimi-k2.6",
                "openrouter/moonshotai/kimi-k2.7-code",
                "openrouter/moonshotai/kimi-k3",
                "openrouter/openai/gpt-5.6",
                "openrouter/~deepseek/deepseek-v4-flash-latest",
            ]
        );
    }

    #[test]
    fn resolves_openai() {
        let (provider, model) = fetch_model("openai/gpt-5.6").unwrap();
        assert_eq!(provider.name(), "openai");
        assert_eq!(provider.base_url(), "https://api.openai.com/v1");
        assert_eq!(model.model(), "gpt-5.6");
    }

    #[test]
    fn resolves_deepseek() {
        let (provider, model) = fetch_model("deepseek/deepseek-v4-flash").unwrap();
        assert_eq!(provider.name(), "deepseek");
        assert_eq!(provider.base_url(), "https://api.deepseek.com");
        assert_eq!(provider.env_api_key(), Some("DEEPSEEK_API_KEY"));
        assert_eq!(model.model(), "deepseek-v4-flash");
    }

    #[test]
    fn resolves_openrouter() {
        let (provider, model) = fetch_model("openrouter/anthropic/claude-sonnet-4").unwrap();
        assert_eq!(provider.name(), "openrouter");
        assert_eq!(provider.base_url(), "https://openrouter.ai/api/v1");
        assert_eq!(provider.env_api_key(), Some("OPENROUTER_API_KEY"));
        assert!(model.supports_api(Api::OpenAiCompatChatCompletions));
        assert_eq!(model.model(), "anthropic/claude-sonnet-4");
    }

    /// Every provider the roster holds is reachable by its bare name, and
    /// hands back the same row `fetch_model` routes through — one roster, not
    /// two ways of reading it.
    #[test]
    fn every_roster_name_resolves_to_its_own_spec() {
        // Qualified: this module's `providers` is shadowed in here by the
        // `testing` helper of the same name, which reads a fixture registry.
        for spec in super::providers() {
            let found = provider(spec.name()).expect("a roster name resolves");
            assert!(
                std::ptr::eq(found, spec),
                "`{}` got a second copy of its provider row",
                spec.name()
            );
        }
    }

    /// A bare name is matched as exactly as a model key is: no case folding,
    /// no prefix, no guess. A declaration naming `OpenAI` is a typo and has to
    /// hear so.
    #[test]
    fn a_name_the_roster_does_not_hold_resolves_to_nothing() {
        for unheld in ["openia", "OpenAI", "openai/", "", "*"] {
            assert!(provider(unheld).is_none(), "`{unheld}`");
        }
    }

    /// OpenRouter's slugs are two segments (`anthropic/claude-sonnet-4`), and
    /// the key's last-segment default would truncate one to `claude-sonnet-4`.
    /// The `model:` field is what ships the FULL slug, so a key under
    /// `openrouter/` has to prove it carries an explicit model.
    #[test]
    fn every_openrouter_entry_ships_its_full_slug_upstream() {
        for (key, spec) in &REGISTRY.models {
            if let Some(slug) = key.strip_prefix("openrouter/") {
                assert_eq!(
                    spec.model(),
                    slug,
                    "`{key}` shipped a truncated slug; the model field must carry \
                     the whole two-segment OpenRouter identifier"
                );
            }
        }
    }

    /// The whole GPT-5.6 family is routable and answered by one provider row;
    /// the entries exist only to register the names.
    #[test]
    fn the_gpt_5_6_family_is_registered() {
        let (base, _) = fetch_model("openai/gpt-5.6").unwrap();
        for name in ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"] {
            let (provider, model) = fetch_model(&format!("openai/{name}")).unwrap();
            assert_eq!(model.model(), name);
            assert!(
                std::ptr::eq(provider, base),
                "`{name}` got a second copy of its provider"
            );
        }
    }

    /// Every shipped entry lists EXACTLY chat completions — the one surface
    /// warpllm implements. Iterates [`REGISTRY`] so a new entry is covered
    /// without being added here.
    ///
    /// Both halves earn their keep, in opposite directions:
    ///
    /// Serving it at all is what the client can route. An entry that did not
    /// would be admitted by the roster and refused by `validate_api`, so this
    /// is where that shows up rather than at request time. When a genuine
    /// non-chat model lands — an embeddings or moderation name — it belongs in
    /// an exclusion here rather than being quietly made to serve chat.
    ///
    /// Serving nothing MORE is the roster rule that a surface is listed only
    /// once warpllm can serve it. `openai_compat_responses` is real, and every
    /// OpenAI model here really does serve it, and it is still absent — because
    /// there is no code behind it, so an entry claiming it would record a
    /// capability nothing can act on. This is what catches it being added
    /// early.
    ///
    /// Both implemented surfaces are listed for every model because every
    /// provider on the roster documents `stream` on the chat-completions
    /// endpoint rather than per model — `specs.yaml` cites each one. A provider
    /// whose models genuinely differ belongs in an exclusion here, the same way
    /// a non-chat model would.
    ///
    /// It also means the shipped roster can no longer show two models of one
    /// provider differing. That is proved over fixtures instead, by
    /// `load::tests::two_models_of_one_provider_serve_different_surfaces`.
    #[test]
    fn every_shipped_model_serves_exactly_the_implemented_surfaces() {
        assert!(!REGISTRY.models.is_empty(), "the registry is empty");
        for (model_str, spec) in &REGISTRY.models {
            assert_eq!(
                spec.supported_apis(),
                [
                    SupportedApi {
                        api: Api::OpenAiCompatChatCompletions
                    },
                    SupportedApi {
                        api: Api::OpenAiCompatChatCompletionsStream
                    }
                ],
                "`{model_str}` lists a surface warpllm does not implement, or \
                 omits one it does"
            );
        }
    }

    /// Every registered model resolves to its own entry. Iterates the registry
    /// so a new `specs.yaml` entry is covered automatically.
    #[test]
    fn registered_models_resolve_to_their_own_entry() {
        assert!(!REGISTRY.models.is_empty(), "the registry is empty");
        for (model_str, spec) in &REGISTRY.models {
            let (provider, resolved) = fetch_model(model_str).unwrap();
            assert_eq!(resolved.model(), spec.model());
            assert_eq!(provider.name(), spec.provider);
            assert_eq!(
                resolved.capabilities().max_concurrent_requests(),
                spec.capabilities().max_concurrent_requests()
            );
        }
    }

    /// What per-model entries are FOR: the V4 pair shares one provider row and
    /// differs only in the limit that motivated splitting them.
    #[test]
    fn model_entries_carry_only_what_differs() {
        let (flash_provider, flash) = fetch_model("deepseek/deepseek-v4-flash").unwrap();
        let (pro_provider, pro) = fetch_model("deepseek/deepseek-v4-pro").unwrap();
        assert!(std::ptr::eq(flash_provider, pro_provider));
        // The divergence that motivated per-model entries: 5x the concurrency.
        assert_eq!(flash.capabilities().max_concurrent_requests(), Some(2500));
        assert_eq!(pro.capabilities().max_concurrent_requests(), Some(500));
    }

    /// The registry is closed: an unlisted name is an error, not a fallback.
    /// `openai/*` is in the list because a pattern matches nothing either.
    ///
    /// `openai/gpt-5-pro` is real upstream and still rejected here: it serves
    /// the Responses API and not chat completions, so the roster leaves it out
    /// rather than register a name every request would fail on.
    #[test]
    fn unregistered_models_are_rejected() {
        for model_str in ["openai/gpt-5-pro", "deepseek/deepseek-v5", "openai/*"] {
            let msg = fetch_model(model_str).unwrap_err().to_string();
            assert!(msg.contains(model_str), "{msg}");
            assert!(msg.contains("no registered model spec"), "{msg}");
        }
    }

    /// OpenCode Zen is the one provider on the roster serving its catalog
    /// across FOUR protocols, and only one of them is a surface warpllm has.
    /// Zen states the endpoint per model and nowhere else — there is nothing
    /// in a request or a response that says which one a name sits on — so a
    /// `/responses` or `/messages` model added here would load, lint, pass
    /// `every_shipped_model_serves_exactly_the_implemented_surfaces`, and then
    /// send a live, billed request to an endpoint that does not serve it.
    ///
    /// This is the only gate on that. It is a prefix check rather than a list
    /// of the 40-odd names, so a model Zen adds to a family it already serves
    /// elsewhere is caught without this test being edited.
    #[test]
    fn no_opencode_entry_sits_on_a_surface_warpllm_cannot_reach() {
        // `/zen/v1/responses`: every GPT, plus Grok and Muse.
        // `/zen/v1/messages`: Claude and Qwen.
        // `/zen/v1/models/<id>`: Gemini.
        let elsewhere = ["gpt-", "grok-", "muse-", "claude-", "qwen", "gemini-"];
        for key in REGISTRY.models.keys() {
            let Some(name) = key.strip_prefix("opencode/") else {
                continue;
            };
            for prefix in elsewhere {
                assert!(
                    !name.starts_with(prefix),
                    "`{key}`: OpenCode Zen serves `{prefix}…` models on an endpoint \
                     other than /chat/completions, so this entry would route a paid \
                     request nowhere. Check the endpoint column of \
                     <https://opencode.ai/docs/zen/> before registering a Zen model."
                );
            }
        }
        // The registry is still closed to them by name, the same as any
        // unlisted model.
        assert!(fetch_model("opencode/gpt-5-nano").is_err());
        assert!(fetch_model("opencode/claude-opus-5").is_err());
    }

    /// A slash-containing name needs an entry of its own, and the registry
    /// has none under `openai/org/`.
    #[test]
    fn unregistered_grouped_names_are_rejected() {
        assert!(fetch_model("openai/org/custom-model").is_err());
    }

    #[test]
    fn rejects_bare_model() {
        let msg = fetch_model("gpt-5.6").unwrap_err().to_string();
        assert!(msg.contains("no registered model spec"), "{msg}");
    }

    #[test]
    fn rejects_unknown_provider() {
        let msg = fetch_model("mistral/large").unwrap_err().to_string();
        assert!(msg.contains("mistral/large"), "{msg}");
        assert!(msg.contains("no registered model spec"), "{msg}");
    }

    /// A provider key is transport, not a model, so it cannot be routed to.
    #[test]
    fn rejects_a_provider_key() {
        assert!(fetch_model("openai/").is_err());
        assert!(fetch_model("openai").is_err());
    }

    /// Nothing but routable entries reaches the model table: every key
    /// carrying the provider it is filed under, every row agreeing with the
    /// key it sits at, and no key reaching for a pattern.
    ///
    /// That last one is asserted HERE and nowhere else. `load` reads a key
    /// literally and has no opinion about `*`, so `openai/*` would register a
    /// model named `*` and quietly serve nothing — this is what keeps the
    /// shipped file from acquiring one, whether by a stale roster or by
    /// somebody expecting the catch-all warpllm used to have.
    ///
    /// The fixtures elsewhere assert the same over hand-written rosters. This
    /// asserts it over [`REGISTRY`], which is the table a caller actually
    /// routes against — a resolver that passed every other test and still
    /// built a bad table would fail here and only here.
    #[test]
    fn the_table_holds_only_routable_entries() {
        for (model_str, spec) in &REGISTRY.models {
            let (provider, name) = model_str
                .split_once('/')
                .unwrap_or_else(|| panic!("`{model_str}` names no provider"));
            assert_eq!(spec.provider, provider, "`{model_str}`");
            assert!(REGISTRY.providers.contains_key(provider), "`{model_str}`");
            assert!(!name.is_empty(), "`{model_str}` names no model");
            assert!(
                !model_str.contains('*') && !spec.model().contains('*'),
                "`{model_str}`: a `*` reached the table, which matches nothing"
            );
        }
    }
}
