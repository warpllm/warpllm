//! Reading `specs.yaml` and turning it into the two tables the lookup answers
//! from.
//!
//! Everything here fails on what leaves no usable spec: syntax, an unknown
//! field, a key that disagrees with where it sits, a required field nobody
//! set. Whether a roster that loads cleanly is any GOOD is `super::lint`'s
//! question, and it is a separate gate for that reason.
//!
//! The YAML schema lives here rather than on the types next door, which are
//! read surfaces. Keeping the two apart is what lets a `ProviderSpec` hold a
//! settled `base_url: String` while the file it came from is free to be
//! missing one — and be told so, by serde, with a line and a column.

use std::collections::{BTreeMap, HashMap};

use serde::Deserialize;

use super::intern::intern;
use super::types::{Capabilities, Credential, ModelSpec, ProviderSpec, Registry, SupportedApi};

/// The whole roster: providers, each holding the models routable under it.
///
/// Hashed, like the tables it becomes. Nothing is looked up in these and
/// nothing reads them in order — they are built once and drained straight
/// into the registry — so a sorted map would buy a string comparison per
/// level on every insert and nothing else. What keeps the FILE readable is
/// `lint`'s ordering check, which reads the text rather than any map.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryFile {
    providers: HashMap<String, ProviderEntry>,
}

/// One provider as written: transport only. Everything but `env_api_key` and
/// `models` is required: there is no inheritance and so nowhere else a value
/// could come from, which is what lets serde report a missing one against the
/// line it is missing from.
///
/// `deny_unknown_fields` is what turns the retired `protocol:` line into an
/// error rather than a silently ignored one — a surface names its own protocol
/// now, and a roster still recording it per provider is stale rather than
/// merely verbose.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderEntry {
    base_url: String,
    env_api_key: Option<String>,
    /// How this provider authenticates when it does NOT read a variable.
    ///
    /// Optional, and its only value is `none`, so the field exists to say one
    /// thing out loud: this host takes no credential. Omitting it keeps the
    /// meaning omission has always had — the roster records no way to
    /// authenticate this provider — which is why the two are separate fields
    /// rather than one with a fallback. A forgotten `env_api_key:` line must
    /// not quietly become an unauthenticated request.
    auth: Option<Auth>,
    /// `Option`, and defaulted, so that both ways of writing "no models yet" —
    /// omitting the key and leaving it empty — reach the lint, which says what
    /// is wrong with that in its own words. Neither is a load failure, because
    /// both leave a perfectly buildable pair of tables.
    #[serde(default)]
    models: Option<HashMap<String, ModelEntry>>,
}

/// The `auth:` vocabulary, closed at one word.
///
/// An enum rather than a `bool` because there is every reason to expect a
/// second scheme — a custom header, a query parameter — and `auth: none`
/// reads the same before and after one lands, where `unauthenticated: true`
/// would have to be deprecated to make room.
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Auth {
    None,
}

/// One model as written: which surfaces it serves, what it ships upstream if
/// that differs from its key, and whatever limits are published for it.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelEntry {
    /// REQUIRED, and the reason a model entry is never bare `{}`.
    ///
    /// Nothing is inherited from the provider, deliberately. A default of
    /// "everything my host serves" would be a claim the roster never made, and
    /// it fails open: an embeddings model under a host that also serves chat
    /// would be admitted to a chat request and fail as a 404 upstream. Silence
    /// stops the roster loading instead, and serde names the entry it is
    /// missing from.
    supported_apis: Vec<SupportedApi>,
    model: Option<String>,
    /// Names `blank` rather than relying on `#[serde(default)]`, which would
    /// need `Capabilities: Default` — the public constructor that type does
    /// not want.
    #[serde(default = "Capabilities::blank")]
    capabilities: Capabilities,
    deprecation_date: Option<String>,
}

/// One roster file, and what to call it in an error message.
///
/// The label rather than a path, because the shipped roster has no path — it
/// is `include_str!`'d — and a stranger reading "could not load the model
/// roster: specs.yaml: …" should not go looking on their disk for it.
pub(super) struct Source<'a> {
    pub label: &'a str,
    pub yaml: &'a str,
}

/// Reads one roster into the provider and model tables.
/// `Err` carries the message a contributor sees.
pub(super) fn load(yaml: &str) -> Result<Registry, String> {
    load_all(&[Source {
        label: "specs.yaml",
        yaml,
    }])
    .map(|(registry, _)| registry)
}

/// Folds several rosters into one pair of tables, later sources winning.
///
/// The unit of replacement is a WHOLE provider entry, models included. Nothing
/// in a roster is inherited — not between the two levels, not between models —
/// and a deep merge would break that in the way that matters most: it would
/// produce a provider spec that appears in neither file, so "which `base_url`
/// did my model actually get" would stop being answerable by reading one. A
/// user naming `openai:` is describing their `openai`, whole.
///
/// Every entry is merged before any is built, so a provider that loses only
/// ever contributed its name to the map — its models never reach the table and
/// cannot outlive the entry that declared them.
///
/// Also hands back the names that were replaced, in order. Reporting them
/// rather than logging them here keeps this a pure function, and makes "an
/// override is announced" a plain assertion rather than a test that has to
/// capture a `tracing` subscriber.
pub(super) fn load_all(sources: &[Source]) -> Result<(Registry, Vec<String>), String> {
    // Sorted, so a file with two bad providers names the same one every run.
    // The build below is where a message comes from, and a `HashMap` there
    // would pick whichever came up first.
    let mut merged: BTreeMap<String, (&str, ProviderEntry)> = BTreeMap::new();
    let mut replaced = Vec::new();
    for source in sources {
        let file = parse(source.yaml).map_err(|e| format!("{}: {e}", source.label))?;
        for (name, entry) in file.providers {
            if merged.insert(name.clone(), (source.label, entry)).is_some() {
                replaced.push(name);
            }
        }
    }

    let mut registry = Registry::default();
    for (name, (label, entry)) in merged {
        build_provider(&mut registry, &name, entry).map_err(|e| format!("{label}: {e}"))?;
    }
    replaced.sort_unstable();
    Ok((registry, replaced))
}

/// One provider entry and its models, checked and filed into both tables.
fn build_provider(registry: &mut Registry, name: &str, entry: ProviderEntry) -> Result<(), String> {
    validate_provider(name)?;
    let ProviderEntry {
        base_url,
        env_api_key,
        auth,
        models,
    } = entry;
    let credential = credential(name, env_api_key, auth)?;
    for (key, model) in models.unwrap_or_default() {
        let spec = build(&key, name, model).map_err(|e| format!("`{key}`: {e}"))?;
        registry.models.insert(key, spec);
    }
    registry.providers.insert(
        name.to_string(),
        ProviderSpec {
            name: intern(name),
            base_url,
            credential,
        },
    );
    Ok(())
}

/// The two authentication fields settled into the one thing they describe.
///
/// Exactly one combination is illegal, and it is the one that contradicts
/// itself: naming a variable while declaring that nothing is sent. Saying
/// neither is legal and always has been — it is a provider whose key plumbing
/// has not landed.
fn credential(
    name: &str,
    env_api_key: Option<String>,
    auth: Option<Auth>,
) -> Result<Credential, String> {
    match (env_api_key, auth) {
        (Some(_), Some(Auth::None)) => Err(format!(
            "`{name}`: `auth: none` says this provider is sent no credential, so \
             there is no variable for it to read — drop whichever of the two \
             lines is wrong"
        )),
        (Some(var), None) => Ok(Credential::EnvVar(intern(&var))),
        (None, Some(Auth::None)) => Ok(Credential::NotRequired),
        (None, None) => Ok(Credential::Unavailable),
    }
}

/// Two passes over a few kilobytes. Only `Value`'s own deserializer rejects
/// duplicate map keys — a `HashMap` silently keeps the last — and only
/// `Value` preserves the order keys appear in the file, which is what the
/// sort check reads. The typed pass is what attaches line and column to type
/// and unknown-field errors. Each owns what it reports best.
fn parse(yaml: &str) -> Result<RegistryFile, String> {
    let _duplicate_key_check: yaml_serde::Value =
        yaml_serde::from_str(yaml).map_err(|e| e.to_string())?;
    yaml_serde::from_str(yaml).map_err(|e| e.to_string())
}

/// One model entry, checked against the key it sits under. The key settles the
/// one thing the entry itself may leave unstated: the name that ships upstream.
fn build(key: &str, provider: &str, entry: ModelEntry) -> Result<ModelSpec, String> {
    validate_model(key, provider)?;
    let name = key
        .rsplit_once('/')
        .expect("the prefix check found a `/`")
        .1;
    Ok(ModelSpec {
        provider: provider.to_string(),
        model: entry.model.unwrap_or_else(|| name.to_string()),
        supported_apis: entry.supported_apis,
        capabilities: entry.capabilities,
        deprecation_date: entry.deprecation_date,
    })
}

// -------------------------------------------------------------------- keys

/// A provider is one segment: it is the whole first part of a `model_str` and
/// holds nothing above or below it.
fn validate_provider(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("a provider name is empty".into());
    }
    if name.contains('/') {
        return Err(format!(
            "`{name}`: a provider name is one segment and carries no `/` — write \
             `{}`, and file its models under its own `models:` map",
            name.trim_end_matches('/')
        ));
    }
    Ok(())
}

/// Everything checkable about a model key: that it agrees with the provider
/// holding it, and that every segment of it is a name.
fn validate_model(key: &str, provider: &str) -> Result<(), String> {
    let Some(name) = key
        .strip_prefix(provider)
        .and_then(|rest| rest.strip_prefix('/'))
    else {
        return Err(format!(
            "a model key is the whole string a caller routes with, so one under \
             provider `{provider}` has to start with `{provider}/`"
        ));
    };
    if name.is_empty() {
        return Err(format!(
            "nothing follows the `{provider}/` prefix, so this key names no model"
        ));
    }
    // A key is read literally, every character of it: there are no patterns
    // to interpret, so the only thing a segment can be wrong about is being
    // absent.
    if name.split('/').any(str::is_empty) {
        return Err("an empty path segment".into());
    }
    // Read literally, `*` is a character like any other, so this key would
    // register a model named `*` and then serve nothing — the one mistake a
    // roster can make that looks like it worked. The registry is closed by
    // design, and it stays closed for a host you own: a name nobody listed is
    // a name nothing routes.
    if key.contains('*') {
        return Err(
            "a `*` is matched literally, not as a pattern, so this key registers \
             one model named `*` and nothing else. The roster is closed — write \
             each model out under its own key"
                .into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::testing::{
        CHAT, LOCAL_ROSTER, OTHER_PROVIDER, PROVIDER, clean, keys, merge, merged, model, models,
        providers, with,
    };
    use super::*;
    use crate::types::Api;

    /// `PROVIDER` with one line dropped, for the cases that ask what happens
    /// when a required field is not there at all.
    fn without(field: &str) -> String {
        PROVIDER
            .lines()
            .filter(|line| !line.trim_start().starts_with(&format!("{field}:")))
            .fold(String::new(), |acc, line| acc + line + "\n")
    }

    // ------------------------------------------------------ the two tables

    /// The split itself: a provider lands in one table, its models in the
    /// other, and neither holds the other's rows.
    #[test]
    fn providers_and_models_land_in_their_own_tables() {
        let registry = clean(&with(&models(&["demo/one", "demo/two"])));
        assert_eq!(providers(&registry), vec!["demo"]);
        assert_eq!(keys(&registry), vec!["demo/one", "demo/two"]);
    }

    /// Transport is stated once for the provider and read from there, so two
    /// models of one provider are answered by the same row rather than by two
    /// copies of it.
    #[test]
    fn a_providers_transport_is_stored_once() {
        let registry = clean(&with(&models(&["demo/one", "demo/two"])));
        let provider = registry.providers.get("demo").unwrap();
        assert_eq!(provider.name(), "demo");
        assert_eq!(provider.base_url(), "https://api.demo.test/v1");
        assert_eq!(provider.env_api_key(), Some("DEMO_API_KEY"));
        for key in ["demo/one", "demo/two"] {
            assert_eq!(registry.models.get(key).unwrap().provider, "demo");
        }
    }

    // ------------------------------------------------------- authentication

    /// The self-hosted case, and the reason `auth` exists: a host on a private
    /// network takes no credential, and says so deliberately.
    #[test]
    fn auth_none_declares_a_provider_that_takes_no_credential() {
        let registry = clean(LOCAL_ROSTER);
        let provider = registry.providers.get("local").unwrap();
        assert!(provider.unauthenticated());
        // Nothing to read, and nothing to suggest reading.
        assert_eq!(provider.env_api_key(), None);
    }

    /// The one combination that contradicts itself. Both fields are optional
    /// and both may be absent; only naming a variable while declaring that
    /// nothing is sent is a roster nobody can act on.
    #[test]
    fn auth_none_beside_an_env_api_key_is_rejected() {
        let err = load(&format!(
            "{}{}",
            PROVIDER.replace("    models:\n", "    auth: none\n    models:\n"),
            model("demo/plain")
        ))
        .unwrap_err();
        assert!(err.contains("is sent no credential"), "{err}");
        assert!(err.contains("drop whichever of the two lines"), "{err}");
    }

    /// A closed vocabulary, like every other in this file: a typo cannot load,
    /// and the message names the word that would have worked.
    #[test]
    fn an_unknown_auth_value_is_rejected() {
        let err = load(
            &with(&model("demo/plain"))
                .replace("    env_api_key: DEMO_API_KEY\n", "    auth: bearer\n"),
        )
        .unwrap_err();
        assert!(err.contains("unknown variant `bearer`"), "{err}");
        assert!(err.contains("none"), "{err}");
    }

    /// A provider naming no environment variable is a valid roster, not an
    /// incomplete one: such a provider authenticates only with a key the
    /// caller supplies.
    ///
    /// Distinct from `auth: none`, and that is the point of having both:
    /// silence still means "this cannot be authenticated", so a forgotten line
    /// on a paid provider never becomes an unauthenticated request.
    #[test]
    fn a_provider_may_name_no_env_api_key() {
        let registry = clean(&format!(
            "{}{}",
            without("env_api_key"),
            model("demo/plain")
        ));
        let provider = registry.providers.get("demo").unwrap();
        assert_eq!(provider.env_api_key(), None);
        assert!(
            !provider.unauthenticated(),
            "silence must not be read as `auth: none`"
        );
        // Everything else still resolved, so this is an absence and not a
        // half-loaded entry.
        assert_eq!(provider.base_url(), "https://api.demo.test/v1");
    }

    // --------------------------------------------------- folding two rosters

    /// The ordinary case a roster file is written for: a second source names a
    /// provider the first does not, and both survive.
    #[test]
    fn a_second_roster_adds_its_providers() {
        let (registry, replaced) = merged(&[&with(&model("demo/plain")), LOCAL_ROSTER]);
        assert_eq!(providers(&registry), vec!["demo", "local"]);
        assert_eq!(keys(&registry), vec!["demo/plain", "local/llama-3.3-70b"]);
        assert!(replaced.is_empty(), "nothing was replaced: {replaced:?}");
    }

    /// A provider entry is replaced WHOLE, models included. Nothing in a roster
    /// is inherited, so a half-merged entry would describe a provider that
    /// appears in neither file — and "which `base_url` did my model get" would
    /// stop being answerable by reading one.
    #[test]
    fn a_later_roster_replaces_a_provider_whole() {
        // `local` renamed to `demo` at the two places it is a KEY, leaving
        // `localhost` in the URL alone.
        let shadowing = LOCAL_ROSTER
            .replace("  local:", "  demo:")
            .replace("local/llama", "demo/llama");
        let (registry, replaced) =
            merged(&[&with(&models(&["demo/keep", "demo/plain"])), &shadowing]);
        assert_eq!(providers(&registry), vec!["demo"]);
        // The earlier entry's models are GONE, not merged alongside.
        assert_eq!(keys(&registry), vec!["demo/llama-3.3-70b"]);
        assert_eq!(
            registry.providers.get("demo").unwrap().base_url(),
            "http://localhost:8000/v1"
        );
        assert_eq!(replaced, vec!["demo"]);
    }

    /// The replacement is reported so a client can say so out loud. Silently
    /// shadowing a provider somebody thought they were still using is the one
    /// outcome nobody could debug.
    #[test]
    fn every_replacement_is_reported_once() {
        let (_, replaced) = merged(&[
            &format!("{}{OTHER_PROVIDER}", with(&model("demo/plain"))),
            &format!(
                "{}{}",
                with(&model("demo/replaced")),
                OTHER_PROVIDER.replace("api.other.test", "api.replaced.test")
            ),
        ]);
        assert_eq!(replaced, vec!["demo", "other"]);
    }

    /// A message names the file it came from, which is the whole reason a
    /// source carries a label: a stranger with a bad roster must not be sent
    /// to read warpllm's own.
    #[test]
    fn a_failure_names_the_source_it_came_from() {
        let err = merge(&[&with(&model("demo/plain")), "providers: [not, a, map]\n"]).unwrap_err();
        assert!(err.starts_with("second.yaml:"), "{err}");
    }

    /// One roster is the same fold with one source, so `load` and `load_all`
    /// cannot drift apart.
    #[test]
    fn one_source_folds_to_itself() {
        let yaml = with(&model("demo/plain"));
        assert_eq!(keys(&merged(&[&yaml]).0), keys(&load(&yaml).unwrap()));
    }

    // ---------------------------------------------------------- model rows

    /// An entry with nothing to say exists purely to make a name routable —
    /// the shape every GPT-5.6 entry uses.
    #[test]
    fn an_empty_entry_is_routable_and_records_no_limits() {
        let registry = clean(&with(&model("demo/plain")));
        let spec = registry.models.get("demo/plain").unwrap();
        assert_eq!(spec.model(), "plain");
        assert_eq!(spec.capabilities().max_input_tokens(), None);
        assert_eq!(spec.capabilities().max_concurrent_requests(), None);
        assert_eq!(spec.deprecation_date(), None);
    }

    /// A recorded retirement reaches the spec verbatim. Nothing else reads the
    /// field, so this is what keeps it from silently ceasing to load.
    #[test]
    fn a_deprecation_date_is_read_when_present() {
        let registry = clean(&with(&format!(
            "      demo/plain:\n{CHAT}        deprecation_date: \"2026-10-23\"\n"
        )));
        let spec = registry.models.get("demo/plain").unwrap();
        assert_eq!(spec.deprecation_date(), Some("2026-10-23"));
    }

    /// Limits are per model and nothing else is: the shipped V4 pair is two
    /// entries for exactly this reason.
    #[test]
    fn capabilities_are_read_per_model() {
        let registry = clean(&with(&format!(
            "      demo/fast:\n{CHAT}        capabilities:\n          \
             max_concurrent_requests: 2500\n      demo/slow:\n{CHAT}        \
             capabilities:\n          max_concurrent_requests: 500\n"
        )));
        let caps = |key| registry.models.get(key).unwrap().capabilities();
        assert_eq!(caps("demo/fast").max_concurrent_requests(), Some(2500));
        assert_eq!(caps("demo/slow").max_concurrent_requests(), Some(500));
        assert_eq!(caps("demo/fast").max_input_tokens(), None);
    }

    // ------------------------------------------------------ supported_apis

    /// The list is read as written, in order, and each entry carries its own
    /// (empty) settings.
    #[test]
    fn a_models_surfaces_are_read_from_its_own_entry() {
        let registry = clean(&with(concat!(
            "      demo/plain:\n",
            "        supported_apis:\n",
            "          - {api: openai_compat_chat_completions}\n",
            "          - {api: openai_compat_responses}\n",
        )));
        assert_eq!(
            registry.models.get("demo/plain").unwrap().supported_apis(),
            [
                SupportedApi {
                    api: Api::OpenAiCompatChatCompletions
                },
                SupportedApi {
                    api: Api::OpenAiCompatResponses
                },
            ]
        );
    }

    /// The point of the whole field: one provider, two models, different
    /// surfaces. Neither borrows anything from the other or from the host.
    #[test]
    fn two_models_of_one_provider_serve_different_surfaces() {
        let registry = clean(&with(concat!(
            "      demo/chat-only:\n",
            "        supported_apis:\n",
            "          - {api: openai_compat_chat_completions}\n",
            "      demo/responses-only:\n",
            "        supported_apis:\n",
            "          - {api: openai_compat_responses}\n",
        )));
        let apis = |key| registry.models.get(key).unwrap().supported_apis().to_vec();
        assert_eq!(
            apis("demo/chat-only"),
            [SupportedApi {
                api: Api::OpenAiCompatChatCompletions
            }]
        );
        assert_eq!(
            apis("demo/responses-only"),
            [SupportedApi {
                api: Api::OpenAiCompatResponses
            }]
        );
        // The one that does not serve chat completions says so by not holding
        // the surface — which is what the client's gate reads.
        assert!(
            !registry
                .models
                .get("demo/responses-only")
                .unwrap()
                .supports_api(Api::OpenAiCompatChatCompletions)
        );
    }

    /// Required, with nothing to fall back on. This is the guard that keeps a
    /// silent entry from meaning "everything my host serves" — a claim the
    /// roster never made, and one that fails open.
    #[test]
    fn a_model_naming_no_surfaces_is_rejected() {
        let err = load(&with("      demo/plain: {}\n")).unwrap_err();
        assert!(err.contains("missing field"), "{err}");
        assert!(err.contains("supported_apis"), "{err}");
    }

    /// The closed vocabulary, at the level that writes it. A typo cannot load,
    /// so it never reaches a live provider as a 404.
    ///
    /// Aimed at the surface's PREVIOUS spelling, which is the misspelling a
    /// roster is most likely to carry: the protocol prefix moved from the
    /// provider's own `protocol:` line into the surface name, so `openai_…`
    /// became `openai_compat_…` and a stale entry says the old thing. The
    /// message has to name the new vocabulary, which is the migration.
    #[test]
    fn a_misspelled_surface_is_rejected() {
        let err = load(&with(
            "      demo/plain:\n        supported_apis:\n          - {api: openai_chat_completions}\n",
        ))
        .unwrap_err();
        assert!(
            err.contains("unknown variant `openai_chat_completions`"),
            "{err}"
        );
        // The message names the vocabulary, so the line can be fixed without
        // opening `types.rs`.
        assert!(err.contains("openai_compat_chat_completions"), "{err}");
    }

    /// An entry is a map, and `api` is the key it must carry. Leaving it out
    /// names no surface at all, which serde reports against the entry.
    #[test]
    fn an_entry_naming_no_api_is_rejected() {
        let err = load(&with(
            "      demo/plain:\n        supported_apis:\n          - {}\n",
        ))
        .unwrap_err();
        assert!(err.contains("missing field `api`"), "{err}");
    }

    /// An entry is as closed as the surface list itself. `api` is the only key
    /// one takes today, so anything beside it is a typo or a stale roster —
    /// not something to accept and drop.
    ///
    /// This is what `input_modalities` will land into: adding it makes this
    /// exact roster line valid, at every surface at once.
    #[test]
    fn an_unknown_key_beside_the_api_is_rejected() {
        let err = load(&with(concat!(
            "      demo/plain:\n",
            "        supported_apis:\n",
            "          - api: openai_compat_chat_completions\n",
            "            input_modalities: [text]\n",
        )))
        .unwrap_err();
        assert!(err.contains("unknown field"), "{err}");
        assert!(err.contains("input_modalities"), "{err}");
    }

    /// The block form of an entry is the same YAML as the inline one the
    /// roster writes, which is what makes opening one up to add a key a
    /// formatting choice rather than a migration.
    #[test]
    fn an_entry_reads_the_same_written_inline_or_as_a_block() {
        let inline = clean(&with(
            "      demo/plain:\n        supported_apis:\n          - {api: openai_compat_chat_completions}\n",
        ));
        let block = clean(&with(
            "      demo/plain:\n        supported_apis:\n          - api: openai_compat_chat_completions\n",
        ));
        assert_eq!(
            inline.models.get("demo/plain").unwrap().supported_apis(),
            block.models.get("demo/plain").unwrap().supported_apis()
        );
    }

    #[test]
    fn the_wire_name_defaults_to_the_keys_last_segment() {
        let registry = clean(&with(&model("demo/plain")));
        assert_eq!(registry.models.get("demo/plain").unwrap().model(), "plain");
    }

    #[test]
    fn an_explicit_model_beats_the_keys_last_segment() {
        let registry = clean(&with(&format!(
            "      demo/chat:\n{CHAT}        model: demo-chat-20240101\n"
        )));
        assert_eq!(
            registry.models.get("demo/chat").unwrap().model(),
            "demo-chat-20240101"
        );
    }

    /// A slash-containing name is one name, not a grouping: nothing inherits
    /// through it, and the wire name is still the last segment.
    #[test]
    fn a_slash_containing_name_is_one_name() {
        let registry = clean(&with(&model("demo/org/custom")));
        assert_eq!(
            registry.models.get("demo/org/custom").unwrap().model(),
            "custom"
        );
    }

    /// Every model row names a provider that is actually in the other table.
    /// The key check is what guarantees it, and this is the guarantee stated
    /// over a resolved roster.
    #[test]
    fn every_model_names_a_provider_that_exists() {
        let registry = clean(&format!("{}{OTHER_PROVIDER}", with(&model("demo/plain"))));
        for (key, spec) in &registry.models {
            let provider = registry
                .providers
                .get(&spec.provider)
                .unwrap_or_else(|| panic!("`{key}` names provider `{}`", spec.provider));
            assert_eq!(key.split('/').next(), Some(provider.name()));
        }
    }

    // ------------------------------------------------------- load failures
    //
    // Nothing below leaves a correct pair of tables to build, so each is
    // rejected.

    /// `protocol:` was a provider field until a surface started naming its own.
    /// A roster still carrying it is stale, not merely verbose, and has to hear
    /// so: silently ignoring the line would leave a fork believing it still
    /// chose the wire format, right up until it wrote one the surfaces
    /// contradict.
    #[test]
    fn a_stale_protocol_field_is_rejected() {
        let stale = with(&model("demo/plain")).replace(
            "    env_api_key: DEMO_API_KEY\n",
            "    env_api_key: DEMO_API_KEY\n    protocol: openai_compat\n",
        );
        let err = load(&stale).unwrap_err();
        assert!(err.contains("unknown field"), "{err}");
        assert!(err.contains("protocol"), "{err}");
        // The surviving fields are named, so the fix is to delete the line
        // rather than to go looking for what replaced it.
        assert!(err.contains("base_url"), "{err}");
    }

    #[test]
    fn misspelled_fields_are_rejected() {
        let err = load(&with("").replace("base_url:", "base_urls:")).unwrap_err();
        assert!(err.contains("unknown field"), "{err}");
        assert!(err.contains("base_url"), "{err}");
    }

    /// A model-level field is checked just as closely as a provider-level one.
    #[test]
    fn a_misspelled_capability_is_rejected() {
        let err = load(&with(
            "      demo/plain:\n        capabilities:\n          max_input_token: 128\n",
        ))
        .unwrap_err();
        assert!(err.contains("unknown field"), "{err}");
        assert!(err.contains("max_input_tokens"), "{err}");
    }

    /// A `HashMap` would keep the last silently; the `Value` pass is what
    /// makes this an error.
    #[test]
    fn duplicate_keys_are_rejected() {
        let err = load(&with(&models(&["demo/plain", "demo/plain"]))).unwrap_err();
        assert!(err.contains("duplicate"), "{err}");
        assert!(err.contains("demo/plain"), "{err}");
    }

    /// The accessors read this without a fallback, and with no inheritance
    /// left there is nowhere else it could come from — so a provider missing
    /// it has no spec to build. `base_url` is the only required provider field
    /// there is now that a surface names its own protocol.
    #[test]
    fn a_missing_provider_field_is_rejected() {
        let err = load(&format!("{}{}", without("base_url"), model("demo/plain"))).unwrap_err();
        assert!(err.contains("missing field"), "{err}");
        assert!(err.contains("base_url"), "{err}");
    }

    /// The key is the whole string a caller routes with. A bare name under a
    /// provider would resolve to nothing, and the message says what to write.
    #[test]
    fn a_model_key_missing_its_provider_prefix_is_rejected() {
        let err = load(&with(&model("plain"))).unwrap_err();
        assert!(err.contains("has to start with `demo/`"), "{err}");
    }

    /// Filed under the wrong provider it would be routable under a transport
    /// that is not its own.
    #[test]
    fn a_model_key_under_the_wrong_provider_is_rejected() {
        let err = load(&with(&model("other/plain"))).unwrap_err();
        assert!(err.contains("has to start with `demo/`"), "{err}");
    }

    #[test]
    fn a_key_that_names_no_model_is_rejected() {
        let err = load(&with(&model("demo/"))).unwrap_err();
        assert!(err.contains("names no model"), "{err}");
    }

    /// An empty segment is the one thing a key's insides can be wrong about,
    /// since every other character is read literally as part of a name.
    #[test]
    fn an_empty_path_segment_is_rejected() {
        let err = load(&with(&model("demo//custom"))).unwrap_err();
        assert!(err.contains("an empty path segment"), "{err}");
    }

    /// The wildcard somebody reaches for when they own the box, and the reason
    /// it has to be a load error rather than nothing.
    ///
    /// A key is read literally, so `demo/*` would register one model named `*`,
    /// route nothing, and look for all the world like it had worked — a 404 on
    /// their own hardware with no line to go and fix. The registry is closed,
    /// and now it says so at the moment the roster is written instead of at the
    /// moment a request fails.
    #[test]
    fn a_wildcard_in_a_model_key_is_rejected() {
        for key in ["demo/*", "demo/gpt-*", "demo/*/nano"] {
            let err = load(&with(&model(key))).unwrap_err();
            assert!(err.contains("matched literally"), "`{key}`: {err}");
            assert!(err.contains("under its own key"), "`{key}`: {err}");
        }
    }

    /// The old shape's `demo/:` key, and the message that migrates it.
    #[test]
    fn a_provider_name_carrying_a_slash_is_rejected() {
        let err = load(&PROVIDER.replace("  demo:", "  demo/:")).unwrap_err();
        assert!(err.contains("a provider name is one segment"), "{err}");
        assert!(err.contains("write `demo`"), "{err}");
    }
}
