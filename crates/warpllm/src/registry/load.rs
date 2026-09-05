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
use std::sync::LazyLock;

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
#[derive(Deserialize, Clone)]
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
#[derive(Deserialize, Clone)]
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
    /// Present-and-null is distinguished from absent, and nothing else here
    /// needs that — which is the whole reason for the second `Option`.
    ///
    /// The outer layer is WHETHER THE LINE WAS WRITTEN. It has to be, because
    /// only an entry that mentions no models at all inherits the ones it
    /// replaces (see [`load_all`]). A bare `models:` is a half-written line,
    /// not a decision to keep eighteen models somebody never listed, and
    /// reading the two the same way would let a forgotten key silently
    /// redirect every model under a shipped provider to a new host.
    ///
    /// So a written `models:` is always a statement, and a valueless one
    /// states nothing — [`build_provider`] settles it to an empty map, which
    /// reaches the lint and is refused there in its own words, exactly as
    /// `models: {}` is. Same judgement a valueless `auth:` gets, and for the
    /// same reason: when a line is half-typed, take the safe reading.
    #[serde(default, deserialize_with = "written")]
    models: Option<Option<HashMap<String, ModelEntry>>>,
}

/// Records that a field was WRITTEN, whatever it was written as.
///
/// `#[serde(default)]` leaves the outer `Option` `None` when the key is
/// absent; reaching this at all means the key is there, so the value — `null`
/// included — comes back wrapped.
fn written<'de, T, D>(deserializer: D) -> Result<Option<T>, D::Error>
where
    T: Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    T::deserialize(deserializer).map(Some)
}

/// The `auth:` vocabulary, closed at one word.
///
/// An enum rather than a `bool` because there is every reason to expect a
/// second scheme — a custom header, a query parameter — and `auth: none`
/// reads the same before and after one lands, where `unauthenticated: true`
/// would have to be deprecated to make room.
#[derive(Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
enum Auth {
    None,
}

/// One model as written: which surfaces it serves, what it ships upstream if
/// that differs from its key, and whatever limits are published for it.
#[derive(Deserialize, Clone)]
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
    .map(|fold| fold.registry)
}

/// What a fold produced, and what it wrote over on the way.
///
/// A struct rather than a tuple because the two kinds of override are not the
/// same event and must not be announced as one: [`Fold::replaced`] dropped
/// models, [`Fold::retargeted`] kept them.
pub(super) struct Fold {
    pub registry: Registry,
    /// Providers whose entry was replaced OUTRIGHT — the later entry listed
    /// its own models, so the earlier one's are gone.
    pub replaced: Vec<String>,
    /// Providers whose transport was replaced while their models carried
    /// over, because the later entry said nothing about models.
    pub retargeted: Vec<String>,
}

/// Folds several rosters into one pair of tables, later sources winning.
///
/// The unit of replacement is a WHOLE provider entry. Nothing in a roster is
/// inherited between fields — a deep merge would produce a provider spec that
/// appears in neither file, so "which `base_url` did my model actually get"
/// would stop being answerable by reading one. A user naming `openai:` is
/// describing their `openai`, whole.
///
/// ONE exception, and it is about the difference between overriding a value
/// and never mentioning it. `models` is the only optional map here, and a
/// later entry that OMITS it keeps the models it replaced. Without that, the
/// three lines it takes to point a shipped provider at an internal proxy
/// instead take a restatement of every model under it — and, worse, a file
/// that merely forgets the key does not misroute but fails to start, telling
/// its author to "delete it" when what they wanted was to keep it.
///
/// Writing the key is still a statement, and still replaces: `models:` with
/// entries means those and no others, and an explicit `models: {}` means none
/// at all, which the lint refuses in its own words. Only silence inherits.
///
/// Every entry is merged before any is built, so a provider that loses
/// outright only ever contributed its name to the map — its models never reach
/// the table and cannot outlive the entry that declared them.
///
/// Also hands back what it overrode. Reporting rather than logging here keeps
/// this a pure function, and makes "an override is announced" a plain
/// assertion rather than a test that has to capture a `tracing` subscriber.
pub(super) fn load_all(sources: &[Source]) -> Result<Fold, String> {
    let parsed: Vec<(&str, RegistryFile)> = sources
        .iter()
        .map(|source| {
            parse(source.yaml)
                .map(|file| (source.label, file))
                .map_err(|e| format!("{}: {e}", source.label))
        })
        .collect::<Result<_, _>>()?;
    fold(parsed)
}

/// The shipped roster, parsed once.
///
/// A client that supplies a roster of its own folds it over this one, and
/// re-reading 39 KB of CONSTANT YAML to do it cost about a millisecond per
/// client — a hundred and forty times what building a client without a roster
/// costs. Parsing is the expensive half and the answer never varies, so it is
/// done once and the entries are cloned into each fold. Cloning a few dozen
/// small structs is the cheap half.
///
/// Deliberately NOT the built [`Registry`](super::REGISTRY) next door: the fold
/// replaces entries as WRITTEN, before a provider name is interned or a model
/// key is checked against the provider holding it, so it is the schema shape
/// this needs and not the read surface.
static SHIPPED: LazyLock<RegistryFile> =
    LazyLock::new(|| parse(super::SHIPPED_YAML).unwrap_or_else(|e| panic!("specs.yaml: {e}")));

/// One roster folded over the shipped one, which is the only fold that ships.
///
/// Exists so the shipped half is not re-parsed per client; the fold itself is
/// [`load_all`]'s, and `one_source_folds_to_itself` plus
/// `folding_over_the_shipped_roster_matches_parsing_it` keep the two honest.
pub(super) fn load_over_shipped(label: &str, yaml: &str) -> Result<Fold, String> {
    let user = parse(yaml).map_err(|e| format!("{label}: {e}"))?;
    fold(vec![("specs.yaml", SHIPPED.clone()), (label, user)])
}

/// The fold itself, over already-parsed sources.
fn fold(sources: Vec<(&str, RegistryFile)>) -> Result<Fold, String> {
    // Sorted, so a file with two bad providers names the same one every run.
    // The build below is where a message comes from, and a `HashMap` there
    // would pick whichever came up first.
    let mut merged: BTreeMap<String, (&str, ProviderEntry)> = BTreeMap::new();
    let mut replaced = Vec::new();
    let mut retargeted = Vec::new();
    for (label, file) in sources {
        for (name, mut entry) in file.providers {
            if let Some((_, previous)) = merged.remove(&name) {
                // Silence inherits; a stated `models:` replaces. The label
                // moves to the winning source either way, so a malformed
                // inherited model would be blamed on the file that kept it —
                // reachable only if an EARLIER source held a bad model, which
                // for the one fold that ships is the roster CI already gates.
                if entry.models.is_none() {
                    entry.models = previous.models;
                    retargeted.push(name.clone());
                } else {
                    replaced.push(name.clone());
                }
            }
            merged.insert(name, (label, entry));
        }
    }

    let mut registry = Registry::default();
    for (name, (label, entry)) in merged {
        build_provider(&mut registry, &name, entry).map_err(|e| format!("{label}: {e}"))?;
    }
    replaced.sort_unstable();
    retargeted.sort_unstable();
    Ok(Fold {
        registry,
        replaced,
        retargeted,
    })
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
    // A written-but-valueless `models:` settles to an empty map here rather
    // than to "unstated": the fold above has already had its say about
    // inheritance, and what is left is a provider that lists no models, which
    // the lint refuses in its own words.
    for (key, model) in models.flatten().unwrap_or_default() {
        let spec = build(&key, name, model).map_err(|e| format!("`{key}`: {e}"))?;
        registry.models.insert(key, spec);
    }
    registry.providers.insert(
        name.to_string(),
        ProviderSpec {
            name: intern(name)?,
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
        (Some(var), None) => Ok(Credential::EnvVar(intern(&var)?)),
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
    /// An `auth:` written with NO VALUE is silence, not `auth: none`.
    ///
    /// YAML reads a bare `auth:` as null, and `Option<Auth>` reads null as
    /// absent — so a half-typed line lands on `Unavailable` rather than
    /// declaring the provider needs no credential. That is the safe direction
    /// and the whole reason the two are separate fields, but nothing pinned it
    /// until now: the alternative reading would turn a forgotten word into an
    /// unauthenticated request against a paid host.
    #[test]
    fn a_bare_auth_key_is_not_auth_none() {
        let registry = clean(&format!(
            "{}{}",
            without("env_api_key").replace("    models:\n", "    auth:\n    models:\n"),
            model("demo/plain")
        ));
        let provider = registry.providers.get("demo").unwrap();
        assert!(
            !provider.unauthenticated(),
            "a valueless `auth:` must read as silence, never as `auth: none`"
        );
        assert_eq!(provider.env_api_key(), None);
    }

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
        let fold = merged(&[&with(&model("demo/plain")), LOCAL_ROSTER]);
        assert_eq!(providers(&fold.registry), vec!["demo", "local"]);
        assert_eq!(
            keys(&fold.registry),
            vec!["demo/plain", "local/llama-3.3-70b"]
        );
        assert!(fold.replaced.is_empty() && fold.retargeted.is_empty());
    }

    /// A provider entry that STATES its models is replaced whole, models
    /// included. Nothing is merged field by field, so a half-merged entry
    /// would describe a provider that appears in neither file — and "which
    /// `base_url` did my model get" would stop being answerable by reading
    /// one.
    #[test]
    fn a_later_roster_replaces_a_provider_whole() {
        // `local` renamed to `demo` at the two places it is a KEY, leaving
        // `localhost` in the URL alone.
        let shadowing = LOCAL_ROSTER
            .replace("  local:", "  demo:")
            .replace("local/llama", "demo/llama");
        let fold = merged(&[&with(&models(&["demo/keep", "demo/plain"])), &shadowing]);
        assert_eq!(providers(&fold.registry), vec!["demo"]);
        // The earlier entry's models are GONE, not merged alongside.
        assert_eq!(keys(&fold.registry), vec!["demo/llama-3.3-70b"]);
        assert_eq!(
            fold.registry.providers.get("demo").unwrap().base_url(),
            "http://localhost:8000/v1"
        );
        assert_eq!(fold.replaced, vec!["demo"]);
        assert!(fold.retargeted.is_empty());
    }

    /// The one thing a later entry inherits: models it says NOTHING about.
    ///
    /// Pointing a shipped provider at an internal proxy is the case, and it is
    /// three lines rather than a restatement of every model under it. Before
    /// this, such a file did not misroute — it failed to start, telling its
    /// author to "delete it" when what they meant was to keep it.
    #[test]
    fn a_later_entry_that_names_no_models_keeps_the_ones_it_replaced() {
        let fold = merged(&[
            &with(&models(&["demo/keep", "demo/plain"])),
            "providers:\n  demo:\n    base_url: \"http://proxy.internal/v1\"\n    \
             env_api_key: DEMO_API_KEY\n",
        ]);
        // Both models still route — and they route to the NEW host.
        assert_eq!(keys(&fold.registry), vec!["demo/keep", "demo/plain"]);
        assert_eq!(
            fold.registry.providers.get("demo").unwrap().base_url(),
            "http://proxy.internal/v1"
        );
        // Announced as its own kind of override: this one kept the models,
        // and saying "replaced, models included" would be a lie.
        assert!(fold.replaced.is_empty());
        assert_eq!(fold.retargeted, vec!["demo"]);
    }

    /// A valueless `models:` is a half-written line, not a decision to keep
    /// models somebody never listed.
    ///
    /// The dangerous reading is the permissive one: `openai:` with a new
    /// `base_url` and a bare `models:` would silently send every shipped
    /// OpenAI model to that host. So a written key is always a statement, and
    /// this one states nothing — refused, exactly as `models: {}` is.
    #[test]
    fn a_valueless_models_key_inherits_nothing() {
        let fold = merged(&[
            &with(&models(&["demo/keep", "demo/plain"])),
            "providers:\n  demo:\n    base_url: \"http://proxy.internal/v1\"\n    \
             env_api_key: DEMO_API_KEY\n    models:\n",
        ]);
        assert!(
            keys(&fold.registry).is_empty(),
            "a bare `models:` must not carry the replaced models over"
        );
        assert_eq!(fold.replaced, vec!["demo"]);
        assert!(fold.retargeted.is_empty());
        let error = super::super::lint::usable(&fold.registry).unwrap_err();
        assert!(error.contains("registers no models"), "{error}");
    }

    /// Inheritance is for SILENCE only. Writing the key is a statement, and an
    /// explicitly empty map states none — which the lint refuses in its own
    /// words rather than this quietly handing the shipped models back.
    #[test]
    fn an_explicitly_empty_models_map_states_none_and_inherits_nothing() {
        let fold = merged(&[
            &with(&model("demo/plain")),
            "providers:\n  demo:\n    base_url: \"http://proxy.internal/v1\"\n    \
             env_api_key: DEMO_API_KEY\n    models: {}\n",
        ]);
        assert!(keys(&fold.registry).is_empty());
        assert_eq!(fold.replaced, vec!["demo"]);
        assert!(super::super::lint::usable(&fold.registry).is_err());
    }

    /// A provider the earlier source never had is an addition, not an
    /// override — so an entry with no models is exactly what it looks like,
    /// and there is nothing to inherit.
    #[test]
    fn a_new_provider_with_no_models_inherits_nothing() {
        let fold = merged(&[
            &with(&model("demo/plain")),
            "providers:\n  fresh:\n    base_url: \"http://fresh.internal/v1\"\n    \
             auth: none\n",
        ]);
        assert!(fold.replaced.is_empty());
        assert!(fold.retargeted.is_empty());
        // Nothing routes to it, and the lint says so rather than the fold.
        assert_eq!(keys(&fold.registry), vec!["demo/plain"]);
        let error = super::super::lint::usable(&fold.registry).unwrap_err();
        assert!(error.contains("fresh"), "{error}");
    }

    /// The replacement is reported so a client can say so out loud. Silently
    /// shadowing a provider somebody thought they were still using is the one
    /// outcome nobody could debug.
    #[test]
    fn every_replacement_is_reported_once() {
        let fold = merged(&[
            &format!("{}{OTHER_PROVIDER}", with(&model("demo/plain"))),
            &format!(
                "{}{}",
                with(&model("demo/replaced")),
                OTHER_PROVIDER.replace("api.other.test", "api.replaced.test")
            ),
        ]);
        assert_eq!(fold.replaced, vec!["demo", "other"]);
    }

    /// A message names the file it came from, which is the whole reason a
    /// source carries a label: a stranger with a bad roster must not be sent
    /// to read warpllm's own.
    #[test]
    fn a_failure_names_the_source_it_came_from() {
        let err = merge(&[&with(&model("demo/plain")), "providers: [not, a, map]\n"])
            .err()
            .expect("a sequence where a provider map belongs cannot load");
        assert!(err.starts_with("second.yaml:"), "{err}");
    }

    /// The cached shipped half and a freshly parsed one fold to the same
    /// thing, so the shortcut `load_for_client` takes cannot drift from the
    /// general fold it stands in for.
    ///
    /// Worth pinning because the shortcut exists purely for speed: nothing
    /// about the result is supposed to differ, and a divergence would show up
    /// as a roster behaving one way in a client and another in every test.
    #[test]
    fn folding_over_the_shipped_roster_matches_parsing_it() {
        let user = "providers:\n  local:\n    base_url: \"http://127.0.0.1:1/v1\"\n    \
                    auth: none\n    models:\n      local/m:\n        supported_apis:\n          \
                    - {api: openai_compat_chat_completions}\n";
        let cached = load_over_shipped("warpllm.yaml", user).unwrap();
        let fresh = load_all(&[
            Source {
                label: "specs.yaml",
                yaml: super::super::SHIPPED_YAML,
            },
            Source {
                label: "warpllm.yaml",
                yaml: user,
            },
        ])
        .unwrap();
        assert_eq!(keys(&cached.registry), keys(&fresh.registry));
        assert_eq!(providers(&cached.registry), providers(&fresh.registry));
        assert_eq!(cached.replaced, fresh.replaced);
        assert_eq!(cached.retargeted, fresh.retargeted);
    }

    /// The shortcut carries the retarget rule too — it is the same fold, and
    /// the case a client is most likely to hit.
    #[test]
    fn the_shipped_shortcut_retargets_like_any_other_fold() {
        let fold = load_over_shipped(
            "warpllm.yaml",
            "providers:\n  openai:\n    base_url: \"http://proxy.internal/v1\"\n    \
             env_api_key: OPENAI_API_KEY\n",
        )
        .unwrap();
        assert_eq!(fold.retargeted, vec!["openai"]);
        assert_eq!(
            fold.registry.providers.get("openai").unwrap().base_url(),
            "http://proxy.internal/v1"
        );
        assert!(fold.registry.models.contains_key("openai/gpt-5.6"));
    }

    /// A bad user file is still blamed on the user file, not on the cached
    /// half it was folded over.
    #[test]
    fn the_shipped_shortcut_still_names_the_users_file() {
        let err = load_over_shipped("warpllm.yaml", "providers: [not, a, map]\n")
            .err()
            .expect("a sequence where a provider map belongs cannot load");
        assert!(err.starts_with("warpllm.yaml:"), "{err}");
    }

    /// One roster is the same fold with one source, so `load` and `load_all`
    /// cannot drift apart.
    #[test]
    fn one_source_folds_to_itself() {
        let yaml = with(&model("demo/plain"));
        assert_eq!(
            keys(&merged(&[&yaml]).registry),
            keys(&load(&yaml).unwrap())
        );
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
