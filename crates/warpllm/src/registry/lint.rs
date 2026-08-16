//! Roster hygiene: everything true of a GOOD roster that loading it does not
//! need in order to succeed.
//!
//! Two audiences now, and they are not held to the same rules. warpllm reads
//! its own `specs.yaml` and, when a client is pointed at one, a roster written
//! by somebody who has never seen this file. The split is between what makes a
//! roster USABLE and what makes the shipped one TIDY:
//!
//! [`usable`] is compiled in and runs against every roster a client loads. Each
//! of its checks catches something that produces perfectly correct tables and
//! then cannot serve a request — a `base_url` an endpoint cannot be appended
//! to, a model listing no surface, a provider nothing routes to. Finding those
//! at construction is the whole value: the alternative is a gateway that starts
//! clean and fails closed on the first request, hours later, somewhere else.
//! Worse, in the one case [`appendable`] describes, a gateway that starts clean
//! and then sends a request to the wrong path and is answered.
//!
//! [`tidy`] stays `cfg(test)`, and holds the one rule that is convention rather
//! than correctness: both maps in ascending key order. That exists so a
//! contributor adding a provider has exactly one place to put it and two PRs do
//! not collide — which is a fact about this repository and nothing at all about
//! a stranger's three-line file. Lecturing them about `LC_ALL=C sort` in their
//! own config would be rude and would block a roster that works.
//!
//! Between the two sits [`shared_env_api_keys`], which is compiled in and
//! returns findings rather than an error. Two providers sharing one variable is
//! the silent cross-authentication it was written to catch — and it is also
//! exactly what somebody fronting OpenAI with their own proxy means to do. The
//! shipped roster is still refused for it, by [`check`]; a user's is warned.
//!
//! [`check`] is what CI runs over `specs.yaml`: load, then tidy, then usable,
//! then the strict reading of the variable collision.

use std::collections::{BTreeMap, HashMap, HashSet};

/// Only the `cfg(test)` half loads a roster from text; [`usable`] is handed
/// tables that have already been folded together.
#[cfg(test)]
use super::load::load;
use super::types::{ModelSpec, ProviderSpec, Registry};

/// Everything a roster has to satisfy before a request can be routed against
/// it, first failure reported.
///
/// Takes the resolved tables rather than the text, because by the time this
/// runs there may be no single text: a client's roster is its file folded over
/// the shipped one, and a complaint about the result belongs to neither file
/// alone. [`tidy`], which reads line numbers out of the source, is the half
/// that must stay on text — and is exactly the half a user is not held to.
pub(super) fn usable(registry: &Registry) -> Result<(), String> {
    if registry.providers.is_empty() {
        return Err("providers: the registry names no providers".into());
    }
    serves_models(registry)?;
    // `HashMap`s, so iterating them directly would make a message name
    // whichever member of a collision came up first. Both loops below read a
    // sorted view instead, so the error a contributor sees is the one CI saw.
    for (name, provider) in by_name(registry) {
        routable(provider).map_err(|e| format!("`{name}`: {e}"))?;
    }
    for (key, spec) in registry.models.iter().collect::<BTreeMap<_, _>>() {
        if spec.model().is_empty() {
            return Err(format!(
                "`{key}`: model is empty; omit the field to ship the key's own \
                 last segment"
            ));
        }
        serves(spec).map_err(|e| format!("`{key}`: {e}"))?;
    }
    Ok(())
}

/// The rules the SHIPPED roster is held to and nobody else is: both maps in
/// ascending key order.
#[cfg(test)]
pub(super) fn tidy(yaml: &str) -> Result<(), String> {
    // Only `Value` keeps the order keys appear in the file; the typed pass has
    // hashed them by the time it returns, and the tables keep no order either.
    let value: yaml_serde::Value = yaml_serde::from_str(yaml).map_err(|e| e.to_string())?;
    sorted(yaml, &value)
}

/// Every policy, in the order a reader would want them, first failure
/// reported. The gate CI runs over `specs.yaml`.
#[cfg(test)]
pub(super) fn check(yaml: &str) -> Result<(), String> {
    // Structure first: a real syntax or key error should be reported as
    // itself, not as whatever hygiene complaint it happens to also trip.
    let registry = load(yaml)?;
    tidy(yaml)?;
    usable(&registry)?;
    match shared_env_api_keys(&registry).first() {
        Some(collision) => Err(format!(
            "{collision}; every provider needs its own environment variable"
        )),
        None => Ok(()),
    }
}

fn by_name(registry: &Registry) -> BTreeMap<&str, &ProviderSpec> {
    registry
        .providers
        .iter()
        .map(|(name, provider)| (name.as_str(), provider))
        .collect()
}

/// Both maps are kept in ascending key order: providers alphabetically, and
/// each provider's models alphabetically within it. A contributor adding an
/// entry then has exactly one place to put it, and two PRs adding different
/// providers do not collide.
///
/// Reads the raw text as well as the parsed value, only to put line numbers in
/// the message. The whole point of this check is that a contributor can fix it
/// without having to work out what "unsorted" means, so it names the key to
/// move, where to move it, and the ordering it is being judged against.
#[cfg(test)]
fn sorted(yaml: &str, value: &yaml_serde::Value) -> Result<(), String> {
    // A missing or malformed `providers:` is the typed pass's error to report.
    let Some(providers) = value.get("providers").and_then(|v| v.as_mapping()) else {
        return Ok(());
    };
    ascending("providers", yaml, providers.keys())?;
    for (name, entry) in providers {
        let (Some(name), Some(models)) = (name.as_str(), entry.get("models")) else {
            continue;
        };
        let Some(models) = models.as_mapping() else {
            continue;
        };
        ascending(&format!("`{name}`'s models"), yaml, models.keys())?;
    }
    Ok(())
}

/// One map's keys, in the order the file writes them.
#[cfg(test)]
fn ascending<'a>(
    what: &str,
    yaml: &str,
    keys: impl Iterator<Item = &'a yaml_serde::Value>,
) -> Result<(), String> {
    let keys: Vec<&str> = keys.filter_map(yaml_serde::Value::as_str).collect();
    for pair in keys.windows(2) {
        // Equal keys never reach this: `load`'s `Value` pass rejects a
        // duplicate before the order of two of them could be in question.
        let (above, below) = (pair[0], pair[1]);
        if above <= below {
            continue;
        }
        return Err(format!(
            "{what}: `{below}`{} is out of order — move it above `{above}`{}.\n\
             Entries are kept in ascending BYTE order, which is what \
             `LC_ALL=C sort` gives and may differ from your editor's sort.",
            line_of(yaml, below),
            line_of(yaml, above),
        ));
    }
    Ok(())
}

/// ` (line N)` for a roster key, or nothing when the text does not show it
/// plainly.
///
/// A scan rather than a span because `yaml_serde::Value` carries no positions.
/// Requiring the `:` immediately after the key is what keeps a search for
/// `openai/gpt-5.6` off the `openai/gpt-5.6-luna` line. It only ever decorates
/// a message, so a miss costs nothing but the number.
#[cfg(test)]
fn line_of(yaml: &str, key: &str) -> String {
    yaml.lines()
        .position(|line| {
            let line = line.trim_start();
            line.strip_prefix(key)
                .or_else(|| line.strip_prefix(&format!("{key:?}")))
                .is_some_and(|rest| rest.starts_with(':'))
        })
        .map_or_else(String::new, |i| format!(" (line {})", i + 1))
}

/// A provider nothing routes to holds a transport no caller can ever reach.
fn serves_models(registry: &Registry) -> Result<(), String> {
    let serving: HashSet<&str> = registry
        .models
        .values()
        .map(|spec| spec.provider.as_str())
        .collect();
    for name in by_name(registry).into_keys() {
        if !serving.contains(name) {
            return Err(format!(
                "`{name}`: this provider registers no models, so nothing can route \
                 to it. Give it a model entry, or delete it."
            ));
        }
    }
    Ok(())
}

/// Whether a provider could actually serve a request. Every one of these loads
/// fine and then misbehaves at runtime, which is exactly why they are worth a
/// test.
fn routable(provider: &ProviderSpec) -> Result<(), String> {
    if provider.base_url().is_empty() {
        return Err("base_url is empty".into());
    }
    if provider.base_url().ends_with('/') {
        return Err(format!(
            "base_url `{}` ends with `/`; endpoints append their own path, which \
             would produce a doubled slash",
            provider.base_url()
        ));
    }
    appendable(provider.base_url())?;
    // Optional — a provider may name no variable at all — but an empty string
    // is never what anyone meant by that. Say so rather than resolve it to a
    // lookup of `""` at request time.
    if provider.env_api_key().is_some_and(str::is_empty) {
        return Err(
            "env_api_key is empty; omit the field entirely if this provider has no \
             key variable yet"
                .into(),
        );
    }
    Ok(())
}

/// Whether an endpoint can actually be APPENDED to this `base_url`.
///
/// warpllm builds a URL by concatenation — `{base_url}/chat/completions` — so
/// the base has to be a string that survives having a path stuck on the end.
/// Three ways it does not, and none of them is visible by reading the line:
///
/// - No scheme. `localhost:8000/v1` is the address a server prints in its own
///   startup log, and it is the most likely thing to be pasted here. It PARSES
///   — as scheme `localhost` with path `8000/v1` — so a parse alone would let
///   it through, and reqwest then refuses to build the request.
/// - A query string. `http://host/v1?token=x` becomes
///   `http://host/v1?token=x/chat/completions`, which reaches the host with the
///   endpoint buried in the query and no path at all. The request is sent and
///   answered — wrongly — which makes this the worst of the three.
/// - A fragment, for the same reason one step further along.
///
/// Parsed with reqwest's own `Url`, deliberately rather than with a hand-rolled
/// prefix check or a second URL crate: the question being asked is whether the
/// thing that will build the request can build it, and only its parser answers
/// that. It is already in the dependency tree for exactly that reason.
///
/// This matters more than it used to. The shipped roster's four `base_url`s
/// were written once and reviewed; a roster file's is typed by somebody
/// copying an address out of their own server's logs, and the whole point of
/// checking at load is that they hear about it while they are still looking at
/// the file.
fn appendable(base_url: &str) -> Result<(), String> {
    let url = reqwest::Url::parse(base_url).map_err(|e| {
        format!(
            "base_url `{base_url}` is not a URL: {e}. Write it whole, scheme \
             included — `http://localhost:8000/v1`"
        )
    })?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(format!(
            "base_url `{base_url}` has scheme `{}`, and warpllm speaks HTTP. A \
             host and port with no scheme reads as one of these rather than \
             failing, so write `http://{base_url}` if that is what was meant",
            url.scheme()
        ));
    }
    if let Some(tail) = url
        .query()
        .map(|q| format!("query string `?{q}`"))
        .or_else(|| url.fragment().map(|f| format!("fragment `#{f}`")))
    {
        return Err(format!(
            "base_url `{base_url}` carries a {tail}; an endpoint is appended to \
             this string, so `/chat/completions` would land inside it rather \
             than in the path. A credential belongs in `env_api_key`, not here"
        ));
    }
    Ok(())
}

/// A model's `supported_apis`: non-empty, and without repeats.
///
/// The list is the model's alone — nothing is inherited — so both of these are
/// things only the roster's own text can get wrong.
///
/// There is deliberately no cross-check against the provider. A surface names
/// the wire format it is spoken in, and the provider records none, so there is
/// nothing left for the two to disagree about — which is the point: one host
/// may serve one model over one protocol and its neighbour over another, and
/// no roster rule has to be taught the exception.
fn serves(spec: &ModelSpec) -> Result<(), String> {
    if spec.supported_apis().is_empty() {
        return Err(
            "supported_apis is empty, so nothing could ever route to this model; \
             name a surface it serves, or delete the entry"
                .into(),
        );
    }
    // On the surface, not the whole entry: two entries naming one surface are
    // one mistake whether or not everything else about them matches, and only
    // the first would ever be found by `supported_api`.
    let mut seen = HashSet::new();
    for entry in spec.supported_apis() {
        let api = entry.api();
        if !seen.insert(api) {
            return Err(format!("`{api:?}` is listed twice"));
        }
    }
    Ok(())
}

/// Every environment variable claimed by more than one provider, described.
///
/// Two providers sharing a variable would authenticate one with the other's
/// credentials, and between two third-party providers that is always an
/// accident. Between `openai` and somebody's own proxy in front of it, it is
/// the point — the proxy forwards the same key, and demanding a second variable
/// holding the same secret buys nothing. So this REPORTS rather than refuses,
/// and the two callers read it differently: [`check`] treats a finding as a
/// failure of the shipped roster, and a client logs it as a warning against a
/// user's.
///
/// In `by_name` order, so a roster with two collisions names the same one every
/// run.
pub(super) fn shared_env_api_keys(registry: &Registry) -> Vec<String> {
    // A lookup, never iterated: the ordering that makes this deterministic is
    // `by_name`'s, one line below.
    let mut owner: HashMap<&str, &str> = HashMap::new();
    let mut collisions = Vec::new();
    for (name, provider) in by_name(registry) {
        // Nothing to collide over when a provider names no variable.
        let Some(env_api_key) = provider.env_api_key() else {
            continue;
        };
        if let Some(other) = owner.insert(env_api_key, name) {
            collisions.push(format!(
                "`{env_api_key}` is claimed by both `{other}` and `{name}`"
            ));
        }
    }
    collisions
}

/// Every case here LOADS fine — the tables it produces are correct. What is
/// wrong with it is the roster, which is why this is the gate and `load` is
/// not.
#[cfg(test)]
mod tests {
    use super::super::testing::{CHAT, OTHER_PROVIDER, PROVIDER, clean, model, models, with};
    use super::*;

    /// One provider's roster, plus a second one that shares its variable.
    fn two_providers(env_api_key: &str) -> String {
        format!(
            "{}{}",
            with(&model("demo/plain")),
            OTHER_PROVIDER.replace("OTHER_API_KEY", env_api_key)
        )
    }

    /// Two providers sharing an environment variable would silently
    /// authenticate one with the other's credentials.
    #[test]
    fn colliding_env_api_keys_are_rejected() {
        let yaml = two_providers("DEMO_API_KEY");
        assert!(
            load(&yaml).is_ok(),
            "the roster is what is wrong, not the tables"
        );
        let err = check(&yaml).unwrap_err();
        assert!(err.contains("DEMO_API_KEY"), "{err}");
        assert!(err.contains("its own environment variable"), "{err}");
    }

    /// Two providers with their own variables are the normal case.
    #[test]
    fn distinct_env_api_keys_are_accepted() {
        clean(&two_providers("OTHER_API_KEY"));
    }

    /// Naming no variable is not a collision, so two providers may both do it.
    /// The check has to skip absent keys rather than treat `None` as a shared
    /// owner, which would reject a perfectly good roster.
    #[test]
    fn two_providers_may_both_name_no_env_api_key() {
        let drop_keys = |yaml: &str| {
            yaml.lines()
                .filter(|line| !line.trim_start().starts_with("env_api_key:"))
                .fold(String::new(), |acc, line| acc + line + "\n")
        };
        clean(&drop_keys(&two_providers("OTHER_API_KEY")));
    }

    /// An empty string is not the same as omitting the field, and resolving it
    /// would send a request-time lookup of `""`.
    #[test]
    fn an_empty_env_api_key_points_at_omitting_it() {
        let yaml = with(&model("demo/plain")).replace("DEMO_API_KEY", "\"\"");
        let err = check(&yaml).unwrap_err();
        assert!(err.contains("env_api_key is empty"), "{err}");
        assert!(err.contains("omit the field entirely"), "{err}");
    }

    /// A misspelled API is a typo, and a closed vocabulary catches it before
    /// the roster even loads — so this one does not reach the lint at all, let
    /// alone a live provider. Held here anyway: what matters is that the typo
    /// is rejected, not which gate does it.
    ///
    /// The message names the whole vocabulary, which is what lets a
    /// contributor fix the line without opening `protocol/types.rs`.
    #[test]
    fn an_api_outside_the_vocabulary_fails_to_load() {
        let yaml = with(&model("demo/plain")).replace(
            "api: openai_compat_chat_completions",
            "api: anthropic_messages",
        );
        let err = load(&yaml).unwrap_err();
        assert!(
            err.contains("unknown variant `anthropic_messages`"),
            "{err}"
        );
        for known in [
            "openai_compat_chat_completions",
            "openai_compat_chat_completions_stream",
            "openai_compat_responses",
        ] {
            assert!(err.contains(known), "vocabulary missing {known}: {err}");
        }
        // `check` loads first, so it reports the same thing rather than some
        // downstream hygiene complaint the typo happens to also trip.
        assert!(
            check(&yaml).unwrap_err().contains("unknown variant"),
            "{err}"
        );
    }

    /// The split this module exists for, stated as a test: a roster that is
    /// merely untidy is perfectly USABLE, and a client loading a stranger's
    /// file must accept it.
    ///
    /// Asserted rather than assumed, because the failure mode is silent in the
    /// worst direction — folding the order check back into `usable` would
    /// refuse a working roster over a convention that means nothing outside
    /// this repository, and every test above would still pass.
    #[test]
    fn an_untidy_roster_is_still_usable() {
        let yaml = with(&models(&["demo/zeta", "demo/alpha"]));
        let registry = load(&yaml).unwrap();
        assert!(tidy(&yaml).is_err(), "the fixture must be untidy");
        usable(&registry).expect("untidiness is not warpllm's business to refuse");
    }

    /// The collision the shipped roster is refused for is REPORTED for anyone
    /// else's, because two providers sharing a variable is somebody fronting
    /// one with a proxy at least as often as it is a mistake.
    #[test]
    fn a_shared_env_api_key_is_reported_rather_than_refused() {
        let registry = load(&two_providers("DEMO_API_KEY")).unwrap();
        usable(&registry).expect("a shared variable still routes");
        let shared = shared_env_api_keys(&registry);
        assert_eq!(shared.len(), 1, "{shared:?}");
        assert!(shared[0].contains("DEMO_API_KEY"), "{shared:?}");
        assert!(shared[0].contains("`demo`"), "{shared:?}");
        assert!(shared[0].contains("`other`"), "{shared:?}");
    }

    /// Both maps are kept sorted so a new entry has one place to go and two
    /// PRs adding different providers do not collide. Purely a convention,
    /// which is exactly why it must not stop the tables being built.
    #[test]
    fn out_of_order_models_are_rejected() {
        let yaml = with(&models(&["demo/zeta", "demo/alpha"]));
        assert!(load(&yaml).is_ok(), "key order cannot break loading");
        let err = check(&yaml).unwrap_err();
        // Names the map, the key to move, where to move it, and the ordering
        // it is judged against, so nobody has to work out what "unsorted"
        // meant.
        assert!(err.contains("`demo`'s models"), "{err}");
        assert!(err.contains("`demo/alpha` (line 9)"), "{err}");
        assert!(err.contains("move it above `demo/zeta` (line 6)"), "{err}");
        assert!(err.contains("ascending BYTE order"), "{err}");
    }

    /// Every provider's models are checked, not just the first one's. The
    /// check walks a map of maps, so a version of it that stopped after one
    /// provider would pass every other case here and quietly let the rest of
    /// the roster drift out of order.
    #[test]
    fn out_of_order_models_are_rejected_under_any_provider() {
        // `demo` above is in order; the misplaced key is in the LAST provider.
        let yaml = format!(
            "{}{OTHER_PROVIDER}{}",
            with(&model("demo/plain")),
            model("other/alpha")
        );
        assert!(load(&yaml).is_ok(), "key order cannot break loading");
        let err = check(&yaml).unwrap_err();
        assert!(err.contains("`other`'s models"), "{err}");
        assert!(err.contains("move it above `other/plain`"), "{err}");
    }

    /// The provider map is held to the same rule, and says which map it is
    /// talking about.
    #[test]
    fn out_of_order_providers_are_rejected() {
        let demo = with(&model("demo/plain"));
        let demo = demo
            .strip_prefix("providers:\n")
            .expect("the fixture's first line");
        let yaml = format!("providers:\n{OTHER_PROVIDER}{demo}");
        assert!(load(&yaml).is_ok(), "key order cannot break loading");
        let err = check(&yaml).unwrap_err();
        assert!(err.contains("providers: `demo`"), "{err}");
        assert!(err.contains("move it above `other`"), "{err}");
    }

    /// The line number points at the misplaced key's OWN line even when a
    /// neighbour starts with the same bytes — the case a naive substring
    /// search gets wrong.
    #[test]
    fn the_order_error_points_at_the_right_line() {
        let err = check(&with(&models(&["demo/model-x", "demo/model"]))).unwrap_err();
        assert!(err.contains("`demo/model` (line 9)"), "{err}");
        assert!(err.contains("above `demo/model-x` (line 6)"), "{err}");
    }

    /// Both ways of writing "no models yet" load fine and are caught here,
    /// because a provider nothing routes to is dead weight in the roster.
    #[test]
    fn a_provider_with_no_models_is_rejected() {
        for yaml in [PROVIDER, &PROVIDER.replace("    models:\n", "")] {
            assert!(load(yaml).is_ok(), "the roster is what is wrong: {yaml}");
            let err = check(yaml).unwrap_err();
            assert!(err.contains("registers no models"), "{err}");
        }
    }

    /// Each of these loads into a spec that is well-formed and then cannot
    /// serve a request: a URL that would double its slash, or a roster with
    /// nothing in it at all.
    #[test]
    fn unroutable_providers_are_rejected() {
        let trailing = with(&model("demo/plain")).replace("api.demo.test/v1", "api.demo.test/v1/");
        let err = check(&trailing).unwrap_err();
        assert!(err.contains("doubled slash"), "{err}");

        let err = check("providers: {}\n").unwrap_err();
        assert!(err.contains("names no providers"), "{err}");
    }

    /// A `base_url` an endpoint cannot be appended to. Every one of these
    /// loads, builds a client, and fails on the FIRST REQUEST — which is
    /// precisely the failure `usable` exists to move to construction, and the
    /// field a stranger writing their own roster is most likely to get wrong.
    ///
    /// The three are separated because they break differently. The first two
    /// never reach the network; the third does, and is answered — with the
    /// endpoint swallowed into the query, so the host sees a request for `/v1`
    /// and warpllm reports whatever it makes of that. A wrong answer is worse
    /// than no answer, which is why a query string is a refusal and not a
    /// warning.
    #[test]
    fn a_base_url_an_endpoint_cannot_be_appended_to_is_rejected() {
        let roster = |base_url: &str| {
            with(&model("demo/plain")).replace("https://api.demo.test/v1", base_url)
        };
        for (base_url, expected) in [
            // Parses as scheme `localhost`, which is why a bare parse is not
            // enough and the scheme is checked by name.
            ("localhost:8000/v1", "speaks HTTP"),
            ("not a url at all", "is not a URL"),
            ("http://api.demo.test/v1?token=x", "query string `?token=x`"),
            ("http://api.demo.test/v1#frag", "fragment `#frag`"),
        ] {
            let yaml = roster(base_url);
            assert!(
                load(&yaml).is_ok(),
                "`{base_url}`: the roster is what is wrong, not the tables"
            );
            let err = check(&yaml).unwrap_err();
            assert!(err.contains(expected), "`{base_url}`: {err}");
        }
    }

    /// The other side of it: an ordinary address, with and without a path,
    /// passes. A check that refused everything would satisfy the case above.
    #[test]
    fn an_ordinary_base_url_is_accepted() {
        for base_url in [
            "http://localhost:8000/v1",
            "https://api.demo.test",
            "https://vllm.internal.example.com:8443/v1",
        ] {
            let yaml = with(&model("demo/plain")).replace("https://api.demo.test/v1", base_url);
            clean(&yaml);
        }
    }

    /// Two models of one provider may serve entirely different surfaces, and
    /// the lint has no opinion about it. This used to be checked against the
    /// provider's own `protocol:`; with that gone there is nothing to agree
    /// with, which is what lets one host serve a chat model beside a
    /// Responses-only one.
    #[test]
    fn one_provider_may_serve_models_with_disjoint_surfaces() {
        clean(&with(concat!(
            "      demo/chat-only:\n",
            "        supported_apis:\n",
            "          - {api: openai_compat_chat_completions}\n",
            "      demo/responses-only:\n",
            "        supported_apis:\n",
            "          - {api: openai_compat_responses}\n",
        )));
    }

    /// The same surface twice is one mistake, however its settings are
    /// written. Caught on the surface's identity, not on the payload, which
    /// today is `{}` in both copies and would compare equal either way.
    #[test]
    fn a_repeated_surface_is_rejected() {
        let yaml = with(concat!(
            "      demo/plain:\n",
            "        supported_apis:\n",
            "          - {api: openai_compat_chat_completions}\n",
            "          - {api: openai_compat_chat_completions}\n",
        ));
        assert!(
            load(&yaml).is_ok(),
            "the roster is what is wrong, not the tables"
        );
        let err = check(&yaml).unwrap_err();
        assert!(err.contains("`demo/plain`"), "{err}");
        assert!(err.contains("listed twice"), "{err}");
    }

    /// An empty list is a model nothing can route to. It loads — the tables
    /// are fine — and the message says what to do, since an entry that serves
    /// nothing is either missing a line or should not be there at all.
    #[test]
    fn a_model_serving_no_surface_is_rejected() {
        let yaml = with("      demo/plain:\n        supported_apis: []\n");
        assert!(load(&yaml).is_ok(), "the roster is what is wrong");
        let err = check(&yaml).unwrap_err();
        assert!(err.contains("`demo/plain`"), "{err}");
        assert!(err.contains("supported_apis is empty"), "{err}");
        assert!(err.contains("delete the entry"), "{err}");
    }

    /// An empty `model` would ship `"model": ""` upstream, which no provider
    /// serves. Omitting the field is how an entry says "use my key's name".
    #[test]
    fn an_empty_model_points_at_omitting_it() {
        let err = check(&with(&format!(
            "      demo/plain:\n{CHAT}        model: \"\"\n"
        )))
        .unwrap_err();
        assert!(err.contains("model is empty"), "{err}");
        assert!(err.contains("omit the field"), "{err}");
    }
}
