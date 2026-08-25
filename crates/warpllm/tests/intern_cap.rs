//! Filling the provider-name interner must not take the shipped roster with it.
//!
//! Its own test binary, and it has to be: the interner is process-wide and this
//! case deliberately exhausts it, so every roster loaded afterwards in the same
//! process can only name providers already interned. Nothing else may share
//! that process.
//!
//! The hazard is not the cap itself, it is the interaction. A provider name
//! lives as long as the process, so the cap exists to keep that bounded. But
//! the shipped registry is built by an infallible `LazyLock` that panics if it
//! cannot be built — so if custom rosters could fill the cap before the shipped
//! identities were interned, the first client built WITHOUT a roster would
//! panic inside that initializer and poison it for good.

use std::fmt::Write as _;

use tempfile::TempDir;
use warpllm::{Client, ClientConfig};

/// A roster naming `count` providers nobody has ever seen, each with one model.
///
/// Named to sort BEFORE every shipped provider, deliberately. Entries are
/// built in sorted order, so names that sorted after the shipped ones would
/// see them interned first and this would pass whether or not anything
/// guaranteed it.
///
/// `auth: none` throughout, so the only strings this can intern are the
/// provider names themselves. The cap counts credential variables in the same
/// table, so an `env_api_key` per provider would fill it twice as fast and
/// leave it unclear which kind of name tripped it.
fn many_providers(count: usize) -> String {
    let mut yaml = String::from("providers:\n");
    for i in 0..count {
        write!(
            yaml,
            "  aaa-{i}:\n    base_url: \"http://127.0.0.1:1/v1\"\n    auth: none\n    \
             models:\n      aaa-{i}/m:\n        supported_apis:\n          \
             - {{api: openai_compat_chat_completions}}\n"
        )
        .unwrap();
    }
    yaml
}

fn client_for(dir: &TempDir, yaml: &str) -> warpllm::Result<Client> {
    let path = dir.path().join("warpllm.yaml");
    std::fs::write(&path, yaml).unwrap();
    Client::new(ClientConfig {
        specs_path: Some(path),
        ..Default::default()
    })
}

#[test]
fn exhausting_the_interner_leaves_the_shipped_roster_usable() {
    let dir = TempDir::new().unwrap();

    // Well past the cap, in one file rather than a loop: the cap counts
    // distinct names for the life of the process, so how they arrive is not
    // what is under test.
    let error = client_for(&dir, &many_providers(5000))
        .err()
        .expect("a roster naming 5000 unseen providers must exhaust the cap")
        .to_string();
    assert!(error.contains("distinct roster names"), "{error}");
    assert!(error.contains("aaa-"), "{error}");

    // THE claim. The interner is full, and a client that supplies no roster is
    // exactly the one that needs the shipped names interned. It must build,
    // and route, rather than panicking inside a `LazyLock` that then stays
    // poisoned for every caller after it.
    let client = Client::new(ClientConfig::default())
        .expect("the shipped roster is interned before any stranger's file");
    let (provider, model) = client.fetch_model("openai/gpt-5.6").unwrap();
    assert_eq!(provider.name(), "openai");
    assert_eq!(provider.env_api_key(), Some("OPENAI_API_KEY"));
    assert_eq!(model.model(), "gpt-5.6");

    // And a roster naming only providers already interned still loads, so a
    // full interner is a bound on NEW names rather than a broken process.
    let retarget = "providers:\n  openai:\n    base_url: \"http://proxy.internal/v1\"\n    \
                    env_api_key: OPENAI_API_KEY\n";
    let client = client_for(&dir, retarget).expect("no new name, so nothing to intern");
    assert_eq!(
        client.fetch_model("openai/gpt-5.6").unwrap().0.base_url(),
        "http://proxy.internal/v1"
    );
}
