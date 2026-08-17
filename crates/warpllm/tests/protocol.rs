//! Protocol wire shapes, checked against the public surface.
//!
//! One module per protocol, each `#[path]`-remapped so the directory tree
//! mirrors `src/protocol/`. Every fixture in a module's `fixtures/` must
//! survive a deserialize → reserialize round trip: any field the upstream API
//! sends that warpllm drops, renames, or mistypes fails the diff.

#[path = "protocol/anthropic/messages/mod.rs"]
mod anthropic_messages;
#[path = "protocol/openai_compat/chat_completions/mod.rs"]
mod openai_compat_chat_completions;

/// Asserts every JSON fixture in `dir` round trips through `T` unchanged.
///
/// Shared by every protocol: what differs between them is the fixtures and the
/// types, never the property. Deserializing to `T` and comparing the
/// reserialized value against the original is the whole check, and it is the
/// one that catches a dropped field without anyone having to think of it.
pub fn assert_fixtures_round_trip<T>(dir: &str)
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let mut checked = 0;
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let raw = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let parsed: T = serde_json::from_value(value.clone())
            .unwrap_or_else(|e| panic!("{} failed to deserialize: {e}", path.display()));
        assert_eq!(
            serde_json::to_value(&parsed).unwrap(),
            value,
            "lossy round trip for {}",
            path.display()
        );
        checked += 1;
    }
    assert!(checked > 0, "no fixtures found in {dir}");
}
