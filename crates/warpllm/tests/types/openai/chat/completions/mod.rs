//! Every fixture in this directory's fixtures/ must survive a deserialize →
//! reserialize round trip byte-for-byte (as JSON values). Any field the
//! upstream API sends that we drop, rename, or mistype fails the diff.
//!
//! `doc-full-response.json` is hand-built from the API reference.

use warpllm::CreateChatCompletionResponse;

fn assert_fixtures_round_trip<T>(dir: &str)
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

#[test]
fn fixtures_round_trip_losslessly() {
    assert_fixtures_round_trip::<CreateChatCompletionResponse>(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/types/openai/chat/completions/fixtures"
    ));
}
