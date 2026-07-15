//! Every fixture in this directory's fixtures/ must survive a deserialize →
//! reserialize round trip byte-for-byte (as JSON values). Any field the
//! upstream API sends that we drop, rename, or mistype fails the diff.
//!
//! `doc-full-response.json` is hand-built from the API reference.

use warpllm::ChatCompletion;

#[test]
fn fixtures_round_trip_losslessly() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/types/openai/chat/completions/fixtures"
    );
    let mut checked = 0;
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let raw = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let completion: ChatCompletion = serde_json::from_value(value.clone())
            .unwrap_or_else(|e| panic!("{} failed to deserialize: {e}", path.display()));
        assert_eq!(
            serde_json::to_value(&completion).unwrap(),
            value,
            "lossy round trip for {}",
            path.display()
        );
        checked += 1;
    }
    assert!(checked > 0, "no fixtures found in {dir}");
}
