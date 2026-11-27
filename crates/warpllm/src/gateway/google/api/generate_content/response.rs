//! Response conversion: Google's `GenerateContentResponse` → the gateway form.
//!
//! INGEST ONLY. Anthropic keeps a `render_response` so tests can prove its
//! residue round-trips; here there is nothing to restore, so there is no
//! render half.
//!
//! # Decisions, and where they were made (#35)
//!
//! * **The finish reason is Google's, not ours.** `finish_reason_raw` is the
//!   string Google sent and `finish_reason` is a plain table over it. A reply
//!   that is a function call therefore carries `Stop`, as Google reports it.
//! * **Usage sums to Google's total.** Tool-use prompt tokens fold into the
//!   input and thinking tokens into the output, so input + output equals
//!   `totalTokenCount`. The cached count is a subset of the prompt already.
//! * **Several candidates are several completions**, in `index` order.
//! * **A blocked prompt is not an error.** No candidates plus a
//!   `promptFeedback.blockReason` is one empty completion with `ContentFilter`.
//! * **Ids.** `responseId` is kept; otherwise a unique one is generated.
//!   `createTime` becomes `created`; otherwise it is `None`. A call Google did
//!   not id gets a fabricated one, because the gateway's is required and a
//!   chat-completions caller must echo it. It carries the response id so two
//!   turns never share one, and the order and names are as received.
//!
//! # Retained, not yet read
//!
//! Each candidate's whole `parts` array rides `ext["google"]["parts"]`.
//! Gateway blocks have no `ext`, so this is the only place a function call's
//! `thoughtSignature` can live. Nothing replays it yet: that is future work for
//! the request side, and a comment there should point here.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::gateway::google::namespaced;
use crate::gateway::types::{
    self, ContentBlock, FinishReason, IngestSource, MediaSource, ProviderExt, RawJson,
    ReasoningDetail, Role,
};
use crate::protocol::UnknownFields;
use crate::protocol::google::generate_content::types::{
    Candidate, Content, GenerateContentResponse, Part, UsageMetadata,
};
use crate::types::Protocol;

/// Infallible: capture, don't validate. The destructures are exhaustive so a
/// newly typed wire field cannot be forgotten here.
// Dead until the transport (subtask 6 of #35) calls it.
#[allow(dead_code)]
pub(crate) fn ingest_response(response: GenerateContentResponse) -> types::ChatResponse {
    let body = plain(&response);
    let GenerateContentResponse {
        candidates,
        usage_metadata,
        model_version,
        response_id,
        create_time,
        prompt_feedback,
        unknown_fields,
    } = response;
    let id = response_id.unwrap_or_else(synthetic_response_id);

    let mut ordered: Vec<(usize, Candidate)> = candidates
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .collect();
    // Stable, and a candidate with no `index` keeps its position.
    ordered.sort_by_key(|(position, candidate)| {
        candidate.index.map_or(*position, |index| index as usize)
    });
    let mut completions: Vec<types::Completion> = ordered
        .into_iter()
        .map(|(position, candidate)| ingest_candidate(candidate, &id, position))
        .collect();

    // A blocked prompt: the request succeeded and there is nothing to say.
    let block_reason = prompt_feedback
        .as_ref()
        .and_then(|feedback| feedback.block_reason.clone());
    if completions.is_empty()
        && let Some(reason) = block_reason
    {
        completions.push(types::Completion {
            message: types::Message {
                role: Role::Assistant,
                content: Vec::new(),
                ext: ProviderExt::new(),
            },
            finish_reason: FinishReason::ContentFilter,
            finish_reason_raw: reason,
            ext: ProviderExt::new(),
        });
    }

    let mut residue = unknown_fields;
    if let Some(feedback) = &prompt_feedback {
        residue.insert("promptFeedback".into(), plain(feedback));
    }
    if let Some(time) = &create_time {
        residue.insert("createTime".into(), Value::String(time.clone()));
    }

    types::ChatResponse {
        id,
        model: model_version.unwrap_or_default(),
        created: create_time.as_deref().and_then(parse_rfc3339),
        completions,
        usage: usage_metadata.map(ingest_usage),
        ext: namespaced(residue),
        source: Some(IngestSource {
            protocol: Protocol::Google,
            body,
        }),
    }
}

/// A response id for a reply that arrived with none.
///
/// Unique within the process. A streamed reply must call this ONCE and reuse
/// the result on every chunk (subtask 8 of #35).
#[allow(dead_code)]
pub(crate) fn synthetic_response_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or(0);
    format!(
        "gemini-{nanos:x}-{}",
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

fn ingest_candidate(candidate: Candidate, response_id: &str, position: usize) -> types::Completion {
    let Candidate {
        content,
        finish_reason,
        index,
        unknown_fields,
    } = candidate;

    let mut message_ext = UnknownFields::new();
    let mut blocks = Vec::new();
    if let Some(content) = content {
        let Content {
            role,
            parts,
            unknown_fields: content_residue,
        } = content;
        if let Some(raw) = role.filter(|role| role != "model") {
            message_ext.insert("role".into(), Value::String(raw));
        }
        let mut calls = 0usize;
        for part in &parts {
            blocks.push(ingest_part(part, response_id, position, &mut calls));
        }
        // Retained: see this module's docs.
        message_ext.insert("parts".into(), plain(&parts));
        if !content_residue.is_empty() {
            message_ext.insert("content_fields".into(), Value::Object(content_residue));
        }
    }

    let mut residue = unknown_fields;
    if let Some(index) = index {
        residue.insert("index".into(), Value::from(index));
    }
    types::Completion {
        message: types::Message {
            role: Role::Assistant,
            content: blocks,
            ext: namespaced(message_ext),
        },
        finish_reason: finish_reason_of(finish_reason.as_deref()),
        finish_reason_raw: finish_reason.unwrap_or_default(),
        ext: namespaced(residue),
    }
}

/// Google's `finishReason` → the gateway's programmatic enum. A table over
/// Google's own string, with nothing derived from the parts.
fn finish_reason_of(raw: Option<&str>) -> FinishReason {
    match raw {
        Some("STOP") => FinishReason::Stop,
        Some("MAX_TOKENS") => FinishReason::Length,
        Some(
            "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "IMAGE_SAFETY",
        ) => FinishReason::ContentFilter,
        _ => FinishReason::Other,
    }
}

/// One part as a gateway block.
///
/// A part is whichever field is present, so the order below is the precedence:
/// a call, then thinking text, then text, then bytes. A part with none of them
/// (a `fileData` reference, say) becomes `Unknown` carrying the whole part.
fn ingest_part(
    part: &Part,
    response_id: &str,
    candidate: usize,
    calls: &mut usize,
) -> ContentBlock {
    if let Some(call) = &part.function_call {
        let n = *calls;
        *calls += 1;
        let empty = Value::Object(Default::default());
        return ContentBlock::ToolCall {
            id: call
                .id
                .clone()
                .unwrap_or_else(|| format!("call_{response_id}_{candidate}_{n}")),
            name: call.name.clone(),
            arguments: RawJson::from_value(call.args.as_ref().unwrap_or(&empty)),
        };
    }
    if let Some(text) = &part.text {
        if part.thought == Some(true) {
            return ContentBlock::Reasoning {
                detail: ReasoningDetail::Text {
                    text: text.clone(),
                    signature: part.thought_signature.clone(),
                },
                provenance: Some(Protocol::Google.as_str().to_string()),
                id: None,
            };
        }
        return ContentBlock::Text {
            text: text.clone(),
            cache: None,
        };
    }
    if let Some(blob) = &part.inline_data {
        let source = MediaSource::Base64 {
            media_type: blob.mime_type.clone(),
            data: blob.data.clone(),
        };
        return if blob.mime_type.starts_with("image/") {
            ContentBlock::Image {
                source,
                detail: None,
                cache: None,
            }
        } else if blob.mime_type.starts_with("audio/") {
            ContentBlock::Audio {
                source,
                format: None,
            }
        } else {
            ContentBlock::Document {
                source,
                title: None,
                cache: None,
            }
        };
    }
    ContentBlock::Unknown(plain(part))
}

/// Google's usage as the gateway's.
///
/// Both folds are Google's documented arithmetic: its `totalTokenCount` is
/// prompt + candidates + tool-use + thoughts, so input and output sum to it.
/// The raw total and the tool-use count are stashed so nothing is lost.
fn ingest_usage(usage: UsageMetadata) -> types::Usage {
    let UsageMetadata {
        prompt_token_count,
        candidates_token_count,
        total_token_count,
        cached_content_token_count,
        thoughts_token_count,
        tool_use_prompt_token_count,
        unknown_fields,
    } = usage;
    let input_tokens = add(prompt_token_count, tool_use_prompt_token_count);
    let output_tokens = add(candidates_token_count, thoughts_token_count);
    let total_tokens = match (input_tokens, output_tokens) {
        (None, None) => None,
        (input, output) => Some(input.unwrap_or(0) + output.unwrap_or(0)),
    };

    let mut residue = unknown_fields;
    if let Some(total) = total_token_count {
        residue.insert("totalTokenCount".into(), Value::from(total));
    }
    if let Some(tool_use) = tool_use_prompt_token_count {
        residue.insert("toolUsePromptTokenCount".into(), Value::from(tool_use));
    }
    types::Usage {
        input_tokens,
        output_tokens,
        total_tokens,
        reasoning_tokens: thoughts_token_count.map(u64::from),
        cache_read_tokens: cached_content_token_count.map(u64::from),
        cache_write_tokens: None,
        ext: namespaced(residue),
    }
}

/// A sum that is absent only when both addends are.
fn add(a: Option<u32>, b: Option<u32>) -> Option<u64> {
    match (a, b) {
        (None, None) => None,
        _ => Some(u64::from(a.unwrap_or(0)) + u64::from(b.unwrap_or(0))),
    }
}

fn plain<T: serde::Serialize>(value: &T) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

/// `createTime` as Unix seconds. Accepts `Z` or a numeric offset and drops any
/// fraction; anything else is `None`, which leaves `created` unset.
fn parse_rfc3339(text: &str) -> Option<u64> {
    let (date, rest) = text.split_once('T')?;
    let mut parts = date.split('-');
    let year: i64 = parts.next()?.parse().ok()?;
    let month: i64 = parts.next()?.parse().ok()?;
    let day: i64 = parts.next()?.parse().ok()?;
    if parts.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let (time, offset) = match rest.strip_suffix('Z') {
        Some(time) => (time, 0),
        None => {
            let at = rest.rfind(['+', '-'])?;
            let (time, offset) = rest.split_at(at);
            let sign = if offset.starts_with('-') { -1 } else { 1 };
            let (hours, minutes) = offset[1..].split_once(':')?;
            let seconds = hours.parse::<i64>().ok()? * 3600 + minutes.parse::<i64>().ok()? * 60;
            (time, sign * seconds)
        }
    };
    let mut clock = time.split('.').next()?.split(':');
    let hours: i64 = clock.next()?.parse().ok()?;
    let minutes: i64 = clock.next()?.parse().ok()?;
    let seconds: i64 = clock.next()?.parse().ok()?;
    if clock.next().is_some() || hours > 23 || minutes > 59 || seconds > 60 {
        return None;
    }
    let total =
        days_from_civil(year, month, day) * 86_400 + hours * 3600 + minutes * 60 + seconds - offset;
    u64::try_from(total).ok()
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Hinnant's algorithm).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn wire(body: Value) -> GenerateContentResponse {
        serde_json::from_value(body).unwrap()
    }

    fn ingest(body: Value) -> types::ChatResponse {
        ingest_response(wire(body))
    }

    #[test]
    fn a_text_reply_keeps_its_id_model_time_and_finish() {
        let response = ingest(json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "hi"}]},
                "finishReason": "STOP",
                "index": 0
            }],
            "modelVersion": "gemini-2.5-flash",
            "responseId": "resp_1",
            "createTime": "2026-11-26T01:14:00.123456Z"
        }));
        assert_eq!(response.id, "resp_1");
        assert_eq!(response.model, "gemini-2.5-flash");
        assert_eq!(response.created, Some(1_795_655_640));
        let completion = &response.completions[0];
        assert!(
            matches!(&completion.message.content[..], [ContentBlock::Text { text, .. }] if text == "hi")
        );
        assert_eq!(completion.finish_reason, FinishReason::Stop);
        assert_eq!(completion.finish_reason_raw, "STOP");
    }

    #[test]
    fn a_missing_response_id_is_generated_and_unique() {
        let first = ingest(json!({"candidates": []}));
        let second = ingest(json!({"candidates": []}));
        assert!(!first.id.is_empty());
        assert_ne!(first.id, second.id);
        assert_eq!(first.created, None);
    }

    #[test]
    fn a_function_call_keeps_order_names_and_a_given_id() {
        let response = ingest(json!({
            "responseId": "r1",
            "candidates": [{
                "content": {"role": "model", "parts": [
                    {"functionCall": {"id": "given", "name": "a", "args": {"x": 1}}},
                    {"functionCall": {"name": "b", "args": {}}},
                    {"functionCall": {"name": "b"}}
                ]},
                "finishReason": "STOP"
            }]
        }));
        let completion = &response.completions[0];
        let calls: Vec<_> = completion
            .message
            .content
            .iter()
            .map(|block| match block {
                ContentBlock::ToolCall {
                    id,
                    name,
                    arguments,
                } => (id.clone(), name.clone(), arguments.as_str().to_string()),
                other => panic!("not a call: {other:?}"),
            })
            .collect();
        assert_eq!(calls[0], ("given".into(), "a".into(), r#"{"x":1}"#.into()));
        assert_eq!(calls[1].0, "call_r1_0_1");
        assert_eq!(calls[2].0, "call_r1_0_2");
        assert_eq!((calls[1].1.as_str(), calls[2].1.as_str()), ("b", "b"));
        assert_eq!(calls[2].2, "{}", "no args is an empty object");
        // Google's own finish reason, not one derived from the parts.
        assert_eq!(completion.finish_reason, FinishReason::Stop);
        assert_eq!(completion.finish_reason_raw, "STOP");
    }

    #[test]
    fn fabricated_call_ids_never_repeat_across_turns() {
        let ids: Vec<String> = ["r1", "r2"]
            .into_iter()
            .map(|id| {
                let response = ingest(json!({
                    "responseId": id,
                    "candidates": [{"content": {"parts": [{"functionCall": {"name": "f"}}]}}]
                }));
                match &response.completions[0].message.content[0] {
                    ContentBlock::ToolCall { id, .. } => id.clone(),
                    other => panic!("{other:?}"),
                }
            })
            .collect();
        assert_ne!(ids[0], ids[1]);
    }

    #[test]
    fn thinking_becomes_reasoning_and_signatures_are_retained() {
        let response = ingest(json!({
            "candidates": [{"content": {"role": "model", "parts": [
                {"text": "hmm", "thought": true, "thoughtSignature": "c2ln"},
                {"functionCall": {"name": "f", "args": {}}, "thoughtSignature": "ZnVuYw=="}
            ]}}]
        }));
        let message = &response.completions[0].message;
        assert!(matches!(
            &message.content[0],
            ContentBlock::Reasoning { detail: ReasoningDetail::Text { text, signature }, provenance, .. }
                if text == "hmm" && signature.as_deref() == Some("c2ln") && provenance.as_deref() == Some("google")
        ));
        // The function call's signature has no block field, so it rides here.
        assert_eq!(
            message.ext["google"]["parts"][1]["thoughtSignature"],
            json!("ZnVuYw==")
        );
    }

    #[test]
    fn usage_folds_tool_use_into_input_and_thoughts_into_output() {
        let response = ingest(json!({"usageMetadata": {
            "promptTokenCount": 10,
            "toolUsePromptTokenCount": 4,
            "candidatesTokenCount": 6,
            "thoughtsTokenCount": 5,
            "cachedContentTokenCount": 3,
            "totalTokenCount": 25
        }}));
        let usage = response.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(14));
        assert_eq!(usage.output_tokens, Some(11));
        assert_eq!(
            usage.total_tokens,
            Some(25),
            "input + output is Google's total"
        );
        assert_eq!(usage.reasoning_tokens, Some(5));
        assert_eq!(usage.cache_read_tokens, Some(3));
        assert_eq!(usage.ext["google"]["totalTokenCount"], json!(25));
        assert_eq!(usage.ext["google"]["toolUsePromptTokenCount"], json!(4));
    }

    #[test]
    fn absent_usage_stays_absent() {
        assert!(ingest(json!({"candidates": []})).usage.is_none());
    }

    #[test]
    fn a_blocked_prompt_is_one_empty_content_filter_completion() {
        let response = ingest(json!({"promptFeedback": {
            "blockReason": "SAFETY",
            "safetyRatings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT"}]
        }}));
        assert_eq!(response.completions.len(), 1);
        let completion = &response.completions[0];
        assert!(completion.message.content.is_empty());
        assert_eq!(completion.finish_reason, FinishReason::ContentFilter);
        assert_eq!(completion.finish_reason_raw, "SAFETY");
        assert_eq!(
            response.ext["google"]["promptFeedback"]["safetyRatings"][0]["category"],
            json!("HARM_CATEGORY_DANGEROUS_CONTENT")
        );
    }

    #[test]
    fn no_candidates_and_no_block_reason_is_no_completions() {
        assert!(ingest(json!({"modelVersion": "x"})).completions.is_empty());
    }

    #[test]
    fn candidates_become_completions_in_index_order() {
        let response = ingest(json!({"candidates": [
            {"index": 1, "content": {"parts": [{"text": "second"}]}},
            {"index": 0, "content": {"parts": [{"text": "first"}]}}
        ]}));
        let texts: Vec<_> = response
            .completions
            .iter()
            .map(|completion| match &completion.message.content[0] {
                ContentBlock::Text { text, .. } => text.as_str(),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(texts, ["first", "second"]);
    }

    #[test]
    fn finish_reasons_map_from_googles_own_string() {
        for (raw, expected) in [
            ("STOP", FinishReason::Stop),
            ("MAX_TOKENS", FinishReason::Length),
            ("SAFETY", FinishReason::ContentFilter),
            ("PROHIBITED_CONTENT", FinishReason::ContentFilter),
            ("MALFORMED_FUNCTION_CALL", FinishReason::Other),
            ("SOMETHING_NEW", FinishReason::Other),
        ] {
            assert_eq!(finish_reason_of(Some(raw)), expected, "{raw}");
        }
        assert_eq!(finish_reason_of(None), FinishReason::Other);
    }

    #[test]
    fn media_and_unrecognized_parts_are_kept() {
        let response = ingest(json!({"candidates": [{"content": {"parts": [
            {"inlineData": {"mimeType": "image/png", "data": "aGk="}},
            {"inlineData": {"mimeType": "audio/wav", "data": "aGk="}},
            {"fileData": {"fileUri": "gs://b/o"}}
        ]}}]}));
        let content = &response.completions[0].message.content;
        assert!(matches!(content[0], ContentBlock::Image { .. }));
        assert!(matches!(content[1], ContentBlock::Audio { .. }));
        assert!(
            matches!(&content[2], ContentBlock::Unknown(value) if value["fileData"]["fileUri"] == "gs://b/o")
        );
    }

    #[test]
    fn candidate_residue_rides_the_completion_ext() {
        let response = ingest(json!({"candidates": [{
            "content": {"parts": [{"text": "x"}]},
            "groundingMetadata": {"webSearchQueries": ["q"]},
            "index": 0
        }]}));
        let ext = &response.completions[0].ext["google"];
        assert_eq!(ext["groundingMetadata"]["webSearchQueries"][0], json!("q"));
        assert_eq!(ext["index"], json!(0));
    }

    #[test]
    fn timestamps_parse_with_offsets_and_reject_nonsense() {
        assert_eq!(parse_rfc3339("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_rfc3339("2026-11-26T01:14:00Z"), Some(1_795_655_640));
        assert_eq!(parse_rfc3339("2026-11-26T01:14:00.9Z"), Some(1_795_655_640));
        assert_eq!(
            parse_rfc3339("2026-11-26T01:14:00-06:00"),
            Some(1_795_677_240)
        );
        for bad in [
            "",
            "yesterday",
            "2026-13-01T00:00:00Z",
            "2026-11-26T25:00:00Z",
            "2026-11-26",
        ] {
            assert_eq!(parse_rfc3339(bad), None, "{bad}");
        }
    }
}
