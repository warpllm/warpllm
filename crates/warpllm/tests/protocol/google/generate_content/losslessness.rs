//! The fixtures next door prove the bodies we thought to write down survive.
//! This proves the ones we didn't: generated requests and responses, every
//! optional field present or absent, unknown fields at every level, asserting
//! that deserialize → reserialize is the identity.
//!
//! The contract it states, exactly:
//!
//! > For any body Google can send, warpllm re-emits what it received, as JSON
//! > values.
//!
//! # What this catches that the fixtures cannot
//!
//! A fixture checks the exact body someone wrote down, so it only finds a bug
//! in a field that body happens to carry. This generates unknown fields at
//! EVERY nesting level, which is what proves each struct's `unknown_fields`
//! catch-all is present and wired.
//!
//! # Passing for the wrong reason
//!
//! Google's `Part` is a struct of optionals with no `type` tag, so there is no
//! `Unknown` arm to guard against the way Anthropic's unions have. The
//! equivalent failure here is a MISSPELLED key in a generator: it would be
//! captured by `unknown_fields` and round trip anyway, proving nothing about
//! the typed field. Each property therefore asserts that the top-level
//! catch-all holds only the generator's own `x_` keys. That guards the top
//! level; nested typos are caught by the unit tests in `types.rs`.
//!
//! # Optionality
//!
//! Fields are absent or a value, never `null`: these types hold Google's
//! optionals as a plain `Option` that is skipped when `None`. If Google is
//! found sending `null`, the field moves to the three-state form and this file
//! gains a `null` arm for it.

use proptest::prelude::*;
use serde_json::{Map, Value, json};
use warpllm::protocol::google::generate_content::types::{
    GenerateContentRequest, GenerateContentResponse,
};

/// Optional, not nullable: absent or a value.
fn optional(value: BoxedStrategy<Value>) -> BoxedStrategy<Option<Value>> {
    prop_oneof![Just(None), value.prop_map(Some)].boxed()
}

fn text() -> BoxedStrategy<Value> {
    "[a-zA-Z0-9 {}\":,_-]{0,24}".prop_map(Value::from).boxed()
}

/// Fields no specification names, which must reach the caller verbatim. The
/// `x_` prefix keeps them from colliding with a field the struct models.
fn unknown_fields() -> BoxedStrategy<Map<String, Value>> {
    prop::collection::vec((0u8..8, text()), 0..3)
        .prop_map(|entries| {
            entries
                .into_iter()
                .map(|(n, value)| (format!("x_{n}"), value))
                .collect()
        })
        .boxed()
}

/// Assembles an object from named slots, dropping the absent ones.
fn object(slots: Vec<(&str, Option<Value>)>, unknown: Map<String, Value>) -> Value {
    let mut map = Map::new();
    for (name, value) in slots {
        if let Some(value) = value {
            map.insert(name.to_owned(), value);
        }
    }
    map.extend(unknown);
    Value::Object(map)
}

fn flag() -> BoxedStrategy<Value> {
    any::<bool>().prop_map(Value::from).boxed()
}

fn count() -> BoxedStrategy<Value> {
    (0u32..4000).prop_map(Value::from).boxed()
}

/// A sampling parameter, as a FLOAT.
///
/// Generated through `f64` deliberately: `serde_json` holds an integer and a
/// float as different `Number` variants that compare unequal, so a whole
/// number generated as an integer would fail on a `f64` field for a reason
/// that has nothing to do with warpllm.
fn unit_interval() -> BoxedStrategy<Value> {
    (0.0f64..=1.0).prop_map(Value::from).boxed()
}

fn json_object() -> BoxedStrategy<Value> {
    prop_oneof![
        Just(json!({})),
        Just(json!({"type": "object", "properties": {"city": {"type": "string"}}})),
    ]
    .boxed()
}

fn part() -> BoxedStrategy<Value> {
    prop_oneof![
        (text(), optional(flag()), optional(text()), unknown_fields()).prop_map(
            |(value, thought, signature, unknown)| object(
                vec![
                    ("text", Some(value)),
                    ("thought", thought),
                    ("thoughtSignature", signature),
                ],
                unknown,
            )
        ),
        (text(), text(), unknown_fields()).prop_map(|(mime, data, unknown)| object(
            vec![(
                "inlineData",
                Some(object(
                    vec![("mimeType", Some(mime)), ("data", Some(data))],
                    unknown.clone(),
                )),
            )],
            unknown,
        )),
        (
            optional(text()),
            text(),
            optional(json_object()),
            unknown_fields(),
        )
            .prop_map(|(id, name, args, unknown)| object(
                vec![(
                    "functionCall",
                    Some(object(
                        vec![("id", id), ("name", Some(name)), ("args", args)],
                        unknown,
                    )),
                )],
                Map::new(),
            )),
        (optional(text()), text(), json_object(), unknown_fields()).prop_map(
            |(id, name, response, unknown)| object(
                vec![(
                    "functionResponse",
                    Some(object(
                        vec![
                            ("id", id),
                            ("name", Some(name)),
                            ("response", Some(response))
                        ],
                        unknown,
                    )),
                )],
                Map::new(),
            )
        ),
        // Not modelled, and must still survive: a `fileData` part has no
        // gateway home yet, so the catch-all is the whole of its support.
        (text(), unknown_fields()).prop_map(|(uri, unknown)| object(
            vec![("fileData", Some(json!({"fileUri": uri})))],
            unknown,
        )),
    ]
    .boxed()
}

/// `parts` is always present: it serializes unconditionally.
fn content() -> BoxedStrategy<Value> {
    (
        optional(text()),
        prop::collection::vec(part(), 0..3),
        unknown_fields(),
    )
        .prop_map(|(role, parts, unknown)| {
            object(
                vec![("role", role), ("parts", Some(Value::from(parts)))],
                unknown,
            )
        })
        .boxed()
}

fn thinking_config() -> BoxedStrategy<Value> {
    (
        optional(flag()),
        optional((-1000i32..1000).prop_map(Value::from).boxed()),
        optional(text()),
        unknown_fields(),
    )
        .prop_map(|(include, budget, level, unknown)| {
            object(
                vec![
                    ("includeThoughts", include),
                    ("thinkingBudget", budget),
                    ("thinkingLevel", level),
                ],
                unknown,
            )
        })
        .boxed()
}

fn generation_config() -> BoxedStrategy<Value> {
    (
        (
            optional((1u32..4).prop_map(Value::from).boxed()),
            optional(
                prop::collection::vec(text(), 0..3)
                    .prop_map(Value::from)
                    .boxed(),
            ),
            optional((1u32..100_000).prop_map(Value::from).boxed()),
            optional(unit_interval()),
            optional(unit_interval()),
            optional((1u32..100).prop_map(Value::from).boxed()),
        ),
        (
            optional(text()),
            optional(json_object()),
            optional(thinking_config()),
            unknown_fields(),
        ),
    )
        .prop_map(
            |(
                (candidates, stops, max_tokens, temperature, top_p, top_k),
                (mime, schema, thinking, unknown),
            )| {
                object(
                    vec![
                        ("candidateCount", candidates),
                        ("stopSequences", stops),
                        ("maxOutputTokens", max_tokens),
                        ("temperature", temperature),
                        ("topP", top_p),
                        ("topK", top_k),
                        ("responseMimeType", mime),
                        ("responseJsonSchema", schema),
                        ("thinkingConfig", thinking),
                    ],
                    unknown,
                )
            },
        )
        .boxed()
}

fn function_declaration() -> BoxedStrategy<Value> {
    (
        text(),
        optional(text()),
        optional(json_object()),
        unknown_fields(),
    )
        .prop_map(|(name, description, schema, unknown)| {
            object(
                vec![
                    ("name", Some(name)),
                    ("description", description),
                    ("parametersJsonSchema", schema),
                ],
                unknown,
            )
        })
        .boxed()
}

fn tool() -> BoxedStrategy<Value> {
    prop_oneof![
        (
            prop::collection::vec(function_declaration(), 0..3),
            unknown_fields()
        )
            .prop_map(|(declarations, unknown)| object(
                vec![("functionDeclarations", Some(Value::from(declarations)))],
                unknown,
            )),
        // A built-in tool, which warpllm passes through whole.
        unknown_fields()
            .prop_map(|unknown| object(vec![("googleSearch", Some(json!({})))], unknown)),
    ]
    .boxed()
}

fn tool_config() -> BoxedStrategy<Value> {
    let calling = (
        optional(text()),
        optional(
            prop::collection::vec(text(), 0..3)
                .prop_map(Value::from)
                .boxed(),
        ),
        unknown_fields(),
    )
        .prop_map(|(mode, names, unknown)| {
            object(
                vec![("mode", mode), ("allowedFunctionNames", names)],
                unknown,
            )
        })
        .boxed();
    (optional(calling), unknown_fields())
        .prop_map(|(calling, unknown)| object(vec![("functionCallingConfig", calling)], unknown))
        .boxed()
}

fn request() -> BoxedStrategy<Value> {
    (
        prop::collection::vec(content(), 0..3),
        optional(content()),
        optional(generation_config()),
        optional(
            prop::collection::vec(tool(), 0..3)
                .prop_map(Value::from)
                .boxed(),
        ),
        optional(tool_config()),
        unknown_fields(),
    )
        .prop_map(
            |(contents, system, generation, tools, tool_config, unknown)| {
                object(
                    vec![
                        ("contents", Some(Value::from(contents))),
                        ("systemInstruction", system),
                        ("generationConfig", generation),
                        ("tools", tools),
                        ("toolConfig", tool_config),
                    ],
                    unknown,
                )
            },
        )
        .boxed()
}

fn candidate() -> BoxedStrategy<Value> {
    (
        optional(content()),
        optional(text()),
        optional((0u32..4).prop_map(Value::from).boxed()),
        unknown_fields(),
    )
        .prop_map(|(content, finish, index, unknown)| {
            object(
                vec![
                    ("content", content),
                    ("finishReason", finish),
                    ("index", index),
                ],
                unknown,
            )
        })
        .boxed()
}

fn usage_metadata() -> BoxedStrategy<Value> {
    (
        (
            optional(count()),
            optional(count()),
            optional(count()),
            optional(count()),
            optional(count()),
        ),
        unknown_fields(),
    )
        .prop_map(|((prompt, candidates, total, cached, thoughts), unknown)| {
            object(
                vec![
                    ("promptTokenCount", prompt),
                    ("candidatesTokenCount", candidates),
                    ("totalTokenCount", total),
                    ("cachedContentTokenCount", cached),
                    ("thoughtsTokenCount", thoughts),
                ],
                unknown,
            )
        })
        .boxed()
}

fn prompt_feedback() -> BoxedStrategy<Value> {
    (optional(text()), unknown_fields())
        .prop_map(|(reason, unknown)| object(vec![("blockReason", reason)], unknown))
        .boxed()
}

fn response() -> BoxedStrategy<Value> {
    (
        optional(
            prop::collection::vec(candidate(), 0..3)
                .prop_map(Value::from)
                .boxed(),
        ),
        optional(usage_metadata()),
        optional(text()),
        optional(text()),
        optional(prompt_feedback()),
        unknown_fields(),
    )
        .prop_map(
            |(candidates, usage, model_version, response_id, feedback, unknown)| {
                object(
                    vec![
                        ("candidates", candidates),
                        ("usageMetadata", usage),
                        ("modelVersion", model_version),
                        ("responseId", response_id),
                        ("promptFeedback", feedback),
                    ],
                    unknown,
                )
            },
        )
        .boxed()
}

/// Asserts the catch-all holds only keys the generator put there on purpose.
fn assert_only_generated_unknowns(unknown: &Map<String, Value>) -> Result<(), TestCaseError> {
    for key in unknown.keys() {
        prop_assert!(
            key.starts_with("x_"),
            "`{key}` landed in the catch-all: a generator key is misspelled"
        );
    }
    Ok(())
}

proptest! {
    #[test]
    fn any_request_survives_a_round_trip(body in request()) {
        let parsed: GenerateContentRequest = serde_json::from_value(body.clone())
            .unwrap_or_else(|e| panic!("{body} failed to deserialize: {e}"));
        assert_only_generated_unknowns(&parsed.unknown_fields)?;
        prop_assert_eq!(serde_json::to_value(&parsed).unwrap(), body);
    }

    #[test]
    fn any_response_survives_a_round_trip(body in response()) {
        let parsed: GenerateContentResponse = serde_json::from_value(body.clone())
            .unwrap_or_else(|e| panic!("{body} failed to deserialize: {e}"));
        assert_only_generated_unknowns(&parsed.unknown_fields)?;
        prop_assert_eq!(serde_json::to_value(&parsed).unwrap(), body);
    }
}
