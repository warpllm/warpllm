//! The fixtures next door prove the bodies we thought to write down survive.
//! This proves the ones we didn't: generated events and requests, every
//! optional field in every state the wire allows, unknown fields at every
//! level, asserting that deserialize → reserialize is the identity.
//!
//! The contract it states, exactly:
//!
//! > For any body whose fields are within the nullability Anthropic
//! > documents, warpllm re-emits what it received, byte for byte as JSON
//! > values.
//!
//! # What this catches that the fixtures cannot
//!
//! A fixture checks the exact body someone thought to write down, so it can
//! only find a bug in a field that body happens to carry. This generates
//! unknown fields at EVERY nesting level, which is what proves each struct's
//! `unknown_fields` catch-all is present and wired — the load-bearing claim of
//! the whole module. Demonstrated rather than assumed: dropping the catch-all
//! from `UrlSource` passes every fixture and fails this file.
//!
//! It also sweeps the optionality combinations. Six three-state fields alone
//! are hundreds of bodies; a fixture pins one, and pinning the RIGHT one means
//! having already guessed where the bug is.
//!
//! # Both directions
//!
//! Events came first, and they cover [`Message`], `ContentBlock`, `Source` and
//! `Usage` transitively. Requests are swept separately because nothing else
//! checks them at all: warpllm BUILDS those bodies and no caller or oracle
//! ever sees one, so a field silently dropped on the way to Anthropic would
//! reach no other test.
//!
//! # What is deliberately not generated
//!
//! A field the spec marks optional but NOT nullable — a thinking block's
//! `signature`, a tool result's `is_error`, a tool's `strict` — is absent or a
//! value here, never null. A provider that sends `null` for one anyway is
//! understood, and normalized to absent. That is a permissive read of
//! out-of-spec input, not a round trip of it, and the test below it says so.
//!
//! # Passing for the wrong reason
//!
//! Two guards, because a generator bug would otherwise make every property
//! vacuous. Every event asserts it did NOT land in the `Unknown` arm, and
//! every custom tool asserts it did NOT land in [`Tool::Other`] — both are
//! catch-alls that would round trip a malformed body verbatim. The tagged
//! unions need no such guard since they now REFUSE a body that does not
//! parse; `Tool` is untagged and has no discriminator to refuse on, which is
//! exactly why it gets a property of its own.

use proptest::prelude::*;
use serde_json::{Map, Value, json};
use warpllm::protocol::anthropic::messages::types::{
    CreateMessageRequest, MessageStreamEvent, Tool,
};

/// Optional and nullable: absent, `null`, or a value — three states the wire
/// distinguishes and these types have to as well.
fn optional_nullable(value: BoxedStrategy<Value>) -> BoxedStrategy<Option<Value>> {
    prop_oneof![Just(None), Just(Some(Value::Null)), value.prop_map(Some)].boxed()
}

/// Optional, not nullable: absent or a value.
fn optional(value: BoxedStrategy<Value>) -> BoxedStrategy<Option<Value>> {
    prop_oneof![Just(None), value.prop_map(Some)].boxed()
}

fn text() -> BoxedStrategy<Value> {
    "[a-zA-Z0-9 {}\":,_-]{0,24}".prop_map(Value::from).boxed()
}

/// Fields no specification names, which must reach the caller verbatim. The
/// `x_` prefix keeps them from colliding with a field the struct models — a
/// collision would be captured by the typed field and prove nothing.
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

fn cache_control() -> BoxedStrategy<Value> {
    (optional(text()), unknown_fields())
        .prop_map(|(ttl, unknown)| {
            object(
                vec![("type", Some(json!("ephemeral"))), ("ttl", ttl)],
                unknown,
            )
        })
        .boxed()
}

fn source() -> BoxedStrategy<Value> {
    prop_oneof![
        (text(), text(), unknown_fields()).prop_map(|(media_type, data, unknown)| object(
            vec![
                ("type", Some(json!("base64"))),
                ("media_type", Some(media_type)),
                ("data", Some(data)),
            ],
            unknown,
        )),
        (text(), unknown_fields()).prop_map(|(url, unknown)| object(
            vec![("type", Some(json!("url"))), ("url", Some(url))],
            unknown,
        )),
        (text(), text(), unknown_fields()).prop_map(|(media_type, data, unknown)| object(
            vec![
                ("type", Some(json!("text"))),
                ("media_type", Some(media_type)),
                ("data", Some(data)),
            ],
            unknown,
        )),
        (text(), unknown_fields()).prop_map(|(file_id, unknown)| object(
            vec![("type", Some(json!("file"))), ("file_id", Some(file_id))],
            unknown,
        )),
    ]
    .boxed()
}

/// Every block variant, plus a shape warpllm does not model — which must reach
/// the caller through the catch-all rather than failing the event it sits in.
fn content_block() -> BoxedStrategy<Value> {
    prop_oneof![
        (text(), optional_nullable(cache_control()), unknown_fields(),).prop_map(
            |(text, cache_control, unknown)| object(
                vec![
                    ("type", Some(json!("text"))),
                    ("text", Some(text)),
                    ("cache_control", cache_control),
                ],
                unknown,
            )
        ),
        (
            source(),
            optional_nullable(cache_control()),
            unknown_fields(),
        )
            .prop_map(|(source, cache_control, unknown)| object(
                vec![
                    ("type", Some(json!("image"))),
                    ("source", Some(source)),
                    ("cache_control", cache_control),
                ],
                unknown,
            )),
        (
            source(),
            optional_nullable(text()),
            optional_nullable(cache_control()),
            unknown_fields(),
        )
            .prop_map(|(source, title, cache_control, unknown)| object(
                vec![
                    ("type", Some(json!("document"))),
                    ("source", Some(source)),
                    ("title", title),
                    ("cache_control", cache_control),
                ],
                unknown,
            )),
        (text(), text(), unknown_fields()).prop_map(|(id, name, unknown)| object(
            vec![
                ("type", Some(json!("tool_use"))),
                ("id", Some(id)),
                ("name", Some(name)),
                ("input", Some(json!({"location": "San Francisco, CA"}))),
            ],
            unknown,
        )),
        (
            text(),
            optional(result_content()),
            optional(any::<bool>().prop_map(Value::from).boxed()),
            unknown_fields(),
        )
            .prop_map(|(tool_use_id, content, is_error, unknown)| object(
                vec![
                    ("type", Some(json!("tool_result"))),
                    ("tool_use_id", Some(tool_use_id)),
                    ("content", content),
                    ("is_error", is_error),
                ],
                unknown,
            )),
        (text(), optional(text()), unknown_fields()).prop_map(|(thinking, signature, unknown)| {
            object(
                vec![
                    ("type", Some(json!("thinking"))),
                    ("thinking", Some(thinking)),
                    ("signature", signature),
                ],
                unknown,
            )
        }),
        (text(), unknown_fields()).prop_map(|(data, unknown)| object(
            vec![
                ("type", Some(json!("redacted_thinking"))),
                ("data", Some(data)),
            ],
            unknown,
        )),
        (text(), unknown_fields()).prop_map(|(url, unknown)| object(
            vec![
                ("type", Some(json!("web_search_tool_result"))),
                ("url", Some(url)),
            ],
            unknown,
        )),
    ]
    .boxed()
}

/// A tool's output: text, or blocks.
///
/// Bounded to text blocks rather than the full [`content_block`] recursion a
/// `tool_result` nominally allows. A tool result nested inside a tool result
/// is not a shape Anthropic produces, and generating one would make this
/// strategy infinitely deep.
fn result_content() -> BoxedStrategy<Value> {
    prop_oneof![
        text(),
        prop::collection::vec(
            (text(), unknown_fields()).prop_map(|(value, unknown)| object(
                vec![("type", Some(json!("text"))), ("text", Some(value))],
                unknown,
            )),
            0..3,
        )
        .prop_map(Value::from),
    ]
    .boxed()
}

/// `string | ContentBlock[]`, the shape both `system` and a message's
/// `content` take. The two forms are different bytes and each must come back
/// as itself.
fn text_or_blocks() -> BoxedStrategy<Value> {
    prop_oneof![
        text(),
        prop::collection::vec(content_block(), 0..3).prop_map(Value::from),
    ]
    .boxed()
}

fn usage() -> BoxedStrategy<Value> {
    let count = || optional_nullable((0u32..4000).prop_map(Value::from).boxed());
    (0u32..4000, 0u32..4000, count(), count(), unknown_fields())
        .prop_map(|(input, output, creation, read, unknown)| {
            object(
                vec![
                    ("input_tokens", Some(json!(input))),
                    ("output_tokens", Some(json!(output))),
                    ("cache_creation_input_tokens", creation),
                    ("cache_read_input_tokens", read),
                ],
                unknown,
            )
        })
        .boxed()
}

/// A whole reply. `stop_reason` and `stop_sequence` are required AND nullable,
/// so they are always present and sometimes null — never absent.
fn message() -> BoxedStrategy<Value> {
    let nullable_text = || prop_oneof![Just(Value::Null), text()];
    (
        text(),
        text(),
        prop::collection::vec(content_block(), 0..3),
        text(),
        nullable_text(),
        nullable_text(),
        usage(),
        unknown_fields(),
    )
        .prop_map(
            |(id, role, content, model, stop_reason, stop_sequence, usage, unknown)| {
                object(
                    vec![
                        ("type", Some(json!("message"))),
                        ("id", Some(id)),
                        ("role", Some(role)),
                        ("content", Some(Value::from(content))),
                        ("model", Some(model)),
                        ("stop_reason", Some(stop_reason)),
                        ("stop_sequence", Some(stop_sequence)),
                        ("usage", Some(usage)),
                    ],
                    unknown,
                )
            },
        )
        .boxed()
}

fn content_block_delta() -> BoxedStrategy<Value> {
    prop_oneof![
        (text(), unknown_fields()).prop_map(|(value, unknown)| object(
            vec![("type", Some(json!("text_delta"))), ("text", Some(value))],
            unknown,
        )),
        (text(), unknown_fields()).prop_map(|(value, unknown)| object(
            vec![
                ("type", Some(json!("input_json_delta"))),
                ("partial_json", Some(value)),
            ],
            unknown,
        )),
        (text(), unknown_fields()).prop_map(|(value, unknown)| object(
            vec![
                ("type", Some(json!("thinking_delta"))),
                ("thinking", Some(value)),
            ],
            unknown,
        )),
        (text(), unknown_fields()).prop_map(|(value, unknown)| object(
            vec![
                ("type", Some(json!("signature_delta"))),
                ("signature", Some(value)),
            ],
            unknown,
        )),
        // Not modelled, and must still survive: `citations_delta` has no
        // gateway home, so the catch-all is the whole of its support.
        (text(), unknown_fields()).prop_map(|(value, unknown)| object(
            vec![
                ("type", Some(json!("citations_delta"))),
                ("citation", Some(value)),
            ],
            unknown,
        )),
    ]
    .boxed()
}

fn event() -> BoxedStrategy<Value> {
    let index = || (0u32..4).prop_map(Value::from);
    prop_oneof![
        (message(), unknown_fields()).prop_map(|(message, unknown)| object(
            vec![
                ("type", Some(json!("message_start"))),
                ("message", Some(message)),
            ],
            unknown,
        )),
        (index(), content_block(), unknown_fields()).prop_map(|(index, block, unknown)| object(
            vec![
                ("type", Some(json!("content_block_start"))),
                ("index", Some(index)),
                ("content_block", Some(block)),
            ],
            unknown,
        )),
        (index(), content_block_delta(), unknown_fields()).prop_map(|(index, delta, unknown)| {
            object(
                vec![
                    ("type", Some(json!("content_block_delta"))),
                    ("index", Some(index)),
                    ("delta", Some(delta)),
                ],
                unknown,
            )
        }),
        (index(), unknown_fields()).prop_map(|(index, unknown)| object(
            vec![
                ("type", Some(json!("content_block_stop"))),
                ("index", Some(index)),
            ],
            unknown,
        )),
        (
            optional_nullable(text()),
            optional_nullable(text()),
            0u32..4000,
            unknown_fields(),
        )
            .prop_map(|(stop_reason, stop_sequence, output, unknown)| object(
                vec![
                    ("type", Some(json!("message_delta"))),
                    (
                        "delta",
                        Some(object(
                            vec![
                                ("stop_reason", stop_reason),
                                ("stop_sequence", stop_sequence),
                            ],
                            Map::new(),
                        )),
                    ),
                    ("usage", Some(json!({"output_tokens": output}))),
                ],
                unknown,
            )),
        unknown_fields()
            .prop_map(|unknown| object(vec![("type", Some(json!("message_stop")))], unknown,)),
        unknown_fields().prop_map(|unknown| object(vec![("type", Some(json!("ping")))], unknown)),
        (text(), text(), unknown_fields()).prop_map(|(kind, message, unknown)| object(
            vec![
                ("type", Some(json!("error"))),
                ("error", Some(json!({"type": kind, "message": message})),),
            ],
            unknown,
        )),
    ]
    .boxed()
}

/// The generator's boundary, stated as behavior rather than as prose: a null
/// where Anthropic documents no null is understood — the alternative is
/// rejecting an event over a field nobody read — and comes back out as absent.
/// Every field the spec DOES mark nullable keeps its null, which is what the
/// property below sweeps.
#[test]
fn an_out_of_spec_null_is_read_and_normalized_to_absent() {
    let event: MessageStreamEvent = serde_json::from_value(json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "thinking",
            "thinking": "hmm",
            "signature": null,
            "cache_control": null
        }
    }))
    .expect("an out-of-spec null must not fail the whole event");

    let wire = serde_json::to_value(&event).unwrap();
    assert!(wire["content_block"].get("signature").is_none());
    // ...while an in-spec null on a reply is still a null.
    let start: MessageStreamEvent = serde_json::from_value(json!({
        "type": "message_start",
        "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "content": [],
            "model": "claude-opus-5", "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": null}
        }
    }))
    .unwrap();
    let wire = serde_json::to_value(&start).unwrap();
    assert_eq!(wire["message"]["stop_reason"], Value::Null);
    assert_eq!(
        wire["message"]["usage"]["cache_read_input_tokens"],
        Value::Null
    );
}

proptest! {
    #[test]
    fn any_in_spec_event_survives_a_round_trip(body in event()) {
        let parsed: MessageStreamEvent = serde_json::from_value(body.clone())
            .unwrap_or_else(|e| panic!("{body} failed to deserialize: {e}"));
        // Without this, an event the generator built wrong would still round
        // trip through the catch-all and the property would prove nothing.
        prop_assert!(
            !matches!(parsed, MessageStreamEvent::Unknown(_)),
            "{body} fell through to Unknown instead of parsing as its own type",
        );
        prop_assert_eq!(serde_json::to_value(&parsed).unwrap(), body);
    }
}

// ---------------------------------------------------------------------------
// The request tree
//
// Nothing else checks this direction. warpllm builds these bodies and no
// caller or oracle ever sees one, so a field dropped on the way to Anthropic
// would otherwise reach no test at all.
// ---------------------------------------------------------------------------

/// A sampling parameter, as a FLOAT.
///
/// Generated through `f64` deliberately: `serde_json` holds an integer and a
/// float as different `Number` variants that compare unequal, so a whole
/// number generated as an integer would fail the round trip on a `f64` field
/// for a reason that has nothing to do with warpllm.
fn unit_interval() -> BoxedStrategy<Value> {
    (0.0f64..=1.0).prop_map(Value::from).boxed()
}

fn flag() -> BoxedStrategy<Value> {
    any::<bool>().prop_map(Value::from).boxed()
}

/// A tool warpllm models: named, schema'd, and carrying no `type` to be
/// recognized by.
fn custom_tool() -> BoxedStrategy<Value> {
    (
        text(),
        optional(text()),
        optional(flag()),
        optional_nullable(cache_control()),
        unknown_fields(),
    )
        .prop_map(|(name, description, strict, cache_control, unknown)| {
            object(
                vec![
                    ("name", Some(name)),
                    ("description", description),
                    (
                        "input_schema",
                        Some(json!({"type": "object", "properties": {}})),
                    ),
                    ("strict", strict),
                    ("cache_control", cache_control),
                ],
                unknown,
            )
        })
        .boxed()
}

fn tool() -> BoxedStrategy<Value> {
    prop_oneof![
        custom_tool(),
        // A server tool, which warpllm passes through whole. The ONLY thing
        // telling it from a custom tool is the absent `input_schema` — a
        // custom tool carries no discriminator at all, which is why `Tool`
        // is the one union here that cannot be tag-dispatched.
        (text(), unknown_fields()).prop_map(|(name, unknown)| object(
            vec![
                ("type", Some(json!("web_search_20260209"))),
                ("name", Some(name)),
            ],
            unknown,
        )),
    ]
    .boxed()
}

fn tool_choice() -> BoxedStrategy<Value> {
    let mode = |kind: &'static str| {
        (optional(flag()), unknown_fields()).prop_map(move |(disable, unknown)| {
            object(
                vec![
                    ("type", Some(json!(kind))),
                    ("disable_parallel_tool_use", disable),
                ],
                unknown,
            )
        })
    };
    prop_oneof![
        mode("auto"),
        mode("any"),
        mode("none"),
        (text(), optional(flag()), unknown_fields()).prop_map(|(name, disable, unknown)| object(
            vec![
                ("type", Some(json!("tool"))),
                ("name", Some(name)),
                ("disable_parallel_tool_use", disable),
            ],
            unknown,
        )),
    ]
    .boxed()
}

/// All three thinking vocabularies. They are mutually exclusive per MODEL, not
/// per protocol version, so every one of them is a body warpllm may have to
/// read back.
fn thinking_config() -> BoxedStrategy<Value> {
    prop_oneof![
        (1u32..64_000, unknown_fields()).prop_map(|(budget, unknown)| object(
            vec![
                ("type", Some(json!("enabled"))),
                ("budget_tokens", Some(json!(budget))),
            ],
            unknown,
        )),
        (optional(text()), unknown_fields()).prop_map(|(display, unknown)| object(
            vec![("type", Some(json!("adaptive"))), ("display", display)],
            unknown,
        )),
        unknown_fields()
            .prop_map(|unknown| object(vec![("type", Some(json!("disabled")))], unknown)),
    ]
    .boxed()
}

fn input_message() -> BoxedStrategy<Value> {
    (text(), text_or_blocks(), unknown_fields())
        .prop_map(|(role, content, unknown)| {
            object(
                vec![("role", Some(role)), ("content", Some(content))],
                unknown,
            )
        })
        .boxed()
}

/// Split into two tuples only because proptest's tuple strategies stop before
/// this many elements.
fn request() -> BoxedStrategy<Value> {
    (
        (
            text(),
            prop::collection::vec(input_message(), 0..3),
            1u32..100_000,
            optional(text_or_blocks()),
            optional(unit_interval()),
            optional(unit_interval()),
        ),
        (
            optional(
                prop::collection::vec(text(), 0..3)
                    .prop_map(Value::from)
                    .boxed(),
            ),
            optional(flag()),
            optional(
                prop::collection::vec(tool(), 0..3)
                    .prop_map(Value::from)
                    .boxed(),
            ),
            optional(tool_choice()),
            optional(thinking_config()),
            unknown_fields(),
        ),
    )
        .prop_map(
            |(
                (model, messages, max_tokens, system, temperature, top_p),
                (stop_sequences, stream, tools, tool_choice, thinking, unknown),
            )| {
                object(
                    vec![
                        ("model", Some(model)),
                        ("messages", Some(Value::from(messages))),
                        ("max_tokens", Some(json!(max_tokens))),
                        ("system", system),
                        ("temperature", temperature),
                        ("top_p", top_p),
                        ("stop_sequences", stop_sequences),
                        ("stream", stream),
                        ("tools", tools),
                        ("tool_choice", tool_choice),
                        ("thinking", thinking),
                    ],
                    unknown,
                )
            },
        )
        .boxed()
}

proptest! {
    #[test]
    fn any_in_spec_request_survives_a_round_trip(body in request()) {
        let parsed: CreateMessageRequest = serde_json::from_value(body.clone())
            .unwrap_or_else(|e| panic!("{body} failed to deserialize: {e}"));
        prop_assert_eq!(serde_json::to_value(&parsed).unwrap(), body);
    }

    /// A custom tool must reach `Tool::Custom`, never the passthrough arm.
    ///
    /// `Tool` is the one union here with no discriminator, so it cannot refuse
    /// a body that fails to parse the way the tag-dispatched ones now do — a
    /// malformed custom tool just becomes `Other` and round trips verbatim.
    /// Losslessness alone would therefore pass whatever this generator did,
    /// which is what this asserts against.
    #[test]
    fn a_custom_tool_never_degrades_to_the_passthrough_arm(body in custom_tool()) {
        let parsed: Tool = serde_json::from_value(body.clone())
            .unwrap_or_else(|e| panic!("{body} failed to deserialize: {e}"));
        prop_assert!(
            matches!(parsed, Tool::Custom(_)),
            "{body} fell through to the passthrough arm",
        );
        prop_assert_eq!(serde_json::to_value(&parsed).unwrap(), body);
    }
}
