"""What the generated types are supposed to say. Checked by `test_typing.py`,
never executed.

Each `# type: ignore[...]` is an assertion that the line IS an error. mypy runs
here with `--warn-unused-ignores`, so a suppression that stops being needed
fails the run -- the same bargain as TypeScript's `@ts-expect-error`, and the
reason these read backwards.
"""

from __future__ import annotations

from warpllm import (
    CreateChatCompletionRequest,
    CreateChatCompletionStreamResponse,
    WarpLLM,
)


def a_plain_dict_is_a_request(client: WarpLLM) -> None:
    client.chat_completions(
        {
            "model": "openai/gpt-5.6",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "max_tokens": 64,
        }
    )


def an_extension_parameter_is_accepted_at_the_boundary(
    client: WarpLLM,
) -> None:
    client.chat_completions(
        {
            "model": "openai/gpt-5.6",
            "messages": [{"role": "user", "content": "hi"}],
            "future_provider_parameter": {"enabled": True},
        }
    )


def the_generated_request_is_strict_when_requested() -> None:
    request: CreateChatCompletionRequest = {
        "model": "openai/gpt-5.6",
        "messages": [{"role": "user", "content": "hi"}],
        "temperture": 0.2,  # type: ignore[typeddict-unknown-key]
    }
    assert request


def the_response_is_typed(client: WarpLLM) -> None:
    completion = client.chat_completions({"model": "m", "messages": []})

    # Reading the response needs no import and no cast...
    content: str | None = completion["choices"][0]["message"]["content"]
    finish: str = completion["choices"][0]["finish_reason"]
    assert content is not None and finish

    # ...and a misspelled field is an error rather than `Any`.
    completion["choicez"]  # type: ignore[typeddict-item]


def iterating_a_stream_needs_no_casts(
    stream: list[CreateChatCompletionStreamResponse],
) -> str:
    """What a caller writes when the chunks start arriving.

    Every read here has to type-check without a cast or an `Any` escape, and
    the annotations have to admit the three states a chunk really has: a key
    that is absent, a key that is `null`, and a value.
    """
    text = ""
    arguments: dict[int, str] = {}
    total_tokens = 0
    for chunk in stream:
        for choice in chunk["choices"]:
            delta = choice["delta"]
            # `.get` twice over, and both are load-bearing: `content` may be
            # absent from a chunk that carries only a tool call, and `null` on
            # the one that opens a refusal.
            fragment = delta.get("content")
            if fragment is not None:
                text += fragment
            for call in delta.get("tool_calls") or []:
                function = call.get("function")
                if function is not None:
                    arguments[call["index"]] = arguments.get(
                        call["index"], ""
                    ) + (function.get("arguments") or "")
            # Required upstream and null until the choice ends, so it is always
            # readable and usually `None`.
            finish: str | None = choice["finish_reason"]
            assert finish is None or finish
        # Absent unless the caller asked for it, and null on every chunk but
        # the last -- which is why the annotation is `| None` and the key is
        # not required.
        usage = chunk.get("usage")
        if usage is not None:
            total_tokens = usage["total_tokens"]
    assert total_tokens >= 0 and arguments is not None
    return text


def the_stream_response_is_strict_when_requested(
    chunk: CreateChatCompletionStreamResponse,
) -> None:
    chunk["choises"]  # type: ignore[typeddict-item]
    chunk["choices"][0]["delta"]["contnet"]  # type: ignore[typeddict-item]


def a_nullable_field_stays_nullable(client: WarpLLM) -> None:
    """`TopLogprob.bytes` is `Option<Vec<u8>>` in Rust with no
    `skip_serializing_if`, so it really does arrive as `null`.

    It was `list[int]` here until the schema emitter started spelling nullables
    as `anyOf`; the annotation was flatly wrong and nothing said so.
    """
    completion = client.chat_completions({"model": "m", "messages": []})
    logprobs = completion["choices"][0].get("logprobs")
    assert logprobs is not None
    content = logprobs.get("content")
    assert content is not None
    first = content[0]["top_logprobs"][0]["bytes"]
    len(first)  # type: ignore[arg-type]


def the_streaming_overload_yields_a_stream(client: WarpLLM) -> None:
    """A dict literal whose keys the TypedDict declares selects streaming.

    No runtime test can prove this: a `chat_completions` that always returned
    the response type would still pass every assertion in `test_chat.py`,
    because the stream it actually returns is iterated dynamically.
    """
    for chunk in client.chat_completions({"stream": True}):
        chunk["choices"][0]["delta"]


def the_default_overload_yields_a_whole_reply(client: WarpLLM) -> None:
    """...and its absence selects the whole reply.

    The discriminator is `message` vs `delta`: a finished choice has the first
    and a chunk's has the second, so asking a whole reply for `delta` is the
    error that says the overloads did not collapse into one.
    """
    completion = client.chat_completions({"model": "m", "messages": []})
    completion["choices"][0]["message"]
    completion["choices"][0]["delta"]  # type: ignore[typeddict-item]


def extra_keys_fall_back_to_the_reply_overload(client: WarpLLM) -> None:
    """The documented limitation, pinned rather than left to be discovered.

    `stream: True` alongside keys the TypedDict does not declare matches no
    TypedDict, so a checker picks the non-streaming overload even though the
    RUNTIME returns a stream. `chat_completions_stream` is the way out, and the
    next probe is what says it works.
    """
    completion = client.chat_completions(
        {"model": "m", "messages": [], "stream": True}
    )
    completion["choices"][0]["message"]


def the_explicit_stream_entrypoint_is_precisely_typed(client: WarpLLM) -> None:
    """No such ambiguity here: the return type does not depend on a value."""
    for chunk in client.chat_completions_stream(
        {"model": "m", "messages": []}
    ):
        chunk["choices"][0]["delta"]
