import asyncio
import time
from types import MappingProxyType

import pytest
from pytest_httpserver import HTTPServer
from warpllm import (
    APIError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
    WarpLLM,
)
from werkzeug.wrappers import Response

MESSAGES = [{"role": "user", "content": "hi"}]


def request(model: str = "openai/gpt-5.6", **extra) -> dict:
    return {"model": model, "messages": MESSAGES, **extra}


def test_sync_openai_happy_path(
    client: WarpLLM, httpserver: HTTPServer, openai_completion_body
):
    httpserver.expect_request(
        "/chat/completions",
        method="POST",
        headers={"Authorization": "Bearer sk-test-openai"},
    ).respond_with_json(openai_completion_body)

    completion = client.chat_completions(request())

    assert completion["choices"][0]["message"]["content"] == "Hello there!"
    assert completion["choices"][0]["finish_reason"] == "stop"
    assert completion["model"] == "openai/gpt-5.6"
    assert completion["usage"]["total_tokens"] == 21
    assert completion["service_tier"] == "default"
    assert completion["system_fingerprint"] == "fp_44709d6fcb"
    assert completion["usage"]["prompt_tokens_details"]["cached_tokens"] == 3
    assert (
        completion["usage"]["prompt_tokens_details"]["cache_write_tokens"] == 2
    )
    assert (
        completion["usage"]["completion_tokens_details"]["reasoning_tokens"]
        == 5
    )

    sent = httpserver.log[0][0].get_json()
    assert sent["model"] == "gpt-5.6"  # provider prefix stripped outbound
    assert sent["messages"] == MESSAGES


async def test_async_openai_happy_path(
    async_client, httpserver: HTTPServer, openai_completion_body
):
    httpserver.expect_request(
        "/chat/completions",
        method="POST",
        headers={"Authorization": "Bearer sk-test-openai"},
    ).respond_with_json(openai_completion_body)

    completion = await async_client.chat_completions(request())

    assert completion["choices"][0]["message"]["content"] == "Hello there!"
    assert completion["model"] == "openai/gpt-5.6"
    assert completion["usage"]["total_tokens"] == 21


def test_the_response_is_not_narrowed_on_the_way_through(
    client: WarpLLM, httpserver: HTTPServer, openai_completion_body
):
    """What Rust serialized is what the caller gets.

    The wrapper used to re-hydrate the body into dataclasses, which meant a
    field warpllm learned to pass through was still dropped here until a
    Python class gained it too. Handing back the parsed body makes that
    class of bug unreachable rather than merely tested for.
    """
    openai_completion_body["some_field_python_never_heard_of"] = {
        "nested": [1]
    }
    httpserver.expect_request("/chat/completions").respond_with_json(
        openai_completion_body
    )

    completion = client.chat_completions(request())

    assert completion["some_field_python_never_heard_of"] == {"nested": [1]}


def test_an_unmodeled_request_field_reaches_rust_and_the_provider(
    client: WarpLLM, httpserver: HTTPServer, openai_completion_body
):
    httpserver.expect_request("/chat/completions").respond_with_json(
        openai_completion_body
    )

    client.chat_completions(
        request(seed=7, future_parameter={"enabled": True})
    )

    sent = httpserver.log[0][0].get_json()
    assert sent["seed"] == 7
    assert sent["future_parameter"] == {"enabled": True}


def test_401_reports_authentication(client: WarpLLM, httpserver: HTTPServer):
    httpserver.expect_request("/chat/completions").respond_with_json(
        {
            "error": {
                "message": "Incorrect API key provided",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }
        },
        status=401,
    )

    with pytest.raises(AuthenticationError) as exc_info:
        client.chat_completions(request())
    # Every failure is an APIError, so one `except` catches the lot.
    assert isinstance(exc_info.value, APIError)
    assert exc_info.value.status_code == 401
    assert "Incorrect API key" in str(exc_info.value)
    # The provider's own slug reaches the caller, not warpllm's spelling
    # of it -- warpllm would have called this one `authentication`.
    assert exc_info.value.code == "invalid_api_key"
    assert exc_info.value.type == "invalid_request_error"


def test_quota_exhaustion_is_not_reported_as_a_rate_limit(
    client: WarpLLM, httpserver: HTTPServer
):
    """A quota exhaustion arrives as a 429 and reads exactly like a rate
    limit, but no amount of backing off buys credit.

    A retry loop keyed on `code == "rate_limited"` must not fire here --
    that is how a billing failure becomes an infinite retry loop.
    """
    httpserver.expect_request("/chat/completions").respond_with_json(
        {
            "error": {
                "message": "You exceeded your current quota",
                "type": "invalid_request_error",
                "code": "insufficient_quota",
            }
        },
        status=429,
    )

    # OpenAI reports both under one class, so the class cannot tell them
    # apart and `code` is the only thing that can.
    with pytest.raises(RateLimitError) as exc_info:
        client.chat_completions(request())
    error = exc_info.value
    assert error.code == "insufficient_quota"
    assert (
        error.code != "rate_limit_exceeded"
    ), "a backoff loop would swallow this"
    assert error.status_code == 429


def test_rate_limit_carries_the_providers_request_id(
    client: WarpLLM, httpserver: HTTPServer
):
    """The upstream's request id reaches the caller. It lives only in a
    header, so it proves the transport kept it."""
    httpserver.expect_request("/chat/completions").respond_with_json(
        {
            "error": {
                "message": "Rate limit reached",
                "type": "rate_limit_error",
            }
        },
        status=429,
        headers={"Retry-After": "30", "x-request-id": "req-abc"},
    )

    with pytest.raises(RateLimitError) as exc_info:
        client.chat_completions(request())
    assert exc_info.value.type == "rate_limit_error"
    assert exc_info.value.request_id == "req-abc"
    assert exc_info.value.headers["retry-after"] == "30"


def test_context_overflow_is_classified(
    client: WarpLLM, httpserver: HTTPServer
):
    """A context overflow must not read as a plain bad request: the remedy
    is a shorter prompt or a bigger model, not a corrected payload."""
    httpserver.expect_request("/chat/completions").respond_with_json(
        {
            "error": {
                "message": "maximum context length is 8192 tokens",
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }
        },
        status=400,
    )

    with pytest.raises(BadRequestError) as exc_info:
        client.chat_completions(request())
    assert exc_info.value.code == "context_length_exceeded"


def test_code_separates_the_providers_rejection_from_warpllms(
    client: WarpLLM, httpserver: HTTPServer
):
    """A provider rejecting the request and warpllm rejecting it read
    almost alike -- both 400, both `invalid_request_error` -- and the
    remedy is not the same: one edits the payload, the other may just need
    a different model. `code` is what tells them apart, since `origin` is
    warpllm's own vocabulary and stays in Rust."""
    httpserver.expect_request("/chat/completions").respond_with_json(
        {"error": {"message": "bad payload", "type": "invalid_request_error"}},
        status=400,
    )

    with pytest.raises(BadRequestError) as upstream:
        client.chat_completions(request())

    # ...and warpllm's own rejection never left the process.
    with pytest.raises(BadRequestError) as local:
        client.chat_completions(request(model="mistral/large"))

    assert upstream.value.type == local.value.type == "invalid_request_error"
    # The provider named no code, and warpllm does not invent one for it.
    assert upstream.value.code is None
    assert local.value.code == "invalid_request"


def test_unknown_provider_rejected(client: WarpLLM):
    with pytest.raises(BadRequestError, match="no registered model spec"):
        client.chat_completions(request(model="mistral/large"))


def test_bare_model_rejected(client: WarpLLM):
    with pytest.raises(BadRequestError, match="no registered model spec"):
        client.chat_completions(request(model="gpt-5.6"))


def test_missing_key_names_env_var(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = WarpLLM()
    with pytest.raises(
        AuthenticationError, match="OPENAI_API_KEY"
    ) as exc_info:
        client.chat_completions(request())
    assert exc_info.value.code == "invalid_api_key"


# The chunks a provider sends for "Hello there!", mirroring
# fixtures/transcript/openai-text.sse. Written out rather than read from the
# shared corpus because these assert the WRAPPER, not the shapes.
_ENVELOPE = (
    '"id":"chatcmpl-1","object":"chat.completion.chunk",'
    '"created":1700000000,"model":"gpt-5.6"'
)

OPENAI_STREAM = (
    "data: {" + _ENVELOPE + ',"choices":[{"index":0,"delta":'
    '{"role":"assistant","content":"Hello"},"logprobs":null,'
    '"finish_reason":null}],"usage":null,"obfuscation":"KtQ3nZ8w"}\n\n'
    ": keepalive\n\n"
    "data: {" + _ENVELOPE + ',"choices":[{"index":0,"delta":'
    '{"content":" there!"},"logprobs":null,"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


def _serve_stream(httpserver: HTTPServer, body: str = OPENAI_STREAM) -> None:
    httpserver.expect_request("/chat/completions").respond_with_data(
        body, content_type="text/event-stream"
    )


def test_stream_is_iterated_with_for(client: WarpLLM, httpserver: HTTPServer):
    _serve_stream(httpserver)

    chunks = list(client.chat_completions(request(stream=True)))

    # The `[DONE]` sentinel is the stream ending, never a chunk.
    assert len(chunks) == 2
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "Hello there!"
    # Every chunk echoes the caller's prefixed string, not the upstream name.
    assert all(c["model"] == "openai/gpt-5.6" for c in chunks)
    assert chunks[1]["choices"][0]["finish_reason"] == "stop"


async def test_async_stream_is_iterated_with_async_for(
    async_client, httpserver: HTTPServer
):
    _serve_stream(httpserver)

    chunks = [
        chunk
        async for chunk in await async_client.chat_completions(
            request(stream=True)
        )
    ]

    assert len(chunks) == 2
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "Hello there!"


async def test_opening_an_async_stream_leaves_the_event_loop_running(
    async_client, httpserver: HTTPServer
):
    """Opening a stream must not stop everything else the loop is doing.

    The native call is one POST, but a provider can sit on its response headers
    for as long as it likes -- and blocking on that from inside a coroutine
    holds the loop's thread, so every other task, timer and callback stops
    until it answers. Releasing the GIL does not help: it is the LOOP that is
    blocked, not the interpreter.

    So this asserts on a SECOND task making progress, which is the only thing
    that tells a non-blocking open from a fast one.
    """

    def slow_headers(_request):
        time.sleep(0.3)
        return Response(OPENAI_STREAM, content_type="text/event-stream")

    httpserver.expect_request("/chat/completions").respond_with_handler(
        slow_headers
    )

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.ensure_future(ticker())
    try:
        stream = await async_client.chat_completions_stream(request())
        chunks = [chunk async for chunk in stream]
    finally:
        task.cancel()

    assert len(chunks) == 2
    assert ticks > 5, f"the loop was blocked while the stream opened ({ticks})"


def test_chat_completions_stream_needs_no_stream_flag(
    client: WarpLLM, httpserver: HTTPServer
):
    """The precisely typed entrypoint sets `stream` itself."""
    _serve_stream(httpserver)

    chunks = list(client.chat_completions_stream(request()))

    assert len(chunks) == 2
    assert httpserver.log[0][0].get_json()["stream"] is True


def test_stream_read_timeout_reaches_the_native_config(
    base_url: str, httpserver: HTTPServer, monkeypatch: pytest.MonkeyPatch
):
    """The Rust config sets `deny_unknown_fields`, so a key misspelled on the
    way across fails at CONSTRUCTION rather than being quietly ignored -- which
    makes this a real check that the option arrives.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    _serve_stream(httpserver)
    bounded = WarpLLM(base_url=base_url, timeout=5, stream_read_timeout=30)

    assert len(list(bounded.chat_completions_stream(request()))) == 2


def test_a_refusal_before_the_stream_opens_raises(
    client: WarpLLM, httpserver: HTTPServer
):
    httpserver.expect_request("/chat/completions").respond_with_json(
        {
            "error": {
                "message": "Rate limit reached",
                "type": "rate_limit_exceeded",
            }
        },
        status=429,
    )

    with pytest.raises(RateLimitError):
        client.chat_completions(request(stream=True))


def test_a_stream_that_stops_before_its_sentinel_raises(
    client: WarpLLM, httpserver: HTTPServer
):
    """A dropped upstream must not end iteration like a finished one.

    `StopIteration` is how a complete answer ends, so raising it here would
    hand back a truncated reply with nothing to distinguish it -- the caller
    would print half a sentence and never learn why.
    """
    _serve_stream(httpserver, OPENAI_STREAM.replace("data: [DONE]\n\n", ""))

    chunks = []
    stream = client.chat_completions(request(stream=True))
    with pytest.raises(APIError, match="before it was complete"):
        for chunk in stream:
            chunks.append(chunk)

    assert len(chunks) == 2, "everything that did arrive is still the caller's"


def test_an_undecodable_event_ends_the_stream_as_a_typed_error(
    client: WarpLLM, httpserver: HTTPServer
):
    _serve_stream(httpserver, "data: not json\n\n")

    stream = client.chat_completions(request(stream=True))
    with pytest.raises(APIError):
        list(stream)


def test_declared_providers_narrow_what_this_client_routes(
    base_url: str, httpserver: HTTPServer, monkeypatch: pytest.MonkeyPatch
):
    """A model under an undeclared provider is refused before any request
    goes out -- with a 400 about the configuration, not the 401 a missing
    credential would give. The deepseek key is set precisely so that a client
    checking credentials first would answer the wrong question.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    client = WarpLLM(base_url=base_url, timeout=5, providers={"openai": {}})

    with pytest.raises(BadRequestError) as exc_info:
        client.chat_completions(request(model="deepseek/deepseek-v4-flash"))

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == "provider_not_declared"
    assert httpserver.log == [], "nothing should have reached an upstream"


def test_an_inline_key_reaches_the_upstream_request(
    base_url: str, httpserver: HTTPServer, monkeypatch: pytest.MonkeyPatch,
    openai_completion_body: dict,
):
    """The point of carrying a key in the config: it authenticates a provider
    whose environment variable is not set at all.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    httpserver.expect_request("/chat/completions").respond_with_json(
        openai_completion_body
    )
    client = WarpLLM(
        base_url=base_url,
        timeout=5,
        providers={"openai": {"api_key": "sk-from-the-config"}},
    )

    client.chat_completions(request())

    sent = httpserver.log[0][0]
    assert sent.headers["Authorization"] == "Bearer sk-from-the-config"


def test_an_unknown_declared_provider_raises_at_construction(
    base_url: str, monkeypatch: pytest.MonkeyPatch
):
    """A misspelling is wrong where it is written, not at the request that
    happened to route there -- and the message hands back the roster.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")

    with pytest.raises(BadRequestError, match="openia"):
        WarpLLM(base_url=base_url, timeout=5, providers={"openia": {}})


def test_providers_accepts_any_mapping_not_only_a_dict(
    base_url: str, monkeypatch: pytest.MonkeyPatch
):
    """The parameter is typed `Mapping`, so a read-only one has to work.

    `json.dumps` serializes only a real dict, which is why every request on
    this boundary is materialized with `dict()` before it crosses. The config
    needs the same materialization, or a valid argument raises a raw
    `TypeError` before reaching Rust.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")

    client = WarpLLM(
        base_url=base_url,
        timeout=5,
        providers=MappingProxyType({"openai": {}}),
    )

    # And it really narrowed, rather than being dropped on the way across.
    with pytest.raises(BadRequestError, match="not declared"):
        client.chat_completions(request(model="deepseek/deepseek-v4-flash"))
