from __future__ import annotations

import json
from typing import (
    TYPE_CHECKING,
    AsyncIterator,
    Iterator,
    Literal,
    Mapping,
    TypedDict,
    cast,
    overload,
)

from warpllm._warpllm import ChatStream as _NativeChatStream
from warpllm._warpllm import Client as _NativeClient
from warpllm._warpllm import BalancedClient as _NativeBalancedClient
from warpllm._warpllm import WarpLLMNativeError

from ._exceptions import raise_from_wire

if TYPE_CHECKING:
    from .types import (
        CreateChatCompletionResponse,
        CreateChatCompletionStreamResponse,
    )


class ProviderOptions(TypedDict, total=False):
    """One declared provider. `{}` means: serve it, key from the environment.

    `api_key` is for callers holding keys somewhere this process's environment
    cannot reach -- a secret manager, a per-tenant record. It wins over the
    variable the provider's registry entry names.
    """

    api_key: str


class BalancedCandidate(TypedDict):
    """One candidate in a weighted round-robin group."""

    model: str
    weight: int


def _build_config(
    base_url: str | None,
    specs_path: str | None,
    timeout: int | None,
    stream_read_timeout: int | None,
    providers: Mapping[str, ProviderOptions] | None,
) -> str:
    """Build the `ClientConfig` JSON shared by all client constructors."""
    config = {
        "base_url": base_url,
        "specs_path": specs_path,
        "timeout_secs": timeout,
        "stream_read_timeout_secs": stream_read_timeout,
        # `dict()`, because the parameter is a `Mapping` and `json.dumps`
        # serializes only a real one -- the same materialization every request
        # on this boundary already does. `dict({})` is still `{}`, so the
        # distinction below survives it.
        "providers": None if providers is None else dict(providers),
    }
    # `is not None`, not truthiness: `providers={}` is a declaration of
    # none and has to survive, while `providers=None` is no declaration at
    # all and has to be dropped. Rust reads the difference.
    return json.dumps({k: v for k, v in config.items() if v is not None})


def _native_client(
    base_url: str | None,
    specs_path: str | None,
    timeout: int | None,
    stream_read_timeout: int | None,
    providers: Mapping[str, ProviderOptions] | None,
) -> _NativeClient:
    try:
        return _NativeClient(
            _build_config(base_url, specs_path, timeout, stream_read_timeout, providers)
        )
    except WarpLLMNativeError as e:
        raise_from_wire(str(e))


def _native_balanced_client(
    base_url: str | None,
    specs_path: str | None,
    timeout: int | None,
    stream_read_timeout: int | None,
    providers: Mapping[str, ProviderOptions] | None,
    candidates: list[BalancedCandidate],
) -> _NativeBalancedClient:
    for i, candidate in enumerate(candidates):
        if "model" not in candidate or "weight" not in candidate:
            raise TypeError(f"candidates[{i}] needs both 'model' and 'weight': {candidate!r}")
    try:
        return _NativeBalancedClient(
            _build_config(base_url, specs_path, timeout, stream_read_timeout, providers),
            json.dumps(candidates),
        )
    except WarpLLMNativeError as e:
        raise_from_wire(str(e))


# Either native client type exposes the same four methods -- this is the
# structural interface the four helpers below actually need, in place of
# `Any` and a runtime `assert hasattr` that `python -O` strips and that
# nothing but these two types could ever fail anyway. Mirrors the Node side's
# `NativeChatClient` (`client.ts`), which took this approach from the start.
_NativeChatClient = _NativeClient | _NativeBalancedClient


def _sync_chat_completions(native: _NativeChatClient, request_json: str) -> str:
    try:
        return native.chat_completions(request_json)
    except WarpLLMNativeError as e:
        raise_from_wire(str(e))


async def _async_chat_completions(native: _NativeChatClient, request_json: str) -> str:
    try:
        return await native.async_chat_completions(request_json)
    except WarpLLMNativeError as e:
        raise_from_wire(str(e))


def _sync_chat_completions_stream(
    native: _NativeChatClient, request_json: str
) -> _NativeChatStream:
    try:
        return native.chat_completions_stream(request_json)
    except WarpLLMNativeError as e:
        raise_from_wire(str(e))


async def _async_chat_completions_stream(
    native: _NativeChatClient, request_json: str
) -> _NativeChatStream:
    try:
        return await native.async_chat_completions_stream(request_json)
    except WarpLLMNativeError as e:
        raise_from_wire(str(e))


class _StreamingRequest(TypedDict):
    """A request that asks for chunks, for overload resolution only.

    Deliberately minimal. A checker matches a dict literal against a TypedDict
    only when its keys are a subset of the declared ones, and warpllm forwards
    fields it does not model -- so declaring more here would make ordinary
    requests stop matching. See `WarpLLM.chat_completions` on why this is
    best-effort.
    """

    stream: Literal[True]


class ChatCompletionStream(Iterator["CreateChatCompletionStreamResponse"]):
    """The chunks of one streamed reply.

    ```python
    stream = client.chat_completions({"model": ..., "messages": ..., "stream": True})
    for chunk in stream:
        print(chunk["choices"][0]["delta"].get("content", ""), end="")
    ```

    Errors raise the same classes a non-streamed call does. One arriving
    mid-iteration is terminal -- whatever produced it also ended the stream.
    """

    def __init__(self, native: _NativeChatStream) -> None:
        self._native = native

    def __iter__(self) -> ChatCompletionStream:
        return self

    def __next__(self) -> CreateChatCompletionStreamResponse:
        try:
            raw = self._native.next()
        except WarpLLMNativeError as e:
            raise_from_wire(str(e))
        # `None` is the stream ending. The `[DONE]` sentinel never reaches
        # here: it is framing, consumed by the transport in Rust.
        if raw is None:
            raise StopIteration
        return cast("CreateChatCompletionStreamResponse", json.loads(raw))


class AsyncChatCompletionStream(
    AsyncIterator["CreateChatCompletionStreamResponse"]
):
    """`ChatCompletionStream` for `async for`."""

    def __init__(self, native: _NativeChatStream) -> None:
        self._native = native

    def __aiter__(self) -> AsyncChatCompletionStream:
        return self

    async def __anext__(self) -> CreateChatCompletionStreamResponse:
        try:
            raw = await self._native.async_next()
        except WarpLLMNativeError as e:
            raise_from_wire(str(e))
        if raw is None:
            raise StopAsyncIteration
        return cast("CreateChatCompletionStreamResponse", json.loads(raw))


class WarpLLM:
    """Synchronous client. Model strings are `provider/model`, e.g.
    `"openai/gpt-5.6"`. API keys come from the environment
    (`OPENAI_API_KEY`); a provider's key is only required when a request
    targets that provider.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        specs_path: str | None = None,
        timeout: int | None = None,
        stream_read_timeout: int | None = None,
        providers: Mapping[str, ProviderOptions] | None = None,
    ) -> None:
        """`specs_path` points at a roster of your own, in the same schema as
        warpllm's built-in `specs.yaml`, merged over it. It is how a
        self-hosted OpenAI-compatible server -- vLLM, TGI, Ollama, llama.cpp --
        becomes routable, and no key is required:

        ```yaml
        providers:
          local:
            base_url: "http://localhost:8000/v1"
            auth: none
            models:
              local/llama-3.3-70b:
                supported_apis:
                  - {api: openai_compat_chat_completions}
        ```

        The built-in providers survive the merge, so adding `local/` leaves
        `openai/` exactly where it was. Reusing a built-in provider's name
        replaces that provider whole, models included, and does so SILENTLY
        here: warpllm warns over Rust's `tracing`, which this binding installs
        no subscriber for. The file is read HERE, when the client is built, so
        a roster that cannot be used raises now rather than failing a request
        later. Unset falls back to the `WARPLLM_SPECS` environment variable,
        and then to the built-in roster alone.

        `stream_read_timeout` bounds how long a stream may go without a
        single byte; unset means never.

        `timeout` is a total deadline and cannot tell a slow stream from a
        wedged one. This bounds the GAP instead and resets on every byte. Set
        it above the slowest time-to-first-token you expect, not merely above
        the gap between chunks -- the wait before the first chunk is a gap too.

        `providers` narrows this client to the providers it names, keyed by
        registry name: `{"openai": {}, "deepseek": {"api_key": "sk-..."}}`.
        OMITTING it -- not passing an empty dict -- serves warpllm's whole
        roster, which is what every client did before this argument existed.
        Declaring narrows what is READ as well as what is routed: only the
        named providers' environment variables are consulted, and a request
        for a model under a provider not listed raises before reaching any
        upstream. A name the registry does not hold raises here, not later.
        """
        self._native = _native_client(
            base_url, specs_path, timeout, stream_read_timeout, providers
        )

    # Signatures 1 and 2 overlap on purpose: a `_StreamingRequest` IS a
    # `Mapping[str, object]`, so mypy warns that a value could match both. That
    # is the whole design -- the streaming overload is strictly more specific
    # and listed first, which is the order resolution reads. `--warn-unused-
    # ignores` keeps this honest if the overlap ever stops happening.
    @overload
    def chat_completions(  # type: ignore[overload-overlap]
        self, request: _StreamingRequest
    ) -> ChatCompletionStream: ...

    @overload
    def chat_completions(
        self, request: Mapping[str, object]
    ) -> CreateChatCompletionResponse: ...

    def chat_completions(
        self, request: Mapping[str, object]
    ) -> CreateChatCompletionResponse | ChatCompletionStream:
        """One method, mirroring Rust's `client.chat_completions(request)`.

        The request crosses verbatim -- its fields are Rust's, so nothing
        here renames them and nothing here has to learn a field warpllm
        gains. The response comes back as Rust serialized it: Rust has
        already parsed and validated it, and re-hydrating it into Python
        objects would re-do that work to hand back the same fields under the
        same names.

        `warpllm.types.CreateChatCompletionRequest` is available when callers
        want strict authoring help. This boundary accepts any mapping because
        Rust deliberately forwards fields it does not model.

        `stream=True` returns a `ChatCompletionStream` to iterate, matching the
        official OpenAI SDK's one-method shape.

        The RUNTIME behaviour of that is exact; the static typing is
        best-effort, and the reason is this signature. The OpenAI SDK overloads
        on a `stream` KEYWORD argument, which a checker reads as a literal.
        warpllm takes one mapping so that unmodeled fields cross untouched, so
        the overload has to match the mapping's type instead -- and a dict
        literal carrying extra keys matches no TypedDict. A checker will infer
        `CreateChatCompletionResponse` for those; use `chat_completions_stream`
        where the annotation has to be right.
        """
        # Dispatch on the VALUE, so behaviour is correct wherever the overload
        # above cannot resolve.
        if request.get("stream") is True:
            return self.chat_completions_stream(request)
        raw = _sync_chat_completions(self._native, json.dumps(dict(request)))
        return cast("CreateChatCompletionResponse", json.loads(raw))

    def chat_completions_stream(
        self, request: Mapping[str, object]
    ) -> ChatCompletionStream:
        """Streaming, precisely typed.

        The same thing `chat_completions({..., "stream": True})` does, for
        callers who need a checker to agree with it. `stream` is set here, so
        the request need not say so.
        """
        native = _sync_chat_completions_stream(
            self._native, json.dumps({**dict(request), "stream": True})
        )
        return ChatCompletionStream(native)


class AsyncWarpLLM:
    """Async client; `await client.chat_completions(...)`."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        specs_path: str | None = None,
        timeout: int | None = None,
        stream_read_timeout: int | None = None,
        providers: Mapping[str, ProviderOptions] | None = None,
    ) -> None:
        """The same arguments as `WarpLLM.__init__`, which documents them --
        including `specs_path`, the roster file that makes a self-hosted
        server routable, and `providers`, which narrows this client to the
        providers it names.
        """
        self._native = _native_client(
            base_url, specs_path, timeout, stream_read_timeout, providers
        )

    # Overlapping on purpose; see `WarpLLM.chat_completions`.
    @overload
    async def chat_completions(  # type: ignore[overload-overlap]
        self, request: _StreamingRequest
    ) -> AsyncChatCompletionStream: ...

    @overload
    async def chat_completions(
        self, request: Mapping[str, object]
    ) -> CreateChatCompletionResponse: ...

    async def chat_completions(
        self, request: Mapping[str, object]
    ) -> CreateChatCompletionResponse | AsyncChatCompletionStream:
        """See `WarpLLM.chat_completions`, including what its overloads can and
        cannot promise. `stream=True` returns an `AsyncChatCompletionStream`.
        """
        if request.get("stream") is True:
            return await self.chat_completions_stream(request)
        raw = await _async_chat_completions(self._native, json.dumps(dict(request)))
        return cast("CreateChatCompletionResponse", json.loads(raw))

    async def chat_completions_stream(
        self, request: Mapping[str, object]
    ) -> AsyncChatCompletionStream:
        """Streaming, precisely typed. See `WarpLLM.chat_completions_stream`.

        Awaits its own native method rather than the sync client's: opening a
        stream is a request whose headers a provider may sit on, and blocking
        on that inside a coroutine stops the whole event loop until it answers.
        """
        native = await _async_chat_completions_stream(
            self._native, json.dumps({**dict(request), "stream": True})
        )
        return AsyncChatCompletionStream(native)


class WarpLLMBalanced:
    """Synchronous load-balanced client. Distributes requests across
    candidates via weighted round-robin. Model strings in requests are
    rewritten to the selected candidate on each call.

    ```python
    client = WarpLLMBalanced(
        candidates=[
            {"model": "openai/gpt-5.6", "weight": 3},
            {"model": "deepseek/deepseek-v4-pro", "weight": 1},
        ],
        providers={"openai": {}, "deepseek": {}},
    )
    ```
    """

    def __init__(
        self,
        *,
        candidates: list[BalancedCandidate],
        base_url: str | None = None,
        specs_path: str | None = None,
        timeout: int | None = None,
        stream_read_timeout: int | None = None,
        providers: Mapping[str, ProviderOptions] | None = None,
    ) -> None:
        self._native = _native_balanced_client(
            base_url, specs_path, timeout, stream_read_timeout, providers, candidates
        )

    @overload
    def chat_completions(  # type: ignore[overload-overlap]
        self, request: _StreamingRequest
    ) -> ChatCompletionStream: ...

    @overload
    def chat_completions(
        self, request: Mapping[str, object]
    ) -> CreateChatCompletionResponse: ...

    def chat_completions(
        self, request: Mapping[str, object]
    ) -> CreateChatCompletionResponse | ChatCompletionStream:
        """One method, mirroring `WarpLLM.chat_completions`. The request's
        `model` field is overwritten with the selected candidate before each
        call."""
        if request.get("stream") is True:
            return self.chat_completions_stream(request)
        raw = _sync_chat_completions(self._native, json.dumps(dict(request)))
        return cast("CreateChatCompletionResponse", json.loads(raw))

    def chat_completions_stream(
        self, request: Mapping[str, object]
    ) -> ChatCompletionStream:
        """Streaming, precisely typed. `stream` is set here."""
        native = _sync_chat_completions_stream(
            self._native, json.dumps({**dict(request), "stream": True})
        )
        return ChatCompletionStream(native)


class AsyncWarpLLMBalanced:
    """Async load-balanced client; `await client.chat_completions(...)`."""

    def __init__(
        self,
        *,
        candidates: list[BalancedCandidate],
        base_url: str | None = None,
        specs_path: str | None = None,
        timeout: int | None = None,
        stream_read_timeout: int | None = None,
        providers: Mapping[str, ProviderOptions] | None = None,
    ) -> None:
        self._native = _native_balanced_client(
            base_url, specs_path, timeout, stream_read_timeout, providers, candidates
        )

    @overload
    async def chat_completions(  # type: ignore[overload-overlap]
        self, request: _StreamingRequest
    ) -> AsyncChatCompletionStream: ...

    @overload
    async def chat_completions(
        self, request: Mapping[str, object]
    ) -> CreateChatCompletionResponse: ...

    async def chat_completions(
        self, request: Mapping[str, object]
    ) -> CreateChatCompletionResponse | AsyncChatCompletionStream:
        """See `WarpLLMBalanced.chat_completions`. `stream=True` returns an
        `AsyncChatCompletionStream`."""
        if request.get("stream") is True:
            return await self.chat_completions_stream(request)
        raw = await _async_chat_completions(self._native, json.dumps(dict(request)))
        return cast("CreateChatCompletionResponse", json.loads(raw))

    async def chat_completions_stream(
        self, request: Mapping[str, object]
    ) -> AsyncChatCompletionStream:
        """Streaming, precisely typed."""
        native = await _async_chat_completions_stream(
            self._native, json.dumps({**dict(request), "stream": True})
        )
        return AsyncChatCompletionStream(native)
