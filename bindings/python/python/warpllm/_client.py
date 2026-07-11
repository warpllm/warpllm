from __future__ import annotations

import json
from typing import Any

from warpllm._warpllm import Client as _NativeClient
from warpllm._warpllm import WarpLLMNativeError

from ._exceptions import raise_from_wire
from ._models import ChatCompletion


def _build_config(
    openai_api_key: str | None,
    base_url: str | None,
    timeout: int | None,
) -> str:
    config = {
        "openai_api_key": openai_api_key,
        "base_url": base_url,
        "timeout_secs": timeout,
    }
    return json.dumps({k: v for k, v in config.items() if v is not None})


def _build_request(
    model: str,
    messages: list[dict[str, str]],
    temperature: float | None,
    max_tokens: int | None,
    top_p: float | None,
    stop: list[str] | None,
    stream: bool,
) -> str:
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stop": stop,
        "stream": stream or None,
    }
    return json.dumps({k: v for k, v in request.items() if v is not None})


def _native_client(
    openai_api_key: str | None,
    base_url: str | None,
    timeout: int | None,
) -> _NativeClient:
    try:
        return _NativeClient(
            _build_config(openai_api_key, base_url, timeout)
        )
    except WarpLLMNativeError as e:
        raise_from_wire(str(e))


class _Completions:
    def __init__(self, native: _NativeClient) -> None:
        self._native = native

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        stream: bool = False,
    ) -> ChatCompletion:
        request = _build_request(
            model, messages, temperature, max_tokens, top_p, stop, stream
        )
        try:
            raw = self._native.chat_completion(request)
        except WarpLLMNativeError as e:
            raise_from_wire(str(e))
        return ChatCompletion.from_dict(json.loads(raw))


class _AsyncCompletions:
    def __init__(self, native: _NativeClient) -> None:
        self._native = native

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        stream: bool = False,
    ) -> ChatCompletion:
        request = _build_request(
            model, messages, temperature, max_tokens, top_p, stop, stream
        )
        try:
            raw = await self._native.async_chat_completion(request)
        except WarpLLMNativeError as e:
            raise_from_wire(str(e))
        return ChatCompletion.from_dict(json.loads(raw))


class _Chat:
    def __init__(self, native: _NativeClient) -> None:
        self.completions = _Completions(native)


class _AsyncChat:
    def __init__(self, native: _NativeClient) -> None:
        self.completions = _AsyncCompletions(native)


class WarpLLM:
    """Synchronous client. Model strings are `provider/model`, e.g.
    `"openai/gpt-4o"`. API keys fall back to `OPENAI_API_KEY`; a provider's
    key is only required when a request targets that provider.
    """

    def __init__(
        self,
        *,
        openai_api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.chat = _Chat(
            _native_client(openai_api_key, base_url, timeout)
        )


class AsyncWarpLLM:
    """Async client; `await client.chat.completions.create(...)`."""

    def __init__(
        self,
        *,
        openai_api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.chat = _AsyncChat(
            _native_client(openai_api_key, base_url, timeout)
        )
