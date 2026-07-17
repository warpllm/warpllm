import pytest
from pytest_httpserver import HTTPServer
from warpllm import AsyncWarpLLM, WarpLLM


@pytest.fixture
def base_url(httpserver: HTTPServer) -> str:
    return httpserver.url_for("")


@pytest.fixture
def client(base_url: str, monkeypatch: pytest.MonkeyPatch) -> WarpLLM:
    # The native client reads OPENAI_API_KEY at construction, so set it first.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    return WarpLLM(base_url=base_url, timeout=5)


@pytest.fixture
def async_client(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> AsyncWarpLLM:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    return AsyncWarpLLM(base_url=base_url, timeout=5)


@pytest.fixture
def openai_completion_body() -> dict:
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "gpt-4o-2024-08-06",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello there!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21,
            "prompt_tokens_details": {
                "cached_tokens": 3,
                "cache_write_tokens": 2,
                "audio_tokens": 0,
            },
            "completion_tokens_details": {
                "reasoning_tokens": 5,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
        "service_tier": "default",
        "system_fingerprint": "fp_44709d6fcb",
    }
