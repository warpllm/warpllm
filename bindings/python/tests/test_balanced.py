"""`WarpLLMBalanced` / `AsyncWarpLLMBalanced`: weighted selection, model
rewriting, and construction-time validation, crossing the FFI boundary.

Previously untested from either binding -- see PR #79's review. The bugs
that review found (weight 0, weight overflow, a candidate missing `weight`)
are all constructor-time and need no network; the happy-path tests below are
the ones that do, confirming the JSON boundary actually wires candidate
selection through to a real (mocked) request.
"""

from collections import Counter

import pytest
from pytest_httpserver import HTTPServer
from warpllm import AsyncWarpLLMBalanced, BadRequestError, WarpLLMBalanced

MESSAGES = [{"role": "user", "content": "hi"}]


def _completion_body(model: str) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
    }


def test_all_zero_weight_is_rejected(base_url: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    with pytest.raises(BadRequestError, match="weight 0"):
        WarpLLMBalanced(
            candidates=[{"model": "openai/gpt-5.6", "weight": 0}],
            base_url=base_url,
        )


def test_a_weight_exceeding_i32_max_is_rejected(
    base_url: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    with pytest.raises(BadRequestError, match="exceeds the maximum"):
        WarpLLMBalanced(
            candidates=[{"model": "openai/gpt-5.6", "weight": 4_294_967_295}],
            base_url=base_url,
        )


def test_a_candidate_missing_weight_raises_before_reaching_rust(
    base_url: str, monkeypatch: pytest.MonkeyPatch
):
    """A `TypeError` naming the candidate, not a Rust JSON-offset message
    about a document the caller never wrote."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    with pytest.raises(TypeError, match="candidates\\[0\\]"):
        WarpLLMBalanced(
            candidates=[{"model": "openai/gpt-5.6"}],  # type: ignore[typeddict-item]
            base_url=base_url,
        )


def test_an_unknown_candidate_model_is_rejected(
    base_url: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    with pytest.raises(BadRequestError, match="no registered model"):
        WarpLLMBalanced(
            candidates=[{"model": "openai/not-a-model", "weight": 1}],
            base_url=base_url,
        )


def test_the_request_model_is_rewritten_to_the_selected_candidate(
    base_url: str, httpserver: HTTPServer, monkeypatch: pytest.MonkeyPatch
):
    """The core contract: a caller's `model` (the group name) never reaches
    the wire -- the selected candidate's does."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    httpserver.expect_request("/chat/completions").respond_with_json(
        _completion_body("gpt-5.6")
    )
    client = WarpLLMBalanced(
        candidates=[{"model": "openai/gpt-5.6", "weight": 1}],
        base_url=base_url,
        timeout=5,
    )
    completion = client.chat_completions({"model": "my-group", "messages": MESSAGES})
    assert completion["model"] == "openai/gpt-5.6"
    sent = httpserver.log[0][0].get_json()
    assert sent["model"] == "gpt-5.6"


def test_weighted_distribution_across_two_candidates(
    base_url: str, httpserver: HTTPServer, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    httpserver.expect_request("/chat/completions").respond_with_json(
        _completion_body("ignored")
    )
    client = WarpLLMBalanced(
        candidates=[
            {"model": "openai/gpt-5.6", "weight": 3},
            {"model": "deepseek/deepseek-v4-flash", "weight": 1},
        ],
        base_url=base_url,
        timeout=5,
    )
    for _ in range(4):
        client.chat_completions({"model": "my-group", "messages": MESSAGES})

    sent_models = Counter(
        entry[0].get_json()["model"] for entry in httpserver.log
    )
    # Exact distribution over one full cycle of weights [3, 1].
    assert sent_models["gpt-5.6"] == 3
    assert sent_models["deepseek-v4-flash"] == 1


async def test_async_balanced_happy_path(
    base_url: str, httpserver: HTTPServer, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    httpserver.expect_request("/chat/completions").respond_with_json(
        _completion_body("gpt-5.6")
    )
    client = AsyncWarpLLMBalanced(
        candidates=[{"model": "openai/gpt-5.6", "weight": 1}],
        base_url=base_url,
        timeout=5,
    )
    completion = await client.chat_completions(
        {"model": "my-group", "messages": MESSAGES}
    )
    assert completion["model"] == "openai/gpt-5.6"
