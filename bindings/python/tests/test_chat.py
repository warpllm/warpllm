import pytest
from pytest_httpserver import HTTPServer
from warpllm import (
    AuthenticationError,
    ChatCompletion,
    InvalidRequestError,
    WarpLLM,
)

MESSAGES = [{"role": "user", "content": "hi"}]


def test_sync_openai_happy_path(
    client: WarpLLM, httpserver: HTTPServer, openai_completion_body
):
    httpserver.expect_request(
        "/chat/completions",
        method="POST",
        headers={"Authorization": "Bearer sk-test-openai"},
    ).respond_with_json(openai_completion_body)

    completion = client.chat.completions.create(
        model="openai/gpt-4o", messages=MESSAGES
    )

    assert isinstance(completion, ChatCompletion)
    assert completion.choices[0].message.content == "Hello there!"
    assert completion.choices[0].finish_reason == "stop"
    assert completion.model == "openai/gpt-4o"
    assert completion.usage.total_tokens == 21
    assert completion.service_tier == "default"
    assert completion.system_fingerprint == "fp_44709d6fcb"
    assert completion.usage.prompt_tokens_details.cached_tokens == 3
    assert completion.usage.prompt_tokens_details.cache_write_tokens == 2
    assert completion.usage.completion_tokens_details.reasoning_tokens == 5

    sent = httpserver.log[0][0].get_json()
    assert sent["model"] == "gpt-4o"  # provider prefix stripped outbound
    assert sent["messages"] == MESSAGES


async def test_async_openai_happy_path(
    async_client, httpserver: HTTPServer, openai_completion_body
):
    httpserver.expect_request(
        "/chat/completions",
        method="POST",
        headers={"Authorization": "Bearer sk-test-openai"},
    ).respond_with_json(openai_completion_body)

    completion = await async_client.chat.completions.create(
        model="openai/gpt-4o", messages=MESSAGES
    )

    assert completion.choices[0].message.content == "Hello there!"
    assert completion.model == "openai/gpt-4o"
    assert completion.usage.total_tokens == 21


def test_401_raises_authentication_error(
    client: WarpLLM, httpserver: HTTPServer
):
    httpserver.expect_request("/chat/completions").respond_with_json(
        {
            "error": {
                "message": "Incorrect API key provided",
                "type": "invalid_request_error",
            }
        },
        status=401,
    )

    with pytest.raises(AuthenticationError) as exc_info:
        client.chat.completions.create(
            model="openai/gpt-4o", messages=MESSAGES
        )
    assert exc_info.value.status_code == 401
    assert exc_info.value.provider == "openai"
    assert "Incorrect API key" in str(exc_info.value)


def test_unknown_provider_rejected(client: WarpLLM):
    with pytest.raises(InvalidRequestError, match="not a supported provider"):
        client.chat.completions.create(
            model="mistral/large", messages=MESSAGES
        )


def test_bare_model_rejected(client: WarpLLM):
    with pytest.raises(InvalidRequestError, match="not a supported provider"):
        client.chat.completions.create(model="gpt-4o", messages=MESSAGES)


def test_missing_key_names_env_var(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = WarpLLM()
    with pytest.raises(AuthenticationError, match="OPENAI_API_KEY"):
        client.chat.completions.create(
            model="openai/gpt-4o", messages=MESSAGES
        )


def test_stream_raises_not_implemented(client: WarpLLM):
    with pytest.raises(NotImplementedError, match="streaming"):
        client.chat.completions.create(
            model="openai/gpt-4o", messages=MESSAGES, stream=True
        )
