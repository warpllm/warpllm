"""ChatCompletion.from_dict deep-parse coverage: every documented optional
object of the chat.completion response must hydrate into its typed class."""

from warpllm import (
    ChatCompletion,
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageFunctionToolCall,
    Error,
    ModerationResults,
)

FULL_BODY = {
    "id": "chatcmpl-123",
    "choices": [
        {
            "finish_reason": "tool_calls",
            "index": 0,
            "logprobs": {
                "content": [
                    {
                        "token": "Hi",
                        "bytes": [72, 105],
                        "logprob": -0.1,
                        "top_logprobs": [
                            {"token": "Hi", "bytes": None, "logprob": -0.1}
                        ],
                    }
                ],
                "refusal": [],
            },
            "message": {
                "content": "Hello there!",
                "refusal": None,
                "role": "assistant",
                "annotations": [
                    {
                        "type": "url_citation",
                        "url_citation": {
                            "end_index": 5,
                            "start_index": 0,
                            "title": "Example",
                            "url": "https://example.com",
                        },
                    }
                ],
                "audio": {
                    "id": "audio-1",
                    "data": "aGk=",
                    "expires_at": 1_700_000_600,
                    "transcript": "hi",
                },
                "function_call": {"arguments": "{}", "name": "legacy_fn"},
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"arguments": '{"q":1}', "name": "search"},
                    },
                    {
                        "id": "call-2",
                        "type": "custom",
                        "custom": {"input": "raw text", "name": "my_tool"},
                    },
                ],
            },
        }
    ],
    "created": 1_700_000_000,
    "model": "gpt-4o",
    "object": "chat.completion",
    "moderation": {
        "input": {
            "type": "moderation_results",
            "model": "omni-moderation-latest",
            "results": [
                {
                    "categories": {"violence": False},
                    "category_applied_input_types": {"violence": ["text"]},
                    "category_scores": {"violence": 0.001},
                    "flagged": False,
                    "model": "omni-moderation-latest",
                    "type": "moderation_result",
                }
            ],
        },
        "output": {
            "type": "error",
            "code": "moderation_unavailable",
            "message": "try again",
        },
    },
    "service_tier": "default",
    "system_fingerprint": "fp_44709d6fcb",
    "usage": {
        "completion_tokens": 12,
        "prompt_tokens": 9,
        "total_tokens": 21,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 5,
            "rejected_prediction_tokens": 0,
        },
        "prompt_tokens_details": {
            "audio_tokens": 0,
            "cache_write_tokens": 2,
            "cached_tokens": 3,
        },
    },
}


def test_full_body_hydrates_every_object():
    completion = ChatCompletion.from_dict(FULL_BODY)

    choice = completion.choices[0]
    assert choice.logprobs.content[0].token == "Hi"
    assert choice.logprobs.content[0].top_logprobs[0].logprob == -0.1
    assert choice.logprobs.refusal == []

    message = choice.message
    assert message.annotations[0].url_citation.url == "https://example.com"
    assert message.audio.transcript == "hi"
    assert message.function_call.name == "legacy_fn"

    function_call, custom_call = message.tool_calls
    assert isinstance(function_call, ChatCompletionMessageFunctionToolCall)
    assert function_call.function.name == "search"
    assert isinstance(custom_call, ChatCompletionMessageCustomToolCall)
    assert custom_call.custom.input == "raw text"

    assert isinstance(completion.moderation.input, ModerationResults)
    assert completion.moderation.input.results[0].flagged is False
    assert isinstance(completion.moderation.output, Error)
    assert completion.moderation.output.code == "moderation_unavailable"

    assert completion.usage.prompt_tokens_details.cache_write_tokens == 2
    assert completion.usage.completion_tokens_details.reasoning_tokens == 5


def test_minimal_body_hydrates_with_absent_optionals():
    completion = ChatCompletion.from_dict(
        {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "hi",
                        "refusal": None,
                        "role": "assistant",
                    },
                }
            ],
            "created": 1_700_000_000,
            "model": "gpt-4o",
            "object": "chat.completion",
        }
    )
    assert completion.moderation is None
    assert completion.usage is None
    assert completion.choices[0].logprobs is None
    assert completion.choices[0].message.tool_calls is None
