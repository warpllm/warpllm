"""API-surface fidelity: our response dataclasses must carry the exact field
names of the corresponding classes in the official openai SDK (a dev-only
dependency, generated from the same OpenAPI spec as the docs). When an SDK
bump adds or renames a field, this fails and names it.

Optionality is not compared here: we omit absent fields on the wire while
the SDK types allow explicit nulls; byte-level shape is covered by the Rust
round-trip tests."""

import dataclasses

import pytest
import warpllm
from openai.types import completion_usage as sdk_usage
from openai.types.chat import chat_completion as sdk_chat
from openai.types.chat import chat_completion_message as sdk_message
from openai.types.chat import (
    chat_completion_message_custom_tool_call as sdk_custom_tool_call,
)
from openai.types.chat import (
    chat_completion_message_function_tool_call as sdk_function_tool_call,
)
from openai.types.chat import chat_completion_token_logprob as sdk_logprob

PAIRS = [
    (warpllm.ChatCompletion, sdk_chat.ChatCompletion),
    (warpllm.Choice, sdk_chat.Choice),
    (warpllm.ChoiceLogprobs, sdk_chat.ChoiceLogprobs),
    (warpllm.ChatCompletionMessage, sdk_message.ChatCompletionMessage),
    (
        warpllm.ChatCompletionTokenLogprob,
        sdk_logprob.ChatCompletionTokenLogprob,
    ),
    (warpllm.TopLogprob, sdk_logprob.TopLogprob),
    (warpllm.ChatCompletionAudio, sdk_message.ChatCompletionAudio),
    (warpllm.Annotation, sdk_message.Annotation),
    (warpllm.AnnotationURLCitation, sdk_message.AnnotationURLCitation),
    (warpllm.FunctionCall, sdk_message.FunctionCall),
    (
        warpllm.ChatCompletionMessageFunctionToolCall,
        sdk_function_tool_call.ChatCompletionMessageFunctionToolCall,
    ),
    (warpllm.Function, sdk_function_tool_call.Function),
    (
        warpllm.ChatCompletionMessageCustomToolCall,
        sdk_custom_tool_call.ChatCompletionMessageCustomToolCall,
    ),
    (warpllm.Custom, sdk_custom_tool_call.Custom),
    (warpllm.CompletionUsage, sdk_usage.CompletionUsage),
    (warpllm.PromptTokensDetails, sdk_usage.PromptTokensDetails),
    (warpllm.CompletionTokensDetails, sdk_usage.CompletionTokensDetails),
    (warpllm.Moderation, sdk_chat.Moderation),
]

# The docs define one shared ModerationResults/Error pair used by both
# moderation `input` and `output`; openai-python instead duplicates the
# classes per side (a codegen artifact). We follow the docs, so each shared
# class is field-checked against BOTH SDK duplicates, and the name-equality
# test skips these pairs.
COLLAPSED_MODERATION_PAIRS = [
    (warpllm.ModerationResults, sdk_chat.ModerationInputModerationResults),
    (warpllm.ModerationResults, sdk_chat.ModerationOutputModerationResults),
    (
        warpllm.ModerationResult,
        sdk_chat.ModerationInputModerationResultsResult,
    ),
    (
        warpllm.ModerationResult,
        sdk_chat.ModerationOutputModerationResultsResult,
    ),
    (warpllm.Error, sdk_chat.ModerationInputError),
    (warpllm.Error, sdk_chat.ModerationOutputError),
]

PAIRS += COLLAPSED_MODERATION_PAIRS


def test_every_class_shares_its_sdk_name():
    collapsed = {sdk for _, sdk in COLLAPSED_MODERATION_PAIRS}
    for ours, sdk in PAIRS:
        if sdk in collapsed:
            continue
        assert ours.__name__ == sdk.__name__


@pytest.mark.parametrize("ours,sdk", PAIRS, ids=lambda cls: cls.__name__)
def test_field_names_match_sdk(ours, sdk):
    our_fields = {f.name for f in dataclasses.fields(ours)}
    sdk_fields = set(sdk.model_fields)
    assert our_fields == sdk_fields, (
        f"missing from warpllm: {sorted(sdk_fields - our_fields)}; "
        f"extra in warpllm: {sorted(our_fields - sdk_fields)}"
    )
