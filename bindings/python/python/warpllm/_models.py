"""Field-for-field copy of the OpenAI `chat.completion` response objects.

Object names and field order mirror the upstream reference:
https://developers.openai.com/api/reference/resources/chat
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AnnotationURLCitation:
    end_index: int
    start_index: int
    title: str
    url: str


@dataclass
class Annotation:
    type: str  # always "url_citation"
    url_citation: AnnotationURLCitation


@dataclass
class FunctionCall:
    """Deprecated upstream in favor of tool calls."""

    arguments: str
    name: str


@dataclass
class Function:
    arguments: str
    name: str


@dataclass
class ChatCompletionMessageFunctionToolCall:
    id: str
    function: Function
    type: str = "function"


@dataclass
class Custom:
    input: str
    name: str


@dataclass
class ChatCompletionMessageCustomToolCall:
    id: str
    custom: Custom
    type: str = "custom"


ChatCompletionMessageToolCall = (
    ChatCompletionMessageFunctionToolCall | ChatCompletionMessageCustomToolCall
)


@dataclass
class ChatCompletionAudio:
    id: str
    data: str  # base64-encoded audio bytes
    expires_at: int
    transcript: str


@dataclass
class TopLogprob:
    token: str
    bytes: list[int] | None
    logprob: float


@dataclass
class ChatCompletionTokenLogprob:
    token: str
    bytes: list[int] | None
    logprob: float
    top_logprobs: list[TopLogprob]


@dataclass
class ChoiceLogprobs:
    """Both arrays are required and non-nullable when `logprobs` is present."""

    content: list[ChatCompletionTokenLogprob]
    refusal: list[ChatCompletionTokenLogprob]


@dataclass
class ChatCompletionMessage:
    content: str | None
    refusal: str | None
    role: str  # always "assistant"
    annotations: list[Annotation] | None = None
    audio: ChatCompletionAudio | None = None
    function_call: FunctionCall | None = None
    tool_calls: list[ChatCompletionMessageToolCall] | None = None


@dataclass
class Choice:
    finish_reason: str
    index: int
    message: ChatCompletionMessage
    logprobs: ChoiceLogprobs | None = None


@dataclass
class CompletionTokensDetails:
    accepted_prediction_tokens: int | None = None
    audio_tokens: int | None = None
    reasoning_tokens: int | None = None
    rejected_prediction_tokens: int | None = None


@dataclass
class PromptTokensDetails:
    audio_tokens: int | None = None
    cache_write_tokens: int | None = None
    cached_tokens: int | None = None


@dataclass
class CompletionUsage:
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails | None = None
    prompt_tokens_details: PromptTokensDetails | None = None


# The docs define one shared ModerationResults/Error pair used by both
# `input` and `output` (openai-python duplicates them per side; that is a
# codegen artifact, not the documented naming).


@dataclass
class ModerationResult:
    """One verdict in `ModerationResults.results`. The docs leave this
    element object unnamed; named here after its `type` string."""

    categories: dict[str, bool]
    category_applied_input_types: dict[str, list[str]]
    category_scores: dict[str, float]
    flagged: bool
    model: str
    type: str = "moderation_result"


@dataclass
class ModerationResults:
    model: str
    results: list[ModerationResult]
    type: str = "moderation_results"


@dataclass
class Error:
    code: str
    message: str
    type: str = "error"


@dataclass
class Moderation:
    input: ModerationResults | Error
    output: ModerationResults | Error


def _tool_call_from_dict(
    data: dict[str, Any],
) -> ChatCompletionMessageToolCall:
    if data["type"] == "custom":
        return ChatCompletionMessageCustomToolCall(
            id=data["id"], custom=Custom(**data["custom"])
        )
    return ChatCompletionMessageFunctionToolCall(
        id=data["id"], function=Function(**data["function"])
    )


def _token_logprob_from_dict(
    data: dict[str, Any],
) -> ChatCompletionTokenLogprob:
    return ChatCompletionTokenLogprob(
        token=data["token"],
        bytes=data.get("bytes"),
        logprob=data["logprob"],
        top_logprobs=[TopLogprob(**t) for t in data["top_logprobs"]],
    )


def _logprobs_from_dict(data: dict[str, Any]) -> ChoiceLogprobs:
    return ChoiceLogprobs(
        content=[_token_logprob_from_dict(t) for t in data["content"]],
        refusal=[_token_logprob_from_dict(t) for t in data["refusal"]],
    )


def _message_from_dict(data: dict[str, Any]) -> ChatCompletionMessage:
    audio = data.get("audio")
    function_call = data.get("function_call")
    annotations = data.get("annotations")
    tool_calls = data.get("tool_calls")
    return ChatCompletionMessage(
        content=data.get("content"),
        refusal=data.get("refusal"),
        role=data["role"],
        annotations=(
            [
                Annotation(
                    type=a["type"],
                    url_citation=AnnotationURLCitation(**a["url_citation"]),
                )
                for a in annotations
            ]
            if annotations is not None
            else None
        ),
        audio=ChatCompletionAudio(**audio) if audio else None,
        function_call=FunctionCall(**function_call) if function_call else None,
        tool_calls=(
            [_tool_call_from_dict(t) for t in tool_calls]
            if tool_calls is not None
            else None
        ),
    )


def _choice_from_dict(data: dict[str, Any]) -> Choice:
    logprobs = data.get("logprobs")
    return Choice(
        finish_reason=data["finish_reason"],
        index=data["index"],
        message=_message_from_dict(data["message"]),
        logprobs=_logprobs_from_dict(logprobs) if logprobs else None,
    )


def _usage_from_dict(data: dict[str, Any]) -> CompletionUsage:
    completion_details = data.get("completion_tokens_details")
    prompt_details = data.get("prompt_tokens_details")
    return CompletionUsage(
        completion_tokens=data["completion_tokens"],
        prompt_tokens=data["prompt_tokens"],
        total_tokens=data["total_tokens"],
        completion_tokens_details=(
            CompletionTokensDetails(**completion_details)
            if completion_details
            else None
        ),
        prompt_tokens_details=(
            PromptTokensDetails(**prompt_details) if prompt_details else None
        ),
    )


def _moderation_value_from_dict(
    data: dict[str, Any],
) -> ModerationResults | Error:
    if data["type"] == "error":
        return Error(code=data["code"], message=data["message"])
    return ModerationResults(
        model=data["model"],
        results=[ModerationResult(**r) for r in data["results"]],
    )


@dataclass
class ChatCompletion:
    id: str
    choices: list[Choice]
    created: int
    model: str
    object: str
    moderation: Moderation | None = None
    service_tier: str | None = None
    system_fingerprint: str | None = None
    usage: CompletionUsage | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatCompletion:
        moderation = data.get("moderation")
        usage = data.get("usage")
        return cls(
            id=data["id"],
            choices=[_choice_from_dict(c) for c in data["choices"]],
            created=data["created"],
            model=data["model"],
            object=data["object"],
            moderation=(
                Moderation(
                    input=_moderation_value_from_dict(moderation["input"]),
                    output=_moderation_value_from_dict(moderation["output"]),
                )
                if moderation
                else None
            ),
            service_tier=data.get("service_tier"),
            system_fingerprint=data.get("system_fingerprint"),
            usage=_usage_from_dict(usage) if usage else None,
        )
