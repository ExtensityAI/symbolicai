"""DeepSeek chat completions wire models.

Locked against https://api-docs.deepseek.com/api/create-chat-completion
Pricing locked against https://api-docs.deepseek.com/quick_start/pricing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, JsonValue

from symai.backend.request import EngineAPIRequest, EngineRequestPayload, EngineResponsePayload
from symai.backend.usage import ModelPricing

API_PINNED = "2026-07-17"


@dataclass(frozen=True)
class DeepSeekModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: bool
    vision: bool
    pricing: ModelPricing


DEEPSEEK_MODEL_SPECS = {
    "deepseek-v4-flash": DeepSeekModelSpec(
        context_tokens=1_000_000,
        response_tokens=384_000,
        reasoning=True,
        vision=False,
        pricing=ModelPricing(input=0.14, output=0.28, cached_input=0.0028),
    ),
    "deepseek-v4-pro": DeepSeekModelSpec(
        context_tokens=1_000_000,
        response_tokens=384_000,
        reasoning=True,
        vision=False,
        pricing=ModelPricing(input=0.435, output=0.87, cached_input=0.003625),
    ),
}

SUPPORTED_MODELS = list(DEEPSEEK_MODEL_SPECS)


def deepseek_model_spec_for(model: str) -> DeepSeekModelSpec:
    try:
        return DEEPSEEK_MODEL_SPECS[model]
    except KeyError as e:
        msg = f"Unsupported DeepSeek model: {model}"
        raise ValueError(msg) from e


class DeepSeekToolCallFunction(EngineRequestPayload):
    name: str
    arguments: str


class DeepSeekToolCall(EngineRequestPayload):
    id: str
    type: Literal["function"]
    function: DeepSeekToolCallFunction | None = None


class DeepSeekMessage(EngineRequestPayload):
    role: Literal["system", "user", "assistant", "tool"]
    # NOTE: DeepSeek returns content=null on assistant tool-call messages, so null must
    # round-trip when replaying conversations. exclude_none omits it from the wire body.
    content: str | None = None
    name: str | None = None
    prefix: bool | None = None
    reasoning_content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[DeepSeekToolCall] | None = None


class DeepSeekPayload(EngineRequestPayload):
    messages: list[DeepSeekMessage]
    model: str
    thinking: dict[Literal["type"], Literal["enabled", "disabled"]] | None = Field(
        default=None,
        min_length=1,
        max_length=1,
    )
    reasoning_effort: Literal["high", "max"] | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    response_format: dict[Literal["type"], Literal["text", "json_object"]] | None = Field(
        default=None,
        min_length=1,
        max_length=1,
    )
    stop: str | list[str] | None = None
    stream: bool | None = None
    stream_options: dict[Literal["include_usage"], bool] | None = Field(
        default=None,
        min_length=1,
        max_length=1,
    )
    temperature: float | int | None = Field(default=None, ge=0, le=2)
    top_p: float | int | None = Field(default=None, ge=0, le=1)
    tools: list[dict[str, JsonValue]] | None = None
    tool_choice: Literal["none", "auto", "required"] | dict[str, JsonValue] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    user_id: str | None = None
    seed: int | None = None
    n: int | None = Field(default=None, gt=0)
    logit_bias: dict[str, int] | None = None
    frequency_penalty: float | int | None = None
    presence_penalty: float | int | None = None


class DeepSeekOptions(EngineRequestPayload):
    extra_headers: dict[str, str] | None = None
    extra_query: dict[str, JsonValue] | None = None
    extra_body: dict[str, JsonValue] | None = None
    timeout: float | None = None


DeepSeekRequest = EngineAPIRequest[
    DeepSeekPayload,
    DeepSeekOptions,
]


class DeepSeekPromptTokensDetails(EngineResponsePayload):
    cached_tokens: int | None = None


class DeepSeekCompletionTokensDetails(EngineResponsePayload):
    reasoning_tokens: int | None = None


class DeepSeekUsage(EngineResponsePayload):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: DeepSeekPromptTokensDetails | None = None
    completion_tokens_details: DeepSeekCompletionTokensDetails | None = None
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0


class DeepSeekResponseMessage(EngineResponsePayload):
    role: str
    # NOTE: required but nullable — a missing content key fails validation while an
    # explicit null (assistant tool-call messages) is a valid provider response.
    content: str | None
    reasoning_content: str | None = None
    tool_calls: list[DeepSeekToolCall] | None = None


class DeepSeekChoice(EngineResponsePayload):
    index: int
    message: DeepSeekResponseMessage
    finish_reason: str | None = None


class DeepSeekResponse(EngineResponsePayload):
    choices: list[DeepSeekChoice] = Field(min_length=1)
    # NOTE: MetadataTracker reads raw_output.usage for token accounting; the API always
    # returns it (streaming requests force stream_options.include_usage).
    usage: DeepSeekUsage
