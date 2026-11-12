"""UberLLM chat completions wire models (OpenAI-compatible gateway API).

Locked against https://api.uberllm.dev/v1/chat/completions
Model facts from https://api.uberllm.dev/v1/models at API_PINNED.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import Field, JsonValue, model_validator

from symai.backend.request import EngineAPIRequest, EngineRequestPayload, EngineResponsePayload

if TYPE_CHECKING:
    from symai.backend.usage import ModelPricing

API_PINNED = "2026-11-12"


@dataclass(frozen=True)
class UberLLMModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: bool
    vision: bool
    pricing: ModelPricing | None


# NOTE: UberLLM reprices per model as upstream sellers change, so a fixed per-token
# price here would go stale; pricing stays None and the live prices are listed at
# https://uberllm.dev/models. Context and output limits come from /v1/models.
UBERLLM_MODEL_SPECS = {
    "deepseek/deepseek-v4.1-flash": UberLLMModelSpec(
        context_tokens=1_048_576,
        response_tokens=384_000,
        reasoning=True,
        vision=True,
        pricing=None,
    ),
    "moonshotai/kimi-k2.5": UberLLMModelSpec(
        context_tokens=256_000,
        response_tokens=65_536,
        reasoning=True,
        vision=True,
        pricing=None,
    ),
}

SUPPORTED_CHAT_MODELS = [
    f"uberllm:{model}" for model, spec in UBERLLM_MODEL_SPECS.items() if not spec.reasoning
]
SUPPORTED_REASONING_MODELS = [
    f"uberllm:{model}" for model, spec in UBERLLM_MODEL_SPECS.items() if spec.reasoning
]
SUPPORTED_UBERLLM_MODELS = [f"uberllm:{model}" for model in UBERLLM_MODEL_SPECS]


def uberllm_strip_prefix(model_name: str) -> str:
    if model_name.startswith("uberllm:"):
        return model_name.removeprefix("uberllm:")
    return model_name


def uberllm_model_spec_for(model: str) -> UberLLMModelSpec:
    model_id = uberllm_strip_prefix(model)
    try:
        return UBERLLM_MODEL_SPECS[model_id]
    except KeyError as e:
        msg = f"Unsupported UberLLM model: {model}. Supported: {SUPPORTED_UBERLLM_MODELS}"
        raise ValueError(msg) from e


class UberLLMToolCallFunction(EngineRequestPayload):
    name: str
    arguments: str


class UberLLMToolCall(EngineRequestPayload):
    id: str
    type: Literal["function"]
    function: UberLLMToolCallFunction | None = None


class UberLLMMessage(EngineRequestPayload):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    # NOTE: assistant tool-call messages carry content=null; exclude_none omits it from
    # the wire body when replaying conversations.
    content: str | list[dict[str, JsonValue]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[UberLLMToolCall] | None = None


class UberLLMPayload(EngineRequestPayload):
    messages: list[UberLLMMessage]
    model: str
    frequency_penalty: float | int | None = None
    logit_bias: dict[str, float | int] | None = None
    logprobs: bool | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    n: int | None = Field(default=None, gt=0)
    parallel_tool_calls: bool | None = None
    presence_penalty: float | int | None = None
    provider: dict[str, JsonValue] | None = None
    reasoning: dict[str, JsonValue] | None = None
    response_format: dict[Literal["type"], Literal["text", "json_object", "json_schema"]] | None = (
        Field(default=None, min_length=1, max_length=1)
    )
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    stream_options: dict[Literal["include_usage"], bool] | None = Field(
        default=None,
        min_length=1,
        max_length=1,
    )
    temperature: float | int | None = Field(default=None, ge=0, le=2)
    tool_choice: Literal["none", "auto", "required"] | dict[str, JsonValue] | None = None
    tools: list[dict[str, JsonValue]] | None = None
    top_logprobs: int | None = Field(default=None, ge=0)
    top_p: float | int | None = Field(default=None, ge=0, le=1)
    transforms: list[str] | None = None
    user: str | None = None


class UberLLMOptions(EngineRequestPayload):
    extra_headers: dict[str, str] | None = None
    extra_query: dict[str, JsonValue] | None = None
    extra_body: dict[str, JsonValue] | None = None
    timeout: float | None = None


UberLLMRequest = EngineAPIRequest[
    UberLLMPayload,
    UberLLMOptions,
]


class UberLLMPromptTokensDetails(EngineResponsePayload):
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    audio_tokens: int | None = None
    video_tokens: int | None = None


class UberLLMUsage(EngineResponsePayload):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: UberLLMPromptTokensDetails | None = None
    # NOTE: a USD cost on the usage object is not guaranteed, so it is kept optional
    # for future cost cross-checks.
    cost: float | None = None


class UberLLMToolCallResult(EngineResponsePayload):
    id: str
    type: str
    function: UberLLMToolCallFunction | None = None
    index: int | None = None


class UberLLMResponseMessage(EngineResponsePayload):
    role: str
    # NOTE: null content is valid on assistant tool-call messages, and the key may be
    # absent entirely there; a text answer without content is a malformed response.
    content: str | None = None
    reasoning: str | None = None
    reasoning_details: list[dict[str, JsonValue]] | None = None
    refusal: str | None = None
    tool_calls: list[UberLLMToolCallResult] | None = None

    @model_validator(mode="after")
    def require_content_or_tool_calls(self):
        if self.content is None and not self.tool_calls:
            msg = "UberLLM response message requires content when no tool calls are present."
            raise ValueError(msg)
        return self


class UberLLMChoice(EngineResponsePayload):
    index: int
    message: UberLLMResponseMessage
    finish_reason: str | None = None


class UberLLMResponse(EngineResponsePayload):
    choices: list[UberLLMChoice] = Field(min_length=1)
    # NOTE: MetadataTracker reads raw_output.usage for token accounting; streaming
    # requests force stream_options.include_usage so the final chunk carries it.
    usage: UberLLMUsage
