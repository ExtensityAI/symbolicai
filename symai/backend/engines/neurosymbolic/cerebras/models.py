"""Cerebras chat completions wire models.

Locked against https://inference-docs.cerebras.ai/api-reference/chat-completions
Pricing: https://www.cerebras.ai/pricing (developer tier)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from symai.backend.request import EngineAPIRequest, EngineRequestPayload, EngineResponsePayload
from symai.backend.usage import ModelPricing

API_PINNED = "2026-07-17"


@dataclass(frozen=True)
class CerebrasModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: bool
    vision: bool
    reasoning_efforts: tuple[str, ...]
    # NOTE: None means the provider has not published per-token prices for the model
    # at API_PINNED (gemma-4-31b is absent from the current pricing tables).
    pricing: ModelPricing | None


CEREBRAS_MODEL_SPECS = {
    "gpt-oss-120b": CerebrasModelSpec(
        context_tokens=131_072,
        response_tokens=40_000,
        reasoning=True,
        vision=False,
        reasoning_efforts=("low", "medium", "high"),
        pricing=ModelPricing(input=0.35, output=0.75),
    ),
    "gemma-4-31b": CerebrasModelSpec(
        context_tokens=131_072,
        response_tokens=40_000,
        reasoning=True,
        vision=False,
        reasoning_efforts=("low", "medium", "high"),
        pricing=None,
    ),
    "zai-glm-4.7": CerebrasModelSpec(
        context_tokens=131_072,
        response_tokens=40_000,
        reasoning=True,
        vision=False,
        reasoning_efforts=("none", "low", "medium", "high"),
        pricing=ModelPricing(input=2.25, output=2.75),
    ),
}

SUPPORTED_CHAT_MODELS = [
    f"cerebras:{model}" for model, spec in CEREBRAS_MODEL_SPECS.items() if not spec.reasoning
]
SUPPORTED_REASONING_MODELS = [
    f"cerebras:{model}" for model, spec in CEREBRAS_MODEL_SPECS.items() if spec.reasoning
]
SUPPORTED_CEREBRAS_MODELS = [f"cerebras:{model}" for model in CEREBRAS_MODEL_SPECS]


def cerebras_strip_prefix(model_name: str) -> str:
    if model_name.startswith("cerebras:"):
        return model_name.removeprefix("cerebras:")
    return model_name


def cerebras_model_spec_for(model: str) -> CerebrasModelSpec:
    model_id = cerebras_strip_prefix(model)
    try:
        return CEREBRAS_MODEL_SPECS[model_id]
    except KeyError as e:
        msg = f"Unsupported Cerebras model: {model}"
        raise ValueError(msg) from e


class CerebrasToolCallFunction(EngineRequestPayload):
    name: str
    arguments: str


class CerebrasToolCall(EngineRequestPayload):
    id: str
    type: Literal["function"]
    function: CerebrasToolCallFunction | None = None


class CerebrasMessage(EngineRequestPayload):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    # NOTE: assistant tool-call messages carry content=null; exclude_none omits it from
    # the wire body when replaying conversations.
    content: str | list[dict[str, JsonValue]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[CerebrasToolCall] | None = None


class CerebrasPayload(EngineRequestPayload):
    messages: list[CerebrasMessage]
    model: str
    clear_thinking: bool | None = None
    disable_reasoning: bool | None = None
    frequency_penalty: float | int | None = None
    logit_bias: dict[str, float | int] | None = None
    logprobs: bool | None = None
    max_completion_tokens: int | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    min_completion_tokens: int | None = Field(default=None, ge=0)
    min_tokens: int | None = Field(default=None, ge=0)
    n: int | None = Field(default=None, gt=0)
    parallel_tool_calls: bool | None = None
    prediction: dict[str, JsonValue] | None = None
    presence_penalty: float | int | None = None
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = None
    reasoning_format: Literal["none", "parsed", "text_parsed", "raw", "hidden"] | None = None
    response_format: dict[str, JsonValue] | None = None
    seed: int | None = None
    service_tier: Literal["auto", "default", "flex", "priority"] | None = None
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
    user: str | None = None


class CerebrasOptions(EngineRequestPayload):
    extra_headers: dict[str, str] | None = None
    extra_query: dict[str, JsonValue] | None = None
    extra_body: dict[str, JsonValue] | None = None
    timeout: float | None = None


CerebrasRequest = EngineAPIRequest[
    CerebrasPayload,
    CerebrasOptions,
]


class CerebrasPromptTokensDetails(EngineResponsePayload):
    cached_tokens: int | None = None


class CerebrasCompletionTokensDetails(EngineResponsePayload):
    accepted_prediction_tokens: int | None = None
    rejected_prediction_tokens: int | None = None
    reasoning_tokens: int | None = None


class CerebrasUsage(EngineResponsePayload):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: CerebrasPromptTokensDetails | None = None
    completion_tokens_details: CerebrasCompletionTokensDetails | None = None


class CerebrasTimeInfo(EngineResponsePayload):
    created: float | None = None
    queue_time: float | None = None
    prompt_time: float | None = None
    completion_time: float | None = None
    total_time: float | None = None


class CerebrasToolCallResult(EngineResponsePayload):
    id: str
    type: str
    function: CerebrasToolCallFunction | None = None
    index: int | None = None


class CerebrasResponseMessage(EngineResponsePayload):
    role: str
    # NOTE: null content is valid on assistant tool-call messages, and the key may be
    # absent entirely there; a text answer without content is a malformed response.
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[CerebrasToolCallResult] | None = None

    @model_validator(mode="after")
    def require_content_or_tool_calls(self):
        if self.content is None and not self.tool_calls:
            msg = "Cerebras response message requires content when no tool calls are present."
            raise ValueError(msg)
        return self


class CerebrasChoice(EngineResponsePayload):
    index: int
    message: CerebrasResponseMessage
    finish_reason: str | None = None


class CerebrasResponse(EngineResponsePayload):
    choices: list[CerebrasChoice] = Field(min_length=1)
    # NOTE: MetadataTracker reads raw_output.usage for token accounting; streaming
    # requests force stream_options.include_usage so the final chunk carries it.
    usage: CerebrasUsage
    time_info: CerebrasTimeInfo | None = None


def cerebras_normalize_model(model: str | None) -> str | None:
    """Canonicalize a bare model name to the prefixed form ('o3' -> 'cerebras:o3').

    DynamicEngine and explicit constructors accept both forms; the wire and the
    supported-model lists use the prefixed form everywhere else.
    """
    if model and ":" not in model:
        return f"cerebras:{model}"
    return model
