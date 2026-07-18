"""Groq chat completions wire models (OpenAI-compatible API).

Locked against https://console.groq.com/docs/api-reference
Pricing: https://groq.com/pricing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from symai.backend.request import EngineAPIRequest, EngineRequestPayload, EngineResponsePayload
from symai.backend.usage import ModelPricing

API_PINNED = "2026-07-17"


@dataclass(frozen=True)
class GroqModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: bool
    vision: bool
    reasoning_efforts: tuple[str, ...]
    default_reasoning_effort: str
    pricing: ModelPricing | None


GROQ_MODEL_SPECS = {
    "openai/gpt-oss-120b": GroqModelSpec(
        context_tokens=131_072,
        response_tokens=32_766,
        reasoning=True,
        vision=False,
        reasoning_efforts=("low", "medium", "high"),
        default_reasoning_effort="low",
        pricing=ModelPricing(input=0.15, output=0.60, cached_input=0.075),
    ),
    "openai/gpt-oss-20b": GroqModelSpec(
        context_tokens=131_072,
        response_tokens=32_768,
        reasoning=True,
        vision=False,
        reasoning_efforts=("low", "medium", "high"),
        default_reasoning_effort="low",
        pricing=ModelPricing(input=0.075, output=0.30),
    ),
}

SUPPORTED_REASONING_MODELS = [
    f"groq:{model}" for model, spec in GROQ_MODEL_SPECS.items() if spec.reasoning
]
SUPPORTED_GROQ_MODELS = [f"groq:{model}" for model in GROQ_MODEL_SPECS]

# NOTE: parameters the Groq chat completions endpoint rejects. The engine drops them
# with a warning instead of failing the request, mirroring the pre-migration behavior.
GROQ_UNSUPPORTED_REQUEST_KWARGS = frozenset(
    {
        "logprobs",
        "logit_bias",
        "top_logprobs",
        "search_settings",
    }
)


def groq_strip_prefix(model_name: str) -> str:
    if model_name.startswith("groq:"):
        return model_name.removeprefix("groq:")
    return model_name


def groq_model_spec_for(model: str) -> GroqModelSpec:
    model_id = groq_strip_prefix(model)
    try:
        return GROQ_MODEL_SPECS[model_id]
    except KeyError as e:
        msg = f"Unsupported Groq model: {model}"
        raise ValueError(msg) from e


class GroqToolCallFunction(EngineRequestPayload):
    name: str
    arguments: str


class GroqToolCall(EngineRequestPayload):
    id: str
    type: Literal["function"]
    function: GroqToolCallFunction | None = None


class GroqMessage(EngineRequestPayload):
    role: Literal["system", "user", "assistant", "tool"]
    # NOTE: assistant tool-call messages carry content=null; exclude_none omits it from
    # the wire body when replaying conversations.
    content: str | list[dict[str, JsonValue]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[GroqToolCall] | None = None


class GroqPayload(EngineRequestPayload):
    messages: list[GroqMessage]
    model: str
    seed: int | None = None
    max_completion_tokens: int | None = Field(default=None, gt=0)
    stop: str | list[str] | None = None
    stream: bool | None = None
    stream_options: dict[Literal["include_usage"], bool] | None = Field(
        default=None,
        min_length=1,
        max_length=1,
    )
    temperature: float | int | None = Field(default=None, ge=0, le=2)
    frequency_penalty: float | int | None = None
    presence_penalty: float | int | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    service_tier: Literal["on_demand", "flex", "auto"] | None = None
    top_p: float | int | None = Field(default=None, ge=0, le=1)
    n: int | None = Field(default=None, gt=0)
    tools: list[dict[str, JsonValue]] | None = None
    tool_choice: Literal["none", "auto", "required"] | dict[str, JsonValue] | None = None
    response_format: dict[Literal["type"], Literal["text", "json_object"]] | None = Field(
        default=None,
        min_length=1,
        max_length=1,
    )
    parallel_tool_calls: bool | None = None
    user: str | None = None


class GroqOptions(EngineRequestPayload):
    extra_headers: dict[str, str] | None = None
    extra_query: dict[str, JsonValue] | None = None
    extra_body: dict[str, JsonValue] | None = None
    timeout: float | None = None


GroqRequest = EngineAPIRequest[
    GroqPayload,
    GroqOptions,
]


class GroqCompletionTokensDetails(EngineResponsePayload):
    reasoning_tokens: int | None = None


class GroqUsage(EngineResponsePayload):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    queue_time: float | None = None
    prompt_time: float | None = None
    completion_time: float | None = None
    total_time: float | None = None
    completion_tokens_details: GroqCompletionTokensDetails | None = None


class GroqToolCallResult(EngineResponsePayload):
    id: str
    type: str
    function: GroqToolCallFunction | None = None
    index: int | None = None


class GroqResponseMessage(EngineResponsePayload):
    role: str
    # NOTE: null content is valid on assistant tool-call messages, and the key may be
    # absent entirely there; a text answer without content is a malformed response.
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[GroqToolCallResult] | None = None

    @model_validator(mode="after")
    def require_content_or_tool_calls(self):
        if self.content is None and not self.tool_calls:
            msg = "Groq response message requires content when no tool calls are present."
            raise ValueError(msg)
        return self


class GroqChoice(EngineResponsePayload):
    index: int
    message: GroqResponseMessage
    finish_reason: str | None = None


class GroqResponse(EngineResponsePayload):
    choices: list[GroqChoice] = Field(min_length=1)
    # NOTE: MetadataTracker reads raw_output.usage for token accounting.
    usage: GroqUsage
