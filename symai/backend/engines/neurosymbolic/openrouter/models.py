"""OpenRouter chat completions wire models (OpenAI-compatible gateway API).

Locked against https://openrouter.ai/docs/api-reference/chat-completion
Model facts from https://openrouter.ai/api/v1/models at API_PINNED.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from symai.backend.request import EngineAPIRequest, EngineRequestPayload, EngineResponsePayload
from symai.backend.usage import ModelPricing

API_PINNED = "2026-07-17"


@dataclass(frozen=True)
class OpenRouterModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: bool
    vision: bool
    pricing: ModelPricing | None


OPENROUTER_MODEL_SPECS = {
    "moonshotai/kimi-k2.5": OpenRouterModelSpec(
        context_tokens=262_144,
        response_tokens=262_144,
        reasoning=True,
        vision=True,
        pricing=ModelPricing(input=0.57, output=2.85, cached_input=0.095),
    ),
}

SUPPORTED_CHAT_MODELS = [
    f"openrouter:{model}" for model, spec in OPENROUTER_MODEL_SPECS.items() if not spec.reasoning
]
SUPPORTED_REASONING_MODELS = [
    f"openrouter:{model}" for model, spec in OPENROUTER_MODEL_SPECS.items() if spec.reasoning
]
SUPPORTED_OPENROUTER_MODELS = [f"openrouter:{model}" for model in OPENROUTER_MODEL_SPECS]


def openrouter_strip_prefix(model_name: str) -> str:
    if model_name.startswith("openrouter:"):
        return model_name.removeprefix("openrouter:")
    return model_name


def openrouter_model_spec_for(model: str) -> OpenRouterModelSpec:
    model_id = openrouter_strip_prefix(model)
    try:
        return OPENROUTER_MODEL_SPECS[model_id]
    except KeyError as e:
        msg = f"Unsupported OpenRouter model: {model}. Supported: {SUPPORTED_OPENROUTER_MODELS}"
        raise ValueError(msg) from e


class OpenRouterToolCallFunction(EngineRequestPayload):
    name: str
    arguments: str


class OpenRouterToolCall(EngineRequestPayload):
    id: str
    type: Literal["function"]
    function: OpenRouterToolCallFunction | None = None


class OpenRouterMessage(EngineRequestPayload):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    # NOTE: assistant tool-call messages carry content=null; exclude_none omits it from
    # the wire body when replaying conversations.
    content: str | list[dict[str, JsonValue]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[OpenRouterToolCall] | None = None


class OpenRouterPayload(EngineRequestPayload):
    messages: list[OpenRouterMessage]
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


class OpenRouterOptions(EngineRequestPayload):
    extra_headers: dict[str, str] | None = None
    extra_query: dict[str, JsonValue] | None = None
    extra_body: dict[str, JsonValue] | None = None
    timeout: float | None = None


OpenRouterRequest = EngineAPIRequest[
    OpenRouterPayload,
    OpenRouterOptions,
]


class OpenRouterPromptTokensDetails(EngineResponsePayload):
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    audio_tokens: int | None = None
    video_tokens: int | None = None


class OpenRouterUsage(EngineResponsePayload):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: OpenRouterPromptTokensDetails | None = None
    # NOTE: OpenRouter bills directly in USD on the usage object; kept for future
    # cost cross-checks against the spec pricing.
    cost: float | None = None
    is_byok: bool | None = None


class OpenRouterToolCallResult(EngineResponsePayload):
    id: str
    type: str
    function: OpenRouterToolCallFunction | None = None
    index: int | None = None


class OpenRouterResponseMessage(EngineResponsePayload):
    role: str
    # NOTE: null content is valid on assistant tool-call messages, and the key may be
    # absent entirely there; a text answer without content is a malformed response.
    content: str | None = None
    reasoning: str | None = None
    reasoning_details: list[dict[str, JsonValue]] | None = None
    refusal: str | None = None
    tool_calls: list[OpenRouterToolCallResult] | None = None

    @model_validator(mode="after")
    def require_content_or_tool_calls(self):
        if self.content is None and not self.tool_calls:
            msg = "OpenRouter response message requires content when no tool calls are present."
            raise ValueError(msg)
        return self


class OpenRouterChoice(EngineResponsePayload):
    index: int
    message: OpenRouterResponseMessage
    finish_reason: str | None = None


class OpenRouterResponse(EngineResponsePayload):
    choices: list[OpenRouterChoice] = Field(min_length=1)
    # NOTE: MetadataTracker reads raw_output.usage for token accounting; streaming
    # requests force stream_options.include_usage so the final chunk carries it.
    usage: OpenRouterUsage


def openrouter_normalize_model(model: str | None) -> str | None:
    """Canonicalize a bare model name to the prefixed form ('o3' -> 'openrouter:o3').

    DynamicEngine and explicit constructors accept both forms; the wire and the
    supported-model lists use the prefixed form everywhere else.
    """
    if model and ":" not in model:
        return f"openrouter:{model}"
    return model
