"""OrcaRouter chat completions wire models (OpenAI-compatible model routing gateway).

Locked against https://api.orcarouter.ai/v1/chat/completions
Model facts from https://api.orcarouter.ai/v1/models at API_PINNED.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import Field, JsonValue, model_validator

from symai.backend.request import EngineAPIRequest, EngineRequestPayload, EngineResponsePayload

if TYPE_CHECKING:
    from symai.backend.usage import ModelPricing

API_PINNED = "2026-08-12"


@dataclass(frozen=True)
class OrcaRouterModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: bool
    vision: bool
    pricing: ModelPricing | None


# NOTE: orcarouter/auto is OrcaRouter's default router: it picks the best underlying
# model per request, so neither the context window nor the per-token price is fixed.
# The limits below are conservative defaults for pre-flight token budgeting, and
# pricing stays None because billing follows the routed model.
ORCAROUTER_MODEL_SPECS = {
    "orcarouter/auto": OrcaRouterModelSpec(
        context_tokens=262_144,
        response_tokens=65_536,
        reasoning=True,
        vision=True,
        pricing=None,
    ),
}

SUPPORTED_CHAT_MODELS = [
    f"orcarouter:{model}" for model, spec in ORCAROUTER_MODEL_SPECS.items() if not spec.reasoning
]
SUPPORTED_REASONING_MODELS = [
    f"orcarouter:{model}" for model, spec in ORCAROUTER_MODEL_SPECS.items() if spec.reasoning
]
SUPPORTED_ORCAROUTER_MODELS = [f"orcarouter:{model}" for model in ORCAROUTER_MODEL_SPECS]


def orcarouter_strip_prefix(model_name: str) -> str:
    if model_name.startswith("orcarouter:"):
        return model_name.removeprefix("orcarouter:")
    return model_name


def orcarouter_model_spec_for(model: str) -> OrcaRouterModelSpec:
    model_id = orcarouter_strip_prefix(model)
    try:
        return ORCAROUTER_MODEL_SPECS[model_id]
    except KeyError as e:
        msg = f"Unsupported OrcaRouter model: {model}. Supported: {SUPPORTED_ORCAROUTER_MODELS}"
        raise ValueError(msg) from e


class OrcaRouterToolCallFunction(EngineRequestPayload):
    name: str
    arguments: str


class OrcaRouterToolCall(EngineRequestPayload):
    id: str
    type: Literal["function"]
    function: OrcaRouterToolCallFunction | None = None


class OrcaRouterMessage(EngineRequestPayload):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    # NOTE: assistant tool-call messages carry content=null; exclude_none omits it from
    # the wire body when replaying conversations.
    content: str | list[dict[str, JsonValue]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[OrcaRouterToolCall] | None = None


class OrcaRouterPayload(EngineRequestPayload):
    messages: list[OrcaRouterMessage]
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


class OrcaRouterOptions(EngineRequestPayload):
    extra_headers: dict[str, str] | None = None
    extra_query: dict[str, JsonValue] | None = None
    extra_body: dict[str, JsonValue] | None = None
    timeout: float | None = None


OrcaRouterRequest = EngineAPIRequest[
    OrcaRouterPayload,
    OrcaRouterOptions,
]


class OrcaRouterPromptTokensDetails(EngineResponsePayload):
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    audio_tokens: int | None = None
    video_tokens: int | None = None


class OrcaRouterUsage(EngineResponsePayload):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: OrcaRouterPromptTokensDetails | None = None
    # NOTE: OrcaRouter bills per routed model; a fixed USD cost on the usage object is
    # not guaranteed, so it is kept optional for future cost cross-checks.
    cost: float | None = None


class OrcaRouterToolCallResult(EngineResponsePayload):
    id: str
    type: str
    function: OrcaRouterToolCallFunction | None = None
    index: int | None = None


class OrcaRouterResponseMessage(EngineResponsePayload):
    role: str
    # NOTE: null content is valid on assistant tool-call messages, and the key may be
    # absent entirely there; a text answer without content is a malformed response.
    content: str | None = None
    reasoning: str | None = None
    reasoning_details: list[dict[str, JsonValue]] | None = None
    refusal: str | None = None
    tool_calls: list[OrcaRouterToolCallResult] | None = None

    @model_validator(mode="after")
    def require_content_or_tool_calls(self):
        if self.content is None and not self.tool_calls:
            msg = "OrcaRouter response message requires content when no tool calls are present."
            raise ValueError(msg)
        return self


class OrcaRouterChoice(EngineResponsePayload):
    index: int
    message: OrcaRouterResponseMessage
    finish_reason: str | None = None


class OrcaRouterResponse(EngineResponsePayload):
    choices: list[OrcaRouterChoice] = Field(min_length=1)
    # NOTE: MetadataTracker reads raw_output.usage for token accounting; streaming
    # requests force stream_options.include_usage so the final chunk carries it.
    usage: OrcaRouterUsage
