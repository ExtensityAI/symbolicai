"""Cheaper Inference chat completions wire models (OpenAI-compatible LLM gateway).

Locked against https://api.cheaperinference.com/v1/chat/completions
Model facts from https://api.cheaperinference.com/public/models at API_PINNED.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from symai.backend.request import EngineAPIRequest, EngineRequestPayload, EngineResponsePayload
from symai.backend.usage import ModelPricing

API_PINNED = "2026-10-28"


@dataclass(frozen=True)
class CheaperInferenceModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: bool
    vision: bool
    pricing: ModelPricing | None


# NOTE: Cheaper Inference serves models under bare ids (no vendor prefix). Limits and
# per-token prices are taken from the public catalog at API_PINNED.
CHEAPERINFERENCE_MODEL_SPECS = {
    "gpt-5.4-mini": CheaperInferenceModelSpec(
        context_tokens=400_000,
        response_tokens=128_000,
        reasoning=True,
        vision=True,
        pricing=ModelPricing(input=0.4875, output=2.925, cached_input=0.04875),
    ),
}

SUPPORTED_CHAT_MODELS = [
    f"cheaperinference:{model}"
    for model, spec in CHEAPERINFERENCE_MODEL_SPECS.items()
    if not spec.reasoning
]
SUPPORTED_REASONING_MODELS = [
    f"cheaperinference:{model}"
    for model, spec in CHEAPERINFERENCE_MODEL_SPECS.items()
    if spec.reasoning
]
SUPPORTED_CHEAPERINFERENCE_MODELS = [
    f"cheaperinference:{model}" for model in CHEAPERINFERENCE_MODEL_SPECS
]


def cheaperinference_strip_prefix(model_name: str) -> str:
    if model_name.startswith("cheaperinference:"):
        return model_name.removeprefix("cheaperinference:")
    return model_name


def cheaperinference_model_spec_for(model: str) -> CheaperInferenceModelSpec:
    model_id = cheaperinference_strip_prefix(model)
    try:
        return CHEAPERINFERENCE_MODEL_SPECS[model_id]
    except KeyError as e:
        msg = f"Unsupported Cheaper Inference model: {model}. Supported: {SUPPORTED_CHEAPERINFERENCE_MODELS}"
        raise ValueError(msg) from e


class CheaperInferenceToolCallFunction(EngineRequestPayload):
    name: str
    arguments: str


class CheaperInferenceToolCall(EngineRequestPayload):
    id: str
    type: Literal["function"]
    function: CheaperInferenceToolCallFunction | None = None


class CheaperInferenceMessage(EngineRequestPayload):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    # NOTE: assistant tool-call messages carry content=null; exclude_none omits it from
    # the wire body when replaying conversations.
    content: str | list[dict[str, JsonValue]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[CheaperInferenceToolCall] | None = None


class CheaperInferencePayload(EngineRequestPayload):
    messages: list[CheaperInferenceMessage]
    model: str
    frequency_penalty: float | int | None = None
    logit_bias: dict[str, float | int] | None = None
    logprobs: bool | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    n: int | None = Field(default=None, gt=0)
    parallel_tool_calls: bool | None = None
    presence_penalty: float | int | None = None
    reasoning_effort: str | None = None
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
    user: str | None = None


class CheaperInferenceOptions(EngineRequestPayload):
    extra_headers: dict[str, str] | None = None
    extra_query: dict[str, JsonValue] | None = None
    extra_body: dict[str, JsonValue] | None = None
    timeout: float | None = None


CheaperInferenceRequest = EngineAPIRequest[
    CheaperInferencePayload,
    CheaperInferenceOptions,
]


class CheaperInferencePromptTokensDetails(EngineResponsePayload):
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    audio_tokens: int | None = None
    video_tokens: int | None = None


class CheaperInferenceUsage(EngineResponsePayload):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: CheaperInferencePromptTokensDetails | None = None


class CheaperInferenceToolCallResult(EngineResponsePayload):
    id: str
    type: str
    function: CheaperInferenceToolCallFunction | None = None
    index: int | None = None


class CheaperInferenceResponseMessage(EngineResponsePayload):
    role: str
    # NOTE: null content is valid on assistant tool-call messages, and the key may be
    # absent entirely there; a text answer without content is a malformed response.
    content: str | None = None
    # NOTE: the reasoning summary arrives in reasoning_content; reasoning is accepted
    # as a fallback for routes that use the older field name.
    reasoning_content: str | None = None
    reasoning: str | None = None
    refusal: str | None = None
    tool_calls: list[CheaperInferenceToolCallResult] | None = None

    @model_validator(mode="after")
    def require_content_or_tool_calls(self):
        if self.content is None and not self.tool_calls:
            msg = "Cheaper Inference response message requires content when no tool calls are present."
            raise ValueError(msg)
        return self


class CheaperInferenceChoice(EngineResponsePayload):
    index: int
    message: CheaperInferenceResponseMessage
    finish_reason: str | None = None


class CheaperInferenceResponse(EngineResponsePayload):
    choices: list[CheaperInferenceChoice] = Field(min_length=1)
    # NOTE: MetadataTracker reads raw_output.usage for token accounting; streaming
    # requests force stream_options.include_usage so the final chunk carries it.
    usage: CheaperInferenceUsage
