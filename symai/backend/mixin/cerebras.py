from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field

from symai.backend.request import EngineAPIRequest, EngineRequestPayload


@dataclass(frozen=True)
class CerebrasModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: bool
    reasoning_efforts: tuple[str, ...]


CEREBRAS_MODEL_SPECS = {
    "gpt-oss-120b": CerebrasModelSpec(
        context_tokens=131_072,
        response_tokens=40_000,
        reasoning=True,
        reasoning_efforts=("low", "medium", "high"),
    ),
    "gemma-4-31b": CerebrasModelSpec(
        context_tokens=131_072,
        response_tokens=40_000,
        reasoning=True,
        reasoning_efforts=("low", "medium", "high"),
    ),
    "zai-glm-4.7": CerebrasModelSpec(
        context_tokens=131_072,
        response_tokens=40_000,
        reasoning=True,
        reasoning_efforts=("none", "low", "medium", "high"),
    ),
}

SUPPORTED_CHAT_MODELS = [
    f"cerebras:{model}" for model, spec in CEREBRAS_MODEL_SPECS.items() if not spec.reasoning
]
SUPPORTED_REASONING_MODELS = [
    f"cerebras:{model}" for model, spec in CEREBRAS_MODEL_SPECS.items() if spec.reasoning
]
SUPPORTED_CEREBRAS_MODELS = [f"cerebras:{model}" for model in CEREBRAS_MODEL_SPECS]


class CerebrasMessage(EngineRequestPayload):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class CerebrasChatCreatePayload(EngineRequestPayload):
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
    prediction: dict[str, Any] | None = None
    presence_penalty: float | int | None = None
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = None
    reasoning_format: Literal["none", "parsed", "text_parsed", "raw", "hidden"] | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    service_tier: Literal["auto", "default", "flex", "priority"] | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    stream_options: dict[str, Any] | None = None
    temperature: float | int | None = Field(default=None, ge=0, le=2)
    tool_choice: Literal["none", "auto", "required"] | dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    top_logprobs: int | None = Field(default=None, ge=0)
    top_p: float | int | None = Field(default=None, ge=0, le=1)
    user: str | None = None


class CerebrasChatCreateCallOptions(EngineRequestPayload):
    extra_headers: dict[str, str] | None = None
    extra_query: dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None
    timeout: float | None = None


CerebrasChatRequest = EngineAPIRequest[
    CerebrasChatCreatePayload,
    CerebrasChatCreateCallOptions,
]


class CerebrasMixin:
    def cerebras_strip_prefix(self, model_name: str) -> str:
        if model_name.startswith("cerebras:"):
            return model_name.removeprefix("cerebras:")
        return model_name

    def cerebras_model_spec_for(self, model: str) -> CerebrasModelSpec:
        model_id = self.cerebras_strip_prefix(model)
        try:
            return CEREBRAS_MODEL_SPECS[model_id]
        except KeyError as e:
            msg = f"Unsupported Cerebras model: {model}"
            raise ValueError(msg) from e

    def cerebras_model_spec(self) -> CerebrasModelSpec:
        return self.cerebras_model_spec_for(self.model)

    def is_cerebras_reasoning_model(self) -> bool:
        return self.cerebras_model_spec().reasoning

    def api_max_context_tokens(self) -> int:
        return self.cerebras_model_spec().context_tokens

    def api_max_response_tokens(self) -> int:
        return self.cerebras_model_spec().response_tokens
