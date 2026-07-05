from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field

from symai.backend.request import EngineAPIRequest, EngineRequestPayload


@dataclass(frozen=True)
class GroqModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: bool
    reasoning_efforts: tuple[str, ...]
    default_reasoning_effort: str


GROQ_MODEL_SPECS = {
    "openai/gpt-oss-120b": GroqModelSpec(
        context_tokens=131_072,
        response_tokens=32_766,
        reasoning=True,
        reasoning_efforts=("low", "medium", "high"),
        default_reasoning_effort="low",
    ),
    "openai/gpt-oss-20b": GroqModelSpec(
        context_tokens=131_072,
        response_tokens=32_768,
        reasoning=True,
        reasoning_efforts=("low", "medium", "high"),
        default_reasoning_effort="low",
    ),
}

SUPPORTED_REASONING_MODELS = [
    f"groq:{model}" for model, spec in GROQ_MODEL_SPECS.items() if spec.reasoning
]
SUPPORTED_GROQ_MODELS = [f"groq:{model}" for model in GROQ_MODEL_SPECS]
GROQ_UNSUPPORTED_REQUEST_KWARGS = frozenset(
    {
        "logprobs",
        "logit_bias",
        "top_logprobs",
        "search_settings",
        "stream",
        "stream_options",
    }
)


class GroqMessage(EngineRequestPayload):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class GroqChatCreatePayload(EngineRequestPayload):
    messages: list[GroqMessage]
    model: str
    seed: int | None = None
    max_completion_tokens: int | None = Field(default=None, gt=0)
    stop: str | list[str] | None = None
    temperature: float | int | None = Field(default=None, ge=0, le=2)
    frequency_penalty: float | int | None = None
    presence_penalty: float | int | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    service_tier: Literal["on_demand", "flex", "auto"] | None = None
    top_p: float | int | None = Field(default=None, ge=0, le=1)
    n: int | None = Field(default=None, gt=0)
    tools: list[dict[str, Any]] | None = None
    tool_choice: Literal["none", "auto", "required"] | dict[str, Any] | None = None
    response_format: dict[Literal["type"], Literal["text", "json_object"]] | None = Field(
        default=None,
        min_length=1,
        max_length=1,
    )
    parallel_tool_calls: bool | None = None
    user: str | None = None


class GroqChatCreateCallOptions(EngineRequestPayload):
    extra_headers: dict[str, str] | None = None
    extra_query: dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None
    timeout: float | None = None


class GroqChatCreateAliases(EngineRequestPayload):
    max_tokens: int | None = Field(default=None, gt=0)


GroqChatRequest = EngineAPIRequest[GroqChatCreatePayload, GroqChatCreateCallOptions]


class GroqMixin:
    def groq_strip_prefix(self, model_name: str) -> str:
        if model_name.startswith("groq:"):
            return model_name.removeprefix("groq:")
        return model_name

    def groq_model_spec_for(self, model: str) -> GroqModelSpec:
        model_id = self.groq_strip_prefix(model)
        try:
            return GROQ_MODEL_SPECS[model_id]
        except KeyError as e:
            msg = f"Unsupported Groq model: {model}"
            raise ValueError(msg) from e

    def groq_model_spec(self) -> GroqModelSpec:
        return self.groq_model_spec_for(self.model)

    def groq_default_reasoning_effort(self) -> str:
        return self.groq_model_spec().default_reasoning_effort

    def groq_supports_reasoning_effort(self, effort: str | None) -> bool:
        if effort is None:
            return True
        return effort in self.groq_model_spec().reasoning_efforts

    def api_max_context_tokens(self) -> int:
        return self.groq_model_spec().context_tokens

    def api_max_response_tokens(self) -> int:
        return self.groq_model_spec().response_tokens
