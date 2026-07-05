from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field

from symai.backend.request import EngineAPIRequest, EngineRequestPayload


@dataclass(frozen=True)
class OpenAIModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: bool
    vision: bool = True
    pro: bool = False
    tokenizer: str = "o200k_base"


OPENAI_MODEL_SPECS = {
    "gpt-5.5": OpenAIModelSpec(
        context_tokens=1_050_000,
        response_tokens=128_000,
        reasoning=True,
    ),
    "gpt-5.5-pro": OpenAIModelSpec(
        context_tokens=1_050_000,
        response_tokens=128_000,
        reasoning=True,
        pro=True,
    ),
    "gpt-5.4": OpenAIModelSpec(
        context_tokens=1_050_000,
        response_tokens=128_000,
        reasoning=True,
    ),
    "gpt-5.4-pro": OpenAIModelSpec(
        context_tokens=1_050_000,
        response_tokens=128_000,
        reasoning=True,
        pro=True,
    ),
    "gpt-5.4-mini": OpenAIModelSpec(
        context_tokens=400_000,
        response_tokens=128_000,
        reasoning=True,
    ),
    "gpt-5.4-nano": OpenAIModelSpec(
        context_tokens=400_000,
        response_tokens=128_000,
        reasoning=True,
    ),
    "o3-pro": OpenAIModelSpec(
        context_tokens=200_000,
        response_tokens=100_000,
        reasoning=True,
        pro=True,
    ),
    "o3": OpenAIModelSpec(
        context_tokens=200_000,
        response_tokens=100_000,
        reasoning=True,
    ),
    "gpt-4.1": OpenAIModelSpec(
        context_tokens=1_047_576,
        response_tokens=32_768,
        reasoning=False,
    ),
    "gpt-4.1-mini": OpenAIModelSpec(
        context_tokens=1_047_576,
        response_tokens=32_768,
        reasoning=False,
    ),
}

SUPPORTED_CHAT_MODELS = [model for model, spec in OPENAI_MODEL_SPECS.items() if not spec.reasoning]
SUPPORTED_REASONING_MODELS = [model for model, spec in OPENAI_MODEL_SPECS.items() if spec.reasoning]
SUPPORTED_OPENAI_MODELS = [f"openai:{model}" for model in OPENAI_MODEL_SPECS]


class OpenAIResponsesCreatePayload(EngineRequestPayload):
    background: bool | None = None
    context_management: list[dict[str, Any]] | None = None
    conversation: str | dict[str, Any] | None = None
    include: list[str] | None = None
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    max_tool_calls: int | None = Field(default=None, gt=0)
    metadata: dict[str, str] | None = None
    model: str
    moderation: dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    previous_response_id: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: Literal["in_memory", "24h"] | None = None
    reasoning: dict[str, Any] | None = None
    safety_identifier: str | None = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None
    store: bool | None = None
    stream: bool | None = None
    stream_options: dict[Literal["include_obfuscation"], bool] | None = Field(
        default=None,
        min_length=1,
        max_length=1,
    )
    temperature: float | int | None = Field(default=None, ge=0, le=2)
    text: dict[str, Any] | None = None
    tool_choice: Literal["none", "auto", "required"] | dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    top_p: float | int | None = Field(default=None, ge=0, le=1)
    truncation: Literal["auto", "disabled"] | None = None
    user: str | None = None


class OpenAIResponsesCreateCallOptions(EngineRequestPayload):
    extra_headers: dict[str, str] | None = None
    extra_query: dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None
    timeout: float | None = None


OpenAIResponsesCreateRequest = EngineAPIRequest[
    OpenAIResponsesCreatePayload,
    OpenAIResponsesCreateCallOptions,
]


class OpenAIMixin:
    def openai_model_spec_for(self, model: str) -> OpenAIModelSpec:
        try:
            return OPENAI_MODEL_SPECS[model]
        except KeyError as e:
            msg = f"Unsupported model: {model}"
            raise ValueError(msg) from e

    def openai_model_spec(self) -> OpenAIModelSpec:
        return self.openai_model_spec_for(self.model)

    def is_openai_reasoning_model(self) -> bool:
        return self.openai_model_spec().reasoning

    def is_openai_pro_model(self) -> bool:
        return self.openai_model_spec().pro

    def supports_openai_vision(self) -> bool:
        return self.openai_model_spec().vision

    def openai_tokenizer_name(self) -> str:
        return self.openai_model_spec().tokenizer

    def api_max_context_tokens(self):
        return self.openai_model_spec().context_tokens

    def api_max_response_tokens(self):
        return self.openai_model_spec().response_tokens

    def api_embedding_dims(self):
        if self.model == "text-embedding-ada-002":
            return 1_536
        if self.model == "text-embedding-3-small":
            return 1_536
        if self.model == "text-embedding-3-large":
            return 3_072
        msg = f"Unsupported model: {self.model}"
        raise ValueError(msg)
