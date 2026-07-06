from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from symai.backend.request import EngineAPIRequest, EngineRequestPayload

# https://api-docs.deepseek.com/quick_start/pricing


@dataclass(frozen=True)
class DeepSeekModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: bool
    vision: bool


DEEPSEEK_MODEL_SPECS = {
    "deepseek-v4-flash": DeepSeekModelSpec(
        context_tokens=1_000_000,
        response_tokens=384_000,
        reasoning=True,
        vision=False,
    ),
    "deepseek-v4-pro": DeepSeekModelSpec(
        context_tokens=1_000_000,
        response_tokens=384_000,
        reasoning=True,
        vision=False,
    ),
}

SUPPORTED_MODELS = list(DEEPSEEK_MODEL_SPECS)


class DeepSeekMessage(EngineRequestPayload):
    role: Literal["system", "user", "assistant", "tool"]
    # NOTE: DeepSeek returns content=null on assistant tool-call messages, so null must
    # round-trip when replaying conversations. exclude_none omits it from the wire body.
    content: str | None = None
    name: str | None = None
    prefix: bool | None = None
    reasoning_content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


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
    tools: list[dict[str, Any]] | None = None
    tool_choice: Literal["none", "auto", "required"] | dict[str, Any] | None = None
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
    extra_query: dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None
    timeout: float | None = None


DeepSeekRequest = EngineAPIRequest[
    DeepSeekPayload,
    DeepSeekOptions,
]


class DeepSeekResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    choices: list[dict[str, Any]] = Field(min_length=1)
    # NOTE: MetadataTracker reads raw_output.usage for token accounting; the API always
    # returns it (streaming requests force stream_options.include_usage).
    usage: dict[str, Any]

    @model_validator(mode="after")
    def require_message_content(self):
        for choice in self.choices:
            message = choice.get("message")
            if not isinstance(message, dict):
                msg = "DeepSeek response choice.message is required."
                raise ValueError(msg)
            if "content" not in message:
                msg = "DeepSeek response choice.message.content is required."
                raise ValueError(msg)
        return self


class DeepSeekMixin:
    def deepseek_model_spec_for(self, model: str) -> DeepSeekModelSpec:
        try:
            return DEEPSEEK_MODEL_SPECS[model]
        except KeyError as e:
            msg = f"Unsupported DeepSeek model: {model}"
            raise ValueError(msg) from e

    def deepseek_model_spec(self) -> DeepSeekModelSpec:
        return self.deepseek_model_spec_for(self.model)

    def api_max_context_tokens(self) -> int:
        return self.deepseek_model_spec().context_tokens
