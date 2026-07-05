from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field

from symai.backend.request import EngineRequestPayload

# https://api-docs.deepseek.com/quick_start/pricing
SUPPORTED_REASONING_MODELS = ["deepseek-reasoner", "deepseek-v4-flash", "deepseek-v4-pro"]


class DeepSeekMessage(EngineRequestPayload):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    prefix: bool | None = None
    reasoning_content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class DeepSeekChatCreatePayload(EngineRequestPayload):
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


class DeepSeekChatCreateCallOptions(EngineRequestPayload):
    extra_headers: dict[str, str] | None = None
    extra_query: dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None
    timeout: float | None = None


@dataclass(frozen=True)
class DeepSeekChatRequest:
    provider: str
    operation: str
    payload: DeepSeekChatCreatePayload
    call_options: DeepSeekChatCreateCallOptions | None = None

    def body(self) -> dict[str, Any]:
        return self.payload.model_dump(exclude_none=True)

    def kwargs(self) -> dict[str, Any]:
        values = self.body()
        if self.call_options is not None:
            values.update(self.call_options.model_dump(exclude_none=True))
        return values

    def body_with_extra(self) -> dict[str, Any]:
        body = self.body()
        if self.call_options is None:
            return body
        extra_body = self.call_options.model_dump(exclude_none=True).get("extra_body")
        if extra_body is None:
            return body
        return {**extra_body, **body}

    def request_options(self) -> dict[str, Any]:
        if self.call_options is None:
            return {}
        options = self.call_options.model_dump(exclude_none=True)
        values = {}
        if "extra_headers" in options:
            values["headers"] = options["extra_headers"]
        if "extra_query" in options:
            values["params"] = options["extra_query"]
        if "timeout" in options:
            values["timeout"] = options["timeout"]
        return values


class DeepSeekMixin:
    def api_max_context_tokens(self):
        if self.model in ("deepseek-v4-flash", "deepseek-v4-pro"):
            return 1_000_000
        if self.model == "deepseek-reasoner":
            return 64_000
        return None

    def api_max_response_tokens(self):
        if self.model in ("deepseek-v4-flash", "deepseek-v4-pro"):
            return 384_000
        if self.model == "deepseek-reasoner":
            return 64_000
        return None
