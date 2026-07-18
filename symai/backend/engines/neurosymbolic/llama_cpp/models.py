"""llama.cpp server wire models (OpenAI-compatible + llama extensions).

Locked against llama.cpp master at TESTED_LLAMA_CPP_COMMIT; see
docs/source/ENGINES/local_engine.md for the build we verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from symai.backend.request import EngineAPIRequest, EngineRequestPayload, EngineResponsePayload

TESTED_LLAMA_CPP_COMMIT = "4937ca83f"


@dataclass(frozen=True)
class LlamaCppModelSpec:
    # NOTE: llama.cpp serves a locally loaded model; context and response budgets come
    # from the running server (/props), not the registry. Reasoning depends on the
    # loaded model (thinking models emit reasoning_content); vision requires --mmproj.
    context_tokens: int
    response_tokens: int
    reasoning: bool
    vision: bool
    pricing: None = None


LLAMACPP_MODEL_SPECS = {
    "llamacpp": LlamaCppModelSpec(
        context_tokens=0,
        response_tokens=0,
        reasoning=True,
        vision=False,
    ),
}

SUPPORTED_MODELS = list(LLAMACPP_MODEL_SPECS)


def llamacpp_model_spec_for(model: str) -> LlamaCppModelSpec:
    model_id = model.removeprefix("llamacpp:") if model.startswith("llamacpp:") else model
    try:
        return LLAMACPP_MODEL_SPECS[model_id]
    except KeyError as e:
        msg = f"Unsupported llama.cpp model: {model}. Supported: {SUPPORTED_MODELS}"
        raise ValueError(msg) from e


class LlamaCppMessage(EngineRequestPayload):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, JsonValue]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, JsonValue]] | None = None


class LlamaCppPayload(EngineRequestPayload):
    messages: list[LlamaCppMessage]
    model: str | None = None
    temperature: float | int | None = None
    frequency_penalty: float | int | None = None
    presence_penalty: float | int | None = None
    top_p: float | int | None = None
    min_p: float | int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    top_k: int | None = None
    repeat_penalty: float | int | None = None
    logits_bias: dict[str, float | int] | None = None
    logprobs: bool | None = None
    grammar: str | None = None
    response_format: dict[str, JsonValue] | None = None
    tools: list[dict[str, JsonValue]] | None = None
    tool_choice: Literal["none", "auto", "required"] | dict[str, JsonValue] | None = None
    stream: bool | None = None
    stream_options: dict[Literal["include_usage"], bool] | None = Field(
        default=None,
        min_length=1,
        max_length=1,
    )


class LlamaCppOptions(EngineRequestPayload):
    extra_headers: dict[str, str] | None = None
    extra_query: dict[str, JsonValue] | None = None
    extra_body: dict[str, JsonValue] | None = None
    timeout: float | None = None


LlamaCppRequest = EngineAPIRequest[
    LlamaCppPayload,
    LlamaCppOptions,
]


class LlamaCppTokenizePayload(EngineRequestPayload):
    content: str


LlamaCppTokenizeRequest = EngineAPIRequest[
    LlamaCppTokenizePayload,
    LlamaCppOptions,
]


class LlamaCppDetokenizePayload(EngineRequestPayload):
    tokens: list[int]


LlamaCppDetokenizeRequest = EngineAPIRequest[
    LlamaCppDetokenizePayload,
    LlamaCppOptions,
]


class LlamaCppApplyTemplatePayload(EngineRequestPayload):
    messages: list[dict[str, JsonValue]]


LlamaCppApplyTemplateRequest = EngineAPIRequest[
    LlamaCppApplyTemplatePayload,
    LlamaCppOptions,
]


class LlamaCppPromptTokensDetails(EngineResponsePayload):
    cached_tokens: int | None = None


class LlamaCppUsage(EngineResponsePayload):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: LlamaCppPromptTokensDetails | None = None


class LlamaCppToolCallResult(EngineResponsePayload):
    id: str | None = None
    type: str
    function: dict[str, JsonValue] | None = None
    index: int | None = None


class LlamaCppResponseMessage(EngineResponsePayload):
    role: str
    # NOTE: tool-call messages may carry empty or missing content; a text answer
    # without content is a malformed response.
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[LlamaCppToolCallResult] | None = None

    @model_validator(mode="after")
    def require_content_or_tool_calls(self):
        if not self.content and not self.tool_calls:
            msg = "llama.cpp response message requires content when no tool calls are present."
            raise ValueError(msg)
        return self


class LlamaCppChoice(EngineResponsePayload):
    index: int
    message: LlamaCppResponseMessage
    finish_reason: str | None = None


class LlamaCppResponse(EngineResponsePayload):
    choices: list[LlamaCppChoice] = Field(min_length=1)
    # NOTE: MetadataTracker reads raw_output.usage for token accounting; streaming
    # requests force stream_options.include_usage so the final chunk carries it.
    usage: LlamaCppUsage


class LlamaCppTokenizeResponse(EngineResponsePayload):
    tokens: list[int]


class LlamaCppDetokenizeResponse(EngineResponsePayload):
    content: str


class LlamaCppApplyTemplateResponse(EngineResponsePayload):
    prompt: str
