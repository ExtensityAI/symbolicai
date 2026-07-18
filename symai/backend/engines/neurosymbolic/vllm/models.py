"""vLLM server wire models (OpenAI-compatible).

Locked against vLLM at TESTED_VLLM_COMMIT; see docs/source/ENGINES/local_engine.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from symai.backend.request import EngineAPIRequest, EngineRequestPayload, EngineResponsePayload

TESTED_VLLM_COMMIT = "595562651a5a4539ffa910d8570c08fb5169bdc9"


@dataclass(frozen=True)
class VLLMModelSpec:
    # NOTE: vLLM serves a locally loaded HF model; context comes from the running
    # server (/v1/models max_model_len), reasoning from --reasoning-parser, and vision
    # from the loaded model — none of which the registry can know ahead of time.
    context_tokens: int
    response_tokens: int
    reasoning: bool
    vision: bool
    pricing: None = None


VLLM_MODEL_SPECS = {
    "vllm": VLLMModelSpec(
        context_tokens=0,
        response_tokens=0,
        reasoning=True,
        vision=False,
    ),
}

SUPPORTED_MODELS = list(VLLM_MODEL_SPECS)


def vllm_model_spec_for(model: str) -> VLLMModelSpec:
    model_id = model.removeprefix("vllm:") if model.startswith("vllm:") else model
    try:
        return VLLM_MODEL_SPECS[model_id]
    except KeyError as e:
        msg = f"Unsupported vLLM model: {model}. Supported: {SUPPORTED_MODELS}"
        raise ValueError(msg) from e


class VLLMMessage(EngineRequestPayload):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, JsonValue]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, JsonValue]] | None = None


class VLLMPayload(EngineRequestPayload):
    messages: list[VLLMMessage]
    model: str | None = None
    temperature: float | int | None = None
    top_p: float | int | None = None
    top_k: int | None = None
    frequency_penalty: float | int | None = None
    presence_penalty: float | int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    logprobs: bool | None = None
    logit_bias: dict[str, float | int] | None = None
    response_format: dict[str, JsonValue] | None = None
    tools: list[dict[str, JsonValue]] | None = None
    tool_choice: Literal["none", "auto", "required"] | dict[str, JsonValue] | None = None
    stream: bool | None = None
    stream_options: dict[Literal["include_usage"], bool] | None = Field(
        default=None,
        min_length=1,
        max_length=1,
    )


class VLLMOptions(EngineRequestPayload):
    extra_headers: dict[str, str] | None = None
    extra_query: dict[str, JsonValue] | None = None
    extra_body: dict[str, JsonValue] | None = None
    timeout: float | None = None


VLLMRequest = EngineAPIRequest[
    VLLMPayload,
    VLLMOptions,
]


class VLLMTokenizePayload(EngineRequestPayload):
    model: str
    prompt: str


VLLMTokenizeRequest = EngineAPIRequest[
    VLLMTokenizePayload,
    VLLMOptions,
]


class VLLMUsage(EngineResponsePayload):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: dict[str, JsonValue] | None = None


class VLLMToolCallResult(EngineResponsePayload):
    id: str | None = None
    type: str
    function: dict[str, JsonValue] | None = None
    index: int | None = None


class VLLMResponseMessage(EngineResponsePayload):
    role: str
    # NOTE: tool-call messages may carry empty or missing content; a text answer
    # without content is a malformed response.
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[VLLMToolCallResult] | None = None

    @model_validator(mode="after")
    def require_content_or_tool_calls(self):
        if not self.content and not self.tool_calls:
            msg = "vLLM response message requires content when no tool calls are present."
            raise ValueError(msg)
        return self


class VLLMChoice(EngineResponsePayload):
    index: int
    message: VLLMResponseMessage
    finish_reason: str | None = None


class VLLMResponse(EngineResponsePayload):
    choices: list[VLLMChoice] = Field(min_length=1)
    # NOTE: MetadataTracker reads raw_output.usage for token accounting; streaming
    # requests force stream_options.include_usage so the final chunk carries it.
    usage: VLLMUsage


class VLLMTokenizeResponse(EngineResponsePayload):
    count: int
    tokens: list[int]
    max_model_len: int | None = None


class VLLMModelInfo(EngineResponsePayload):
    id: str
    max_model_len: int | None = None


class VLLMModelsResponse(EngineResponsePayload):
    data: list[VLLMModelInfo]
