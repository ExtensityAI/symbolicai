from enum import StrEnum
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from symai.backend.integrations.base import StrictModel, TolerantModel
from symai.backend.integrations.cerebras.client.spec import (
    MODEL_SPECS,
    Model,
    ReasoningEffort,
)


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(StrictModel):
    role: Role
    content: str


class JsonSchemaSpec(StrictModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    json_schema_body: dict[str, Any] = Field(alias="schema")
    strict: bool = True


class ResponseFormat(StrictModel):
    type: Literal["json_schema"]
    json_schema: JsonSchemaSpec


class ChatRequest(StrictModel):
    messages: tuple[Message, ...] = Field(min_length=1)
    model: Model
    temperature: float = Field(default=1.0, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    seed: int | None = None
    stop: tuple[str, ...] | None = None
    reasoning_effort: ReasoningEffort | None = None
    response_format: ResponseFormat | None = None

    @model_validator(mode="after")
    def _check_reasoning_effort_supported(self):
        if self.reasoning_effort is None:
            return self

        spec = MODEL_SPECS.get(self.model)

        if spec is None:
            msg = f"model {self.model!r} has no registered spec"
            raise ValueError(msg)

        if spec.reasoning is None:
            msg = f"model {self.model!r} does not support reasoning"
            raise ValueError(msg)

        if self.reasoning_effort not in spec.reasoning.efforts:
            msg = (
                f"model {self.model!r} does not support reasoning_effort={self.reasoning_effort!r}"
            )
            raise ValueError(msg)

        return self


class PromptTokensDetails(TolerantModel):
    cached_tokens: int | None = None


class CompletionTokensDetails(TolerantModel):
    accepted_prediction_tokens: int | None = None
    rejected_prediction_tokens: int | None = None
    reasoning_tokens: int | None = None


class Usage(TolerantModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    image_tokens: int | None = None
    prompt_tokens_details: PromptTokensDetails | None = None
    completion_tokens_details: CompletionTokensDetails | None = None


class ResponseMessage(TolerantModel):
    role: str
    content: str | None = None
    reasoning: str | None = None


class Choice(TolerantModel):
    index: int
    message: ResponseMessage
    finish_reason: str | None = None


class ChatResponse(TolerantModel):
    choices: tuple[Choice, ...]
    usage: Usage
