from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from symai.backend.providers.cerebras.spec import (
    CEREBRAS_MODEL_SPECS,
    CerebrasModel,
    ReasoningEffort,
)


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    role: Role
    content: str


class JsonSchemaSpec(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", populate_by_name=True)

    name: str
    json_schema_body: dict[str, Any] = Field(alias="schema")
    strict: bool = True


class CerebrasResponseFormat(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    type: Literal["json_schema"]
    json_schema: JsonSchemaSpec


class ChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    messages: tuple[Message, ...] = Field(min_length=1)
    model: CerebrasModel
    temperature: float = Field(default=1, ge=0, le=2)
    top_p: float = Field(default=1, ge=0, le=1)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    seed: int | None = None
    stop: tuple[str, ...] | None = None
    reasoning_effort: ReasoningEffort | None = None
    response_format: CerebrasResponseFormat | None = None

    @model_validator(mode="after")
    def _check_reasoning_effort_supported(self):
        if self.reasoning_effort is None:
            return self

        spec = CEREBRAS_MODEL_SPECS[self.model]

        if spec.reasoning is None:
            msg = f"model {self.model!r} does not support reasoning"
            raise ValueError(msg)

        if self.reasoning_effort not in spec.reasoning.efforts:
            msg = (
                f"model {self.model!r} does not support reasoning_effort={self.reasoning_effort!r}"
            )
            raise ValueError(msg)

        return self
