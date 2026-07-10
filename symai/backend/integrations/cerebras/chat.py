from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AliasChoices, ConfigDict, Field, JsonValue

from symai.backend.integrations.base import StrictModel, TolerantModel

PATH = "/chat/completions"


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(StrictModel):
    role: Role
    content: str


class JsonSchemaSpec(StrictModel):
    model_config = ConfigDict(validate_by_alias=True, validate_by_name=True)

    name: str
    json_schema_body: JsonValue = Field(
        validation_alias=AliasChoices("json_schema_body", "schema"),
        serialization_alias="schema",
    )
    strict: bool = False


class ResponseFormat(StrictModel):
    type: Literal["json_schema"]
    json_schema: JsonSchemaSpec


class ReasoningEffort(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ChatRequest(StrictModel):
    messages: tuple[Message, ...] = Field(min_length=1)
    model: str
    temperature: float = Field(default=1.0, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    seed: int | None = None
    stop: str | Annotated[tuple[str, ...], Field(max_length=4)] | None = None
    reasoning_effort: ReasoningEffort | None = None
    response_format: ResponseFormat | None = None


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
