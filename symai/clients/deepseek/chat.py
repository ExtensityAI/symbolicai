from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from symai.clients._models import StrictModel, TolerantModel

PATH = "/chat/completions"


class SystemMessage(StrictModel):
    role: Literal["system"]
    content: str
    name: str | None = None


class UserMessage(StrictModel):
    role: Literal["user"]
    content: str
    name: str | None = None


class AssistantMessage(StrictModel):
    role: Literal["assistant"]
    content: str | None
    name: str | None = None


Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage,
    Field(discriminator="role"),
]


class ThinkingType(StrEnum):
    ENABLED = "enabled"
    DISABLED = "disabled"


class Thinking(StrictModel):
    type: ThinkingType = ThinkingType.ENABLED


class TextResponseFormat(StrictModel):
    type: Literal["text"]


class JsonObjectResponseFormat(StrictModel):
    type: Literal["json_object"]


ResponseFormat = Annotated[
    TextResponseFormat | JsonObjectResponseFormat,
    Field(discriminator="type"),
]


_StopSequences = Annotated[tuple[str, ...], Field(max_length=16)]
_UserID = Annotated[str, Field(max_length=512, pattern=r"^[a-zA-Z0-9_-]+$")]


class CreateChatCompletionRequest(StrictModel):
    messages: tuple[Message, ...] = Field(min_length=1)
    model: str
    thinking: Thinking | None = None
    reasoning_effort: Literal["high", "max"] | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    response_format: ResponseFormat | None = None
    stop: str | _StopSequences | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    user_id: _UserID | None = None


class ResponseMessage(TolerantModel):
    role: str
    content: str | None
    reasoning_content: str | None = None


class TopLogprob(TolerantModel):
    token: str
    logprob: float
    bytes: tuple[int, ...] | None


class TokenLogprob(TopLogprob):
    top_logprobs: tuple[TopLogprob, ...]


class Logprobs(TolerantModel):
    content: tuple[TokenLogprob, ...] | None
    reasoning_content: tuple[TokenLogprob, ...] | None = None


class Choice(TolerantModel):
    finish_reason: str
    index: int
    message: ResponseMessage
    logprobs: Logprobs | None = None


class CompletionTokensDetails(TolerantModel):
    reasoning_tokens: int | None = None


class Usage(TolerantModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    prompt_cache_hit_tokens: int | None = None
    prompt_cache_miss_tokens: int | None = None
    completion_tokens_details: CompletionTokensDetails | None = None


class ChatCompletion(TolerantModel):
    id: str
    choices: tuple[Choice, ...]
    created: int
    model: str
    object: str
    system_fingerprint: str | None = None
    usage: Usage
