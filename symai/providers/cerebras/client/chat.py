from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, JsonValue

from symai.providers._client.models import ModelId, StrictModel, TolerantModel

PATH = "/chat/completions"

Model = Literal["gpt-oss-120b", "gemma-4-31b", "zai-glm-4.7"]


class TextContentPart(StrictModel):
    type: Literal["text"]
    text: str


class ImageURL(StrictModel):
    url: str


class ImageContentPart(StrictModel):
    type: Literal["image_url"]
    image_url: ImageURL


class SystemMessage(StrictModel):
    role: Literal["system"]
    content: str | tuple[TextContentPart, ...]
    name: str | None = None


class DeveloperMessage(StrictModel):
    role: Literal["developer"]
    content: str | tuple[TextContentPart, ...]
    name: str | None = None


class UserMessage(StrictModel):
    role: Literal["user"]
    content: str | tuple[TextContentPart | ImageContentPart, ...]
    name: str | None = None


class AssistantMessage(StrictModel):
    role: Literal["assistant"]
    content: str | tuple[TextContentPart, ...] | None = None
    reasoning: str | None = None
    name: str | None = None


Message = Annotated[
    SystemMessage | DeveloperMessage | UserMessage | AssistantMessage,
    Field(discriminator="role"),
]


class ReasoningEffort(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class ReasoningSpec:
    efforts: tuple[ReasoningEffort, ...]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: ReasoningSpec | None


MODEL_SPECS: dict[Model, ModelSpec] = {
    "gpt-oss-120b": ModelSpec(
        131_072,
        40_000,
        reasoning=ReasoningSpec(
            (
                ReasoningEffort.LOW,
                ReasoningEffort.MEDIUM,
                ReasoningEffort.HIGH,
            )
        ),
    ),
    "gemma-4-31b": ModelSpec(
        131_072,
        40_000,
        reasoning=ReasoningSpec(
            (
                ReasoningEffort.LOW,
                ReasoningEffort.MEDIUM,
                ReasoningEffort.HIGH,
            )
        ),
    ),
    "zai-glm-4.7": ModelSpec(
        131_072,
        40_000,
        reasoning=ReasoningSpec(
            (
                ReasoningEffort.NONE,
                ReasoningEffort.LOW,
                ReasoningEffort.MEDIUM,
                ReasoningEffort.HIGH,
            )
        ),
    ),
}


class ReasoningFormat(StrEnum):
    NONE = "none"
    PARSED = "parsed"
    RAW = "raw"
    HIDDEN = "hidden"


class ServiceTier(StrEnum):
    AUTO = "auto"
    DEFAULT = "default"
    FLEX = "flex"
    PRIORITY = "priority"


class Prediction(StrictModel):
    type: Literal["content"]
    content: str | tuple[TextContentPart, ...]


class TextResponseFormat(StrictModel):
    type: Literal["text"]


class JsonObjectResponseFormat(StrictModel):
    type: Literal["json_object"]


class JsonSchemaSpec(StrictModel):
    model_config = ConfigDict(validate_by_name=True, serialize_by_alias=True)

    name: str
    description: str | None = None
    body: Annotated[JsonValue | None, Field(alias="schema")] = None
    """The JSON Schema that the model's response must satisfy."""
    strict: bool = False


class JsonSchemaResponseFormat(StrictModel):
    type: Literal["json_schema"]
    json_schema: JsonSchemaSpec


ResponseFormat = Annotated[
    TextResponseFormat | JsonObjectResponseFormat | JsonSchemaResponseFormat,
    Field(discriminator="type"),
]


_LogitBiasValue = Annotated[
    float,
    Field(ge=-100, le=100, allow_inf_nan=False),
]
_PositiveCompletionTokens = Annotated[int, Field(gt=0)]
_StopSequence = Annotated[tuple[str, ...], Field(max_length=4)]


class CreateChatCompletionRequest(StrictModel):
    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue] = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        init=False
    )

    messages: tuple[Message, ...] = Field(min_length=1)
    model: Model | ModelId
    clear_thinking: bool | None = None
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    logit_bias: dict[str, _LogitBiasValue] | None = None
    logprobs: bool | None = None
    max_completion_tokens: Literal[-1] | _PositiveCompletionTokens | None = None
    prediction: Prediction | None = None
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    prompt_cache_key: str | None = Field(default=None, max_length=1_024)
    reasoning_effort: ReasoningEffort | None = None
    reasoning_format: ReasoningFormat | None = None
    response_format: ResponseFormat | None = None
    seed: int | None = None
    service_tier: ServiceTier | None = None
    stop: str | _StopSequence | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    top_p: float | None = Field(default=None, ge=0, le=1)
    user: str | None = None


class PromptTokensDetails(TolerantModel):
    cached_tokens: int | None = None


class CompletionTokensDetails(TolerantModel):
    accepted_prediction_tokens: int | None = None
    rejected_prediction_tokens: int | None = None
    reasoning_tokens: int | None = None


class Usage(TolerantModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    image_tokens: int | None = None
    prompt_tokens_details: PromptTokensDetails | None = None
    completion_tokens_details: CompletionTokensDetails | None = None


class TimeInfo(TolerantModel):
    queue_time: float | None = None
    prompt_time: float | None = None
    completion_time: float | None = None
    total_time: float | None = None
    created: float | None = None


class ResponseMessage(TolerantModel):
    role: str | None = None
    content: str | None = None
    reasoning: str | None = None


class Choice(TolerantModel):
    finish_reason: str | None = None
    index: int | None = None
    message: ResponseMessage | None = None
    logprobs: dict[str, JsonValue] | None = None
    reasoning_logprobs: dict[str, JsonValue] | None = None


class ChatCompletion(TolerantModel):
    id: str | None = None
    choices: tuple[Choice, ...] | None = None
    created: int | None = None
    model: str | None = None
    object: str | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None
    service_tier_used: str | None = None
    usage: Usage | None = None
    time_info: TimeInfo | None = None
