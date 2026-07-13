from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class Provider(StrEnum):
    OPENAI = "openai"
    CEREBRAS = "cerebras"
    DEEPSEEK = "deepseek"


class ImageDetail(StrEnum):
    AUTO = "auto"
    LOW = "low"
    HIGH = "high"


class MessageRole(StrEnum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"


class ContentType(StrEnum):
    TEXT = "text"
    IMAGE = "image"


class ResponseFormatType(StrEnum):
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


class ReasoningEffort(StrEnum):
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class ReasoningSummary(StrEnum):
    AUTO = "auto"
    CONCISE = "concise"
    DETAILED = "detailed"


class ReasoningFormat(StrEnum):
    NONE = "none"
    PARSED = "parsed"
    RAW = "raw"
    HIDDEN = "hidden"


class ReasoningField(StrEnum):
    ENABLED = "enabled"
    EFFORT = "effort"
    SUMMARY = "summary"
    FORMAT = "format"
    CLEAR = "clear"


class SamplingField(StrEnum):
    MAX_TOKENS = "max_tokens"
    TEMPERATURE = "temperature"
    TOP_P = "top_p"
    STOP = "stop"
    SEED = "seed"
    FREQUENCY_PENALTY = "frequency_penalty"
    PRESENCE_PENALTY = "presence_penalty"
    LOGPROBS = "logprobs"
    TOP_LOGPROBS = "top_logprobs"
    LOGIT_BIAS = "logit_bias"


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
JsonScalar = str | bool | int | FiniteFloat | None


class JsonEntry(FrozenModel):
    key: str
    value: JsonValue


class JsonArray(FrozenModel):
    type: Literal["array"] = "array"
    values: tuple[JsonValue, ...]

    def to_builtin(self) -> list[object]:
        return [_json_value_to_builtin(value) for value in self.values]


class JsonObject(FrozenModel):
    type: Literal["object"] = "object"
    entries: tuple[JsonEntry, ...]

    @model_validator(mode="after")
    def validate_unique_keys(self) -> Self:
        keys = tuple(entry.key for entry in self.entries)
        if len(keys) != len(set(keys)):
            msg = "JSON object keys must be unique"
            raise ValueError(msg)

        return self

    @classmethod
    def parse(cls, mapping: Mapping[str, object]) -> JsonObject:
        entries = tuple(
            JsonEntry(key=key, value=_parse_json_value(value)) for key, value in mapping.items()
        )
        return cls(entries=entries)

    def to_builtin(self) -> dict[str, object]:
        return {entry.key: _json_value_to_builtin(entry.value) for entry in self.entries}


JsonValue = JsonScalar | JsonObject | JsonArray

JsonEntry.model_rebuild(_types_namespace={"JsonValue": JsonValue})
JsonArray.model_rebuild(_types_namespace={"JsonValue": JsonValue})
JsonObject.model_rebuild(_types_namespace={"JsonValue": JsonValue})


def _parse_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            msg = "JSON numbers must be finite"
            raise ValueError(msg)

        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            msg = "JSON object keys must be strings"
            raise TypeError(msg)

        return JsonObject.parse(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return JsonArray(values=tuple(_parse_json_value(item) for item in value))

    msg = f"Unsupported JSON value: {type(value).__name__}"
    raise TypeError(msg)


def _json_value_to_builtin(value: JsonValue) -> object:
    if isinstance(value, JsonObject):
        return value.to_builtin()
    if isinstance(value, JsonArray):
        return value.to_builtin()
    return value


class TextContent(FrozenModel):
    type: Literal["text"] = "text"
    text: str


class ImageContent(FrozenModel):
    type: Literal["image"] = "image"
    url: str = Field(min_length=1)
    detail: ImageDetail | None = None


Content = Annotated[TextContent | ImageContent, Field(discriminator="type")]


class SystemMessage(FrozenModel):
    role: Literal["system"] = "system"
    content: tuple[TextContent, ...] = Field(min_length=1)


class DeveloperMessage(FrozenModel):
    role: Literal["developer"] = "developer"
    content: tuple[TextContent, ...] = Field(min_length=1)


class UserMessage(FrozenModel):
    role: Literal["user"] = "user"
    content: tuple[Content, ...] = Field(min_length=1)


class AssistantMessage(FrozenModel):
    role: Literal["assistant"] = "assistant"
    content: tuple[TextContent, ...] = ()
    reasoning: TextContent | None = None

    @model_validator(mode="after")
    def validate_content_or_reasoning(self) -> Self:
        if not self.content and self.reasoning is None:
            msg = "Assistant messages require content or reasoning"
            raise ValueError(msg)

        return self


Message = Annotated[
    SystemMessage | DeveloperMessage | UserMessage | AssistantMessage,
    Field(discriminator="role"),
]


class TextResponseFormat(FrozenModel):
    type: Literal["text"] = "text"


class JsonObjectResponseFormat(FrozenModel):
    type: Literal["json_object"] = "json_object"


class JsonSchemaResponseFormat(FrozenModel):
    type: Literal["json_schema"] = "json_schema"
    name: str = Field(min_length=1)
    schema_: JsonObject = Field(alias="schema", serialization_alias="schema")
    description: str | None = None
    strict: bool

    @property
    def schema(self) -> JsonObject:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.schema_


ResponseFormat = Annotated[
    TextResponseFormat | JsonObjectResponseFormat | JsonSchemaResponseFormat,
    Field(discriminator="type"),
]


class MetadataLabel(FrozenModel):
    key: str
    value: str


class LogitBias(FrozenModel):
    token: str
    value: float = Field(ge=-100, le=100, allow_inf_nan=False)


class ReasoningConfig(FrozenModel):
    enabled: bool | None = None
    effort: ReasoningEffort | None = None
    summary: ReasoningSummary | None = None
    format: ReasoningFormat | None = None
    clear: bool | None = None


class SamplingConfig(FrozenModel):
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2, allow_inf_nan=False)
    top_p: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    stop: tuple[str, ...] = ()
    seed: int | None = None
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2, allow_inf_nan=False)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2, allow_inf_nan=False)
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    logit_bias: tuple[LogitBias, ...] = ()


class LanguageModelRequest(FrozenModel):
    messages: tuple[Message, ...] = Field(min_length=1)
    response_format: ResponseFormat = TextResponseFormat()
    reasoning: ReasoningConfig | None = None
    sampling: SamplingConfig = SamplingConfig()
    user: str | None = None
    metadata: tuple[MetadataLabel, ...] = ()


class EmbeddingRequest(FrozenModel):
    inputs: tuple[str, ...] = Field(min_length=1)
    dimensions: int | None = Field(default=None, gt=0)
    user: str | None = None


class TokenUsage(FrozenModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cached_prompt_tokens: int = Field(default=0, ge=0)
    cache_miss_prompt_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    image_tokens: int = Field(default=0, ge=0)
    accepted_prediction_tokens: int = Field(default=0, ge=0)
    rejected_prediction_tokens: int = Field(default=0, ge=0)


class RateLimitMetadata(FrozenModel):
    limit_requests_day: int | None = Field(default=None, ge=0)
    limit_tokens_minute: int | None = Field(default=None, ge=0)
    remaining_requests_day: int | None = Field(default=None, ge=0)
    remaining_tokens_minute: int | None = Field(default=None, ge=0)
    reset_requests_day: NonNegativeFiniteFloat | None = None
    reset_tokens_minute: NonNegativeFiniteFloat | None = None


class ResponseMetadata(FrozenModel):
    provider: Provider
    model: str = Field(min_length=1)
    status_code: int = Field(ge=100, le=599)
    request_id: str | None = None
    retry_after: NonNegativeFiniteFloat | None = None
    response_id: str | None = None
    created_at: NonNegativeFiniteFloat | None = None
    system_fingerprint: str | None = None
    usage: TokenUsage | None = None
    rate_limit: RateLimitMetadata | None = None


class LanguageModelOutput(FrozenModel):
    index: int = Field(ge=0)
    message: AssistantMessage
    refusal: str | None = None
    finish_reason: FinishReason

    @property
    def text(self) -> str:
        return "".join(part.text for part in self.message.content)


class LanguageModelResponse(FrozenModel):
    outputs: tuple[LanguageModelOutput, ...] = Field(min_length=1)
    metadata: ResponseMetadata


class EmbeddingVector(FrozenModel):
    index: int = Field(ge=0)
    values: tuple[FiniteFloat, ...] = Field(min_length=1)

    @field_validator("values", mode="before")
    @classmethod
    def validate_float_values(cls, values: object) -> object:
        if isinstance(values, tuple) and any(type(value) is not float for value in values):
            msg = "Embedding values must be floats"
            raise TypeError(msg)

        return values


class EmbeddingResponse(FrozenModel):
    vectors: tuple[EmbeddingVector, ...] = Field(min_length=1)
    metadata: ResponseMetadata


class LanguageModelSpec(FrozenModel):
    context_tokens: int = Field(gt=0)
    response_tokens: int = Field(gt=0)
    message_roles: tuple[MessageRole, ...] = Field(min_length=1)
    content_types: tuple[ContentType, ...] = Field(min_length=1)
    response_formats: tuple[ResponseFormatType, ...] = Field(min_length=1)
    reasoning_fields: tuple[ReasoningField, ...] = ()
    reasoning_efforts: tuple[ReasoningEffort, ...] = ()
    reasoning_summaries: tuple[ReasoningSummary, ...] = ()
    reasoning_formats: tuple[ReasoningFormat, ...] = ()
    sampling_fields: tuple[SamplingField, ...] = ()
    vision: bool = False


class EmbeddingModelSpec(FrozenModel):
    context_tokens: int = Field(gt=0)
    dimensions: int = Field(gt=0)


class TransportConfig(FrozenModel):
    request_timeout: PositiveFiniteFloat = 600.0
    connect_timeout: PositiveFiniteFloat = 10.0
    connect_retries: int = Field(default=0, ge=0)


class ProviderEngineConfig(FrozenModel):
    provider: Provider
    model: str = Field(min_length=1)
    api_key: SecretStr
    transport: TransportConfig = TransportConfig()


class RuntimeConfig(FrozenModel):
    language_model: ProviderEngineConfig | None = None
    embedding: ProviderEngineConfig | None = None

    @model_validator(mode="after")
    def validate_nonempty(self) -> Self:
        if self.language_model is None and self.embedding is None:
            msg = "Runtime configuration requires at least one engine"
            raise ValueError(msg)

        return self
