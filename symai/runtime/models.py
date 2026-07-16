from __future__ import annotations

from enum import StrEnum
from re import fullmatch
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)


class FrozenModel(BaseModel):
    # `hide_input_in_errors` keeps a rejected value out of the ValidationError. These models
    # carry prompts and provider payloads, and a validation error names the offending field
    # — so without this, passing a prompt where a container was expected writes the prompt
    # into any log that records the exception. The field, location, and rule are still
    # reported; only the value is withheld.
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", hide_input_in_errors=True)


def _normalize_provider_id(value: object) -> str:
    if not isinstance(value, str):
        msg = "Provider ID must be a string"
        raise ValueError(msg)

    normalized = value.lower()
    if fullmatch(r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?", normalized) is None:
        msg = "Provider ID must be a nonempty canonical identifier without whitespace"
        raise ValueError(msg)
    return normalized


ProviderId = Annotated[str, BeforeValidator(_normalize_provider_id)]


class ImageDetail(StrEnum):
    AUTO = "auto"
    LOW = "low"
    HIGH = "high"


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


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]

# Only meaningful for a provider reached over HTTP. Named for what it is so that a field
# holding one is obviously optional: see `ResponseMetadata.status_code`.
HttpStatusCode = Annotated[int, Field(ge=100, le=599)]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


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


class AssistantOutputMessage(FrozenModel):
    role: Literal["assistant"] = "assistant"
    content: tuple[TextContent, ...] = ()
    reasoning: TextContent | None = None


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
    json_schema: JsonValue
    description: str | None = None
    strict: bool


ResponseFormat = Annotated[
    TextResponseFormat | JsonObjectResponseFormat | JsonSchemaResponseFormat,
    Field(discriminator="type"),
]


class MetadataLabel(FrozenModel):
    key: str
    value: str


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


class LanguageModelRequest(FrozenModel):
    messages: tuple[Message, ...] = Field(min_length=1)
    response_format: ResponseFormat = TextResponseFormat()
    reasoning: ReasoningConfig | None = None
    sampling: SamplingConfig = SamplingConfig()
    user: str | None = None
    metadata: tuple[MetadataLabel, ...] = ()

    @model_validator(mode="after")
    def validate_unique_metadata_keys(self) -> Self:
        seen: set[str] = set()
        for label in self.metadata:
            if label.key in seen:
                msg = "Metadata label keys must be unique"
                raise ValueError(msg)
            seen.add(label.key)

        return self


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
    """Normalized facts about a successful provider call.

    Every field an integration cannot always supply is optional, because this is the
    contract a provider is normalized *into* — requiring something only an HTTP API has
    would make "an engine abstracts over its client" false for anything else. `status_code`
    is the field that made it false: it was required and range-checked to 100..599, so a
    binding driving a local binary could satisfy this model only by reporting an HTTP
    status it never received. `ErrorMetadata.status_code` was already optional, so the
    error path was transport-agnostic while the success path was not.
    """

    provider: ProviderId
    requested_model: str = Field(min_length=1)
    response_model: str | None = Field(default=None, min_length=1)
    status_code: HttpStatusCode | None = None
    request_id: str | None = None
    retry_after: NonNegativeFiniteFloat | None = None
    response_id: str | None = None
    created_at: NonNegativeFiniteFloat | None = None
    system_fingerprint: str | None = None
    usage: TokenUsage | None = None
    rate_limit: RateLimitMetadata | None = None


class LanguageModelOutput(FrozenModel):
    index: int = Field(ge=0)
    message: AssistantOutputMessage
    refusal: str | None = Field(default=None, min_length=1)
    finish_reason: FinishReason

    @model_validator(mode="after")
    def validate_content_reasoning_or_refusal(self) -> Self:
        has_content = any(part.text for part in self.message.content)
        has_reasoning = self.message.reasoning is not None and bool(self.message.reasoning.text)
        if (
            not has_content
            and not has_reasoning
            and self.refusal is None
            and self.finish_reason is not FinishReason.CONTENT_FILTER
        ):
            msg = (
                "Language model outputs require nonempty content, reasoning, refusal, "
                "or a content-filter finish reason"
            )
            raise ValueError(msg)

        return self

    @property
    def text(self) -> str:
        return "".join(part.text for part in self.message.content)


class LanguageModelResponse(FrozenModel):
    outputs: tuple[LanguageModelOutput, ...] = Field(min_length=1)
    metadata: ResponseMetadata

    def output_text(self, output_index: int = 0, /) -> str:
        """Return one output's text, selected by its own index rather than position.

        Raises:
            IndexError: if no output carries `output_index`.
        """
        for output in self.outputs:
            if output.index == output_index:
                return output.text

        msg = f"Language response did not contain output index {output_index}"
        raise IndexError(msg)

    @property
    def text(self) -> str:
        """Raw text of the first output, for the common single-output case.

        No normalization is applied; use `symai.decoding` to decode into a value.
        """
        return self.output_text()


class EmbeddingVector(FrozenModel):
    index: int = Field(ge=0)
    # `values` is validated entirely by the annotation: strict mode rejects str and bool,
    # and allow_inf_nan=False rejects inf and nan. A Python-level pre-scan over every
    # element costs more than the validation it would guard, at batch sizes where the
    # payload is millions of floats.
    values: tuple[FiniteFloat, ...] = Field(min_length=1)


class EmbeddingResponse(FrozenModel):
    vectors: tuple[EmbeddingVector, ...] = Field(min_length=1)
    metadata: ResponseMetadata


class LanguageModelSpec(FrozenModel):
    response_tokens: int = Field(gt=0)
    reasoning_fields: tuple[ReasoningField, ...] = ()
    reasoning_efforts: tuple[ReasoningEffort, ...] = ()
    reasoning_summaries: tuple[ReasoningSummary, ...] = ()
    reasoning_formats: tuple[ReasoningFormat, ...] = ()
    sampling_fields: tuple[SamplingField, ...] = ()
    vision: bool = False


class EmbeddingModelSpec(FrozenModel):
    dimensions: int = Field(gt=0)
