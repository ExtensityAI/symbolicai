from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, JsonValue

from symai.clients._models import ModelId, StrictModel, TolerantModel

PATH = "/responses"

ResponseModel = Literal[
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.4",
    "gpt-5.4-pro",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "o3-pro",
    "o3",
    "gpt-4.1",
    "gpt-4.1-mini",
]


class ItemStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"


class ResponseStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    CANCELLED = "cancelled"
    QUEUED = "queued"
    INCOMPLETE = "incomplete"


class ReasoningEffort(StrEnum):
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


@dataclass(frozen=True, slots=True)
class ReasoningSpec:
    efforts: tuple[ReasoningEffort, ...]


@dataclass(frozen=True, slots=True)
class ResponseModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: ReasoningSpec | None
    vision: bool = True


_REASONING = ReasoningSpec(tuple(ReasoningEffort))
MODEL_SPECS: dict[ResponseModel, ResponseModelSpec] = {
    "gpt-5.5": ResponseModelSpec(1_050_000, 128_000, reasoning=_REASONING),
    "gpt-5.5-pro": ResponseModelSpec(1_050_000, 128_000, reasoning=_REASONING),
    "gpt-5.4": ResponseModelSpec(1_050_000, 128_000, reasoning=_REASONING),
    "gpt-5.4-pro": ResponseModelSpec(1_050_000, 128_000, reasoning=_REASONING),
    "gpt-5.4-mini": ResponseModelSpec(400_000, 128_000, reasoning=_REASONING),
    "gpt-5.4-nano": ResponseModelSpec(400_000, 128_000, reasoning=_REASONING),
    "o3-pro": ResponseModelSpec(200_000, 100_000, reasoning=_REASONING),
    "o3": ResponseModelSpec(200_000, 100_000, reasoning=_REASONING),
    "gpt-4.1": ResponseModelSpec(1_047_576, 32_768, reasoning=None),
    "gpt-4.1-mini": ResponseModelSpec(1_047_576, 32_768, reasoning=None),
}


class ReasoningSummary(StrEnum):
    AUTO = "auto"
    CONCISE = "concise"
    DETAILED = "detailed"


class ServiceTier(StrEnum):
    AUTO = "auto"
    DEFAULT = "default"
    FLEX = "flex"
    SCALE = "scale"
    PRIORITY = "priority"


class Truncation(StrEnum):
    AUTO = "auto"
    DISABLED = "disabled"


class Verbosity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PromptCacheBreakpoint(StrictModel):
    mode: Literal["explicit"]


class InputText(StrictModel):
    type: Literal["input_text"]
    text: str
    prompt_cache_breakpoint: PromptCacheBreakpoint | None = None


class InputImage(StrictModel):
    type: Literal["input_image"]
    detail: Literal["low", "high", "auto", "original"] = "auto"
    file_id: str | None = None
    image_url: str | None = None
    prompt_cache_breakpoint: PromptCacheBreakpoint | None = None


class InputFile(StrictModel):
    type: Literal["input_file"]
    detail: Literal["auto", "low", "high"] | None = None
    file_data: str | None = None
    file_id: str | None = None
    file_url: str | None = None
    filename: str | None = None
    prompt_cache_breakpoint: PromptCacheBreakpoint | None = None


class InputAudio(StrictModel):
    type: Literal["input_audio"]
    data: str
    format: Literal["mp3", "wav"]


InputContent = Annotated[
    InputText | InputImage | InputFile | InputAudio,
    Field(discriminator="type"),
]


class InputMessage(StrictModel):
    role: Literal["user", "assistant", "system", "developer"]
    content: str | tuple[InputContent, ...]
    phase: Literal["commentary", "final_answer"] | None = None
    type: Literal["message"] = "message"


class Conversation(StrictModel):
    id: str


class ContextCompaction(StrictModel):
    type: Literal["compaction"]
    compact_threshold: int | None = Field(default=None, gt=0)


class PromptReference(StrictModel):
    id: str
    variables: dict[str, JsonValue] | None = None
    version: str | None = None


class PromptCacheOptions(StrictModel):
    ttl: Literal["in_memory", "24h"] | None = None


class ReasoningConfig(StrictModel):
    effort: ReasoningEffort | None = None
    summary: ReasoningSummary | None = None
    generate_summary: ReasoningSummary | None = None


class ResponseReasoningConfig(TolerantModel):
    effort: ReasoningEffort | None = None
    summary: ReasoningSummary | None = None
    generate_summary: ReasoningSummary | None = None


class TextFormat(StrictModel):
    type: Literal["text"]


class JsonObjectFormat(StrictModel):
    type: Literal["json_object"]


class JsonSchemaFormat(StrictModel):
    model_config = ConfigDict(validate_by_name=True, serialize_by_alias=True)

    type: Literal["json_schema"]
    name: str
    description: str | None = None
    schema_: JsonValue = Field(alias="schema")
    strict: bool | None = None


OutputFormat = Annotated[
    TextFormat | JsonObjectFormat | JsonSchemaFormat,
    Field(discriminator="type"),
]


class TextConfig(StrictModel):
    format: OutputFormat | None = None
    verbosity: Verbosity | None = None


class ResponseTextConfig(TolerantModel):
    format: OutputFormat | None = None
    verbosity: Verbosity | None = None


class ModerationConfig(StrictModel):
    type: str | None = None


Include = Literal[
    "message.input_image.image_url",
    "message.output_text.logprobs",
    "reasoning.encrypted_content",
]


class CreateResponseRequest(StrictModel):
    input: str | tuple[InputMessage, ...]
    model: ResponseModel | ModelId
    background: bool | None = None
    context_management: tuple[ContextCompaction, ...] | None = None
    conversation: str | Conversation | None = None
    include: tuple[Include, ...] | None = None
    instructions: str | tuple[InputMessage, ...] | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    metadata: dict[str, str] | None = None
    moderation: ModerationConfig | None = None
    previous_response_id: str | None = None
    prompt: PromptReference | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: Literal["in_memory", "24h"] | None = None
    prompt_cache_options: PromptCacheOptions | None = None
    reasoning: ReasoningConfig | None = None
    safety_identifier: str | None = None
    service_tier: ServiceTier | None = None
    store: bool | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    text: TextConfig | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    top_p: float | None = Field(default=None, ge=0, le=1)
    truncation: Truncation | None = None
    user: str | None = None


class FileCitation(StrictModel):
    type: Literal["file_citation"]
    file_id: str
    filename: str
    index: int


class URLCitation(StrictModel):
    type: Literal["url_citation"]
    url: str
    title: str
    start_index: int
    end_index: int


class ContainerFileCitation(StrictModel):
    type: Literal["container_file_citation"]
    container_id: str
    file_id: str
    filename: str
    start_index: int
    end_index: int


class FilePath(StrictModel):
    type: Literal["file_path"]
    file_id: str
    index: int


Annotation = Annotated[
    FileCitation | URLCitation | ContainerFileCitation | FilePath,
    Field(discriminator="type"),
]


class TopLogprob(TolerantModel):
    token: str
    logprob: float
    bytes: tuple[int, ...] | None = None


class TokenLogprob(TopLogprob):
    top_logprobs: tuple[TopLogprob, ...] = ()


class OutputText(TolerantModel):
    type: Literal["output_text"]
    text: str
    annotations: tuple[Annotation, ...] = ()
    logprobs: tuple[TokenLogprob, ...] = ()


class Refusal(TolerantModel):
    type: Literal["refusal"]
    refusal: str


OutputContent = Annotated[
    OutputText | Refusal,
    Field(discriminator="type"),
]


class OutputMessage(TolerantModel):
    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    status: ItemStatus
    content: tuple[OutputContent, ...]
    phase: Literal["commentary", "final_answer"] | None = None


class SummaryText(TolerantModel):
    type: Literal["summary_text"]
    text: str


class ReasoningText(TolerantModel):
    type: Literal["reasoning_text"]
    text: str


class ReasoningOutput(TolerantModel):
    id: str
    type: Literal["reasoning"]
    status: ItemStatus | None = None
    summary: tuple[SummaryText, ...] = ()
    content: tuple[ReasoningText, ...] = ()
    encrypted_content: str | None = None


class CompactionOutput(TolerantModel):
    id: str
    type: Literal["compaction"]
    encrypted_content: str


OutputItem = Annotated[
    OutputMessage | ReasoningOutput | CompactionOutput,
    Field(discriminator="type"),
]


class InputTokensDetails(TolerantModel):
    cached_tokens: int = 0


class OutputTokensDetails(TolerantModel):
    reasoning_tokens: int = 0


class Usage(TolerantModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: InputTokensDetails = InputTokensDetails()
    output_tokens_details: OutputTokensDetails = OutputTokensDetails()


class ResponseErrorCode(StrEnum):
    SERVER_ERROR = "server_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_PROMPT = "invalid_prompt"
    BIO_POLICY = "bio_policy"
    VECTOR_STORE_TIMEOUT = "vector_store_timeout"
    INVALID_IMAGE = "invalid_image"
    INVALID_IMAGE_FORMAT = "invalid_image_format"
    INVALID_BASE64_IMAGE = "invalid_base64_image"
    INVALID_IMAGE_URL = "invalid_image_url"
    IMAGE_TOO_LARGE = "image_too_large"
    IMAGE_TOO_SMALL = "image_too_small"
    IMAGE_PARSE_ERROR = "image_parse_error"
    IMAGE_CONTENT_POLICY_VIOLATION = "image_content_policy_violation"
    INVALID_IMAGE_MODE = "invalid_image_mode"
    IMAGE_FILE_TOO_LARGE = "image_file_too_large"
    UNSUPPORTED_IMAGE_MEDIA_TYPE = "unsupported_image_media_type"
    EMPTY_IMAGE_FILE = "empty_image_file"
    FAILED_TO_DOWNLOAD_IMAGE = "failed_to_download_image"
    IMAGE_FILE_NOT_FOUND = "image_file_not_found"


class ResponseError(TolerantModel):
    code: ResponseErrorCode
    message: str


class IncompleteDetails(TolerantModel):
    reason: Literal["max_output_tokens", "content_filter"] | None = None


class RetrieveResponseParams(StrictModel):
    include: tuple[Include, ...] | None = None


class ListInputItemsParams(StrictModel):
    after: str | None = None
    include: tuple[Include, ...] | None = None
    limit: int | None = Field(default=None, ge=1, le=100)
    order: Literal["asc", "desc"] | None = None


InputItem = InputMessage | OutputMessage | ReasoningOutput | CompactionOutput


class Response(TolerantModel):
    id: str
    object: Literal["response"]
    created_at: float
    status: ResponseStatus
    background: bool
    error: ResponseError | None
    incomplete_details: IncompleteDetails | None
    instructions: str | tuple[InputMessage, ...] | None
    max_output_tokens: int | None
    model: str
    output: tuple[OutputItem, ...]
    previous_response_id: str | None = None
    prompt: PromptReference | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: Literal["in_memory", "24h"] | None = None
    reasoning: ResponseReasoningConfig | None = None
    safety_identifier: str | None = None
    service_tier: ServiceTier | None = None
    store: bool
    temperature: float | None = None
    text: ResponseTextConfig | None = None
    top_logprobs: int | None = None
    top_p: float | None = None
    truncation: Truncation
    usage: Usage | None
    user: str | None = None
    metadata: dict[str, str]


class DeletedResponse(StrictModel):
    id: str
    object: Literal["response.deleted"]
    deleted: Literal[True]


class InputItemList(TolerantModel):
    object: Literal["list"]
    data: tuple[InputItem, ...]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool
