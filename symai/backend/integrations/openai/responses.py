from typing import Literal

from pydantic import ConfigDict, Field, JsonValue

from symai.backend.integrations.base import StrictModel, TolerantModel

PATH = "/responses"


class ResponsesRequest(StrictModel):
    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue] = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        init=False
    )

    input: str | tuple[dict[str, JsonValue], ...]
    model: str
    background: bool | None = None
    context_management: tuple[dict[str, JsonValue], ...] | None = None
    conversation: str | dict[str, JsonValue] | None = None
    include: tuple[str, ...] | None = None
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    max_tool_calls: int | None = Field(default=None, gt=0)
    metadata: dict[str, str] | None = None
    moderation: dict[str, JsonValue] | None = None
    parallel_tool_calls: bool | None = None
    previous_response_id: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: Literal["in_memory", "24h"] | None = None
    reasoning: dict[str, JsonValue] | None = None
    safety_identifier: str | None = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None
    store: bool | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    text: dict[str, JsonValue] | None = None
    tool_choice: Literal["none", "auto", "required"] | dict[str, JsonValue] | None = None
    tools: tuple[dict[str, JsonValue], ...] | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    top_p: float | None = Field(default=None, ge=0, le=1)
    truncation: Literal["auto", "disabled"] | None = None
    user: str | None = None


class OutputContent(TolerantModel):
    type: str | None = None
    text: str | None = None


class SummaryContent(TolerantModel):
    type: str | None = None
    text: str | None = None


class OutputItem(TolerantModel):
    type: str
    content: tuple[OutputContent, ...] = ()
    summary: tuple[SummaryContent, ...] = ()
    arguments: str | None = None
    name: str | None = None
    call_id: str | None = None


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


class ResponsesResponse(TolerantModel):
    id: str | None = None
    object: str | None = None
    model: str | None = None
    output: tuple[OutputItem, ...]
    output_text: str | None = None
    usage: Usage
