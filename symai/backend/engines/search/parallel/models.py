"""Parallel Search / Extract / Task API wire models (GA v1 endpoints).

Locked against the official Parallel OpenAPI specs:
https://docs.parallel.ai/api-reference/search-api/search
https://docs.parallel.ai/api-reference/extract-api/extract
https://docs.parallel.ai/api-reference/task-api/create-task-run
https://docs.parallel.ai/api-reference/task-api/get-task-run-result
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from symai.backend.request import EngineAPIRequest, EngineRequestPayload, EngineResponsePayload

API_PINNED = "2026-07-18"

PARALLEL_API_BASE = "https://api.parallel.ai"
PARALLEL_SEARCH_PATH = "/v1/search"
PARALLEL_EXTRACT_PATH = "/v1/extract"
PARALLEL_TASK_RUNS_PATH = "/v1/tasks/runs"


class ParallelSourcePolicy(EngineRequestPayload):
    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None
    after_date: str | None = None


class ParallelExcerptSettings(EngineRequestPayload):
    max_chars_per_result: int | None = Field(default=None, gt=0)


class ParallelFetchPolicy(EngineRequestPayload):
    max_age_seconds: int | None = Field(default=None, ge=0)
    timeout_seconds: int | None = Field(default=None, gt=0)
    disable_cache_fallback: bool | None = None


class ParallelSearchAdvancedSettings(EngineRequestPayload):
    source_policy: ParallelSourcePolicy | None = None
    fetch_policy: ParallelFetchPolicy | None = None
    excerpt_settings: ParallelExcerptSettings | None = None
    location: str | None = None
    max_results: int | None = Field(default=None, gt=0)


class ParallelSearchRequest(EngineRequestPayload):
    mode: Literal["turbo", "basic", "advanced"] = "advanced"
    objective: str | None = None
    search_queries: list[str] = Field(min_length=1)
    max_chars_total: int | None = Field(default=None, gt=0)
    advanced_settings: ParallelSearchAdvancedSettings | None = None


class ParallelFullContentSettings(EngineRequestPayload):
    max_chars_per_result: int | None = Field(default=None, gt=0)


class ParallelExtractAdvancedSettings(EngineRequestPayload):
    fetch_policy: ParallelFetchPolicy | None = None
    excerpt_settings: ParallelExcerptSettings | None = None
    full_content: bool | ParallelFullContentSettings = False


class ParallelExtractRequest(EngineRequestPayload):
    urls: list[str] = Field(min_length=1)
    objective: str | None = None
    search_queries: list[str] | None = None
    max_chars_total: int | None = Field(default=None, gt=0)
    advanced_settings: ParallelExtractAdvancedSettings | None = None


class ParallelTaskOutputSchema(EngineRequestPayload):
    type: Literal["json", "text", "auto"]
    json_schema: dict[str, JsonValue] | None = None
    description: str | None = None


class ParallelTaskSpec(EngineRequestPayload):
    output_schema: ParallelTaskOutputSchema | str | None = None
    input_schema: dict[str, JsonValue] | None = None


class ParallelMCPServer(EngineRequestPayload):
    type: Literal["url"]
    url: str
    name: str | None = None
    headers: dict[str, str] | None = None
    allowed_tools: list[str] | None = None


class ParallelTaskRunCreateRequest(EngineRequestPayload):
    processor: str
    input: str | dict[str, JsonValue]
    task_spec: ParallelTaskSpec | str | None = None
    metadata: dict[str, str] | None = None
    source_policy: ParallelSourcePolicy | None = None
    previous_interaction_id: str | None = None
    mcp_servers: list[ParallelMCPServer] | None = None


class ParallelTaskPollOptions(EngineRequestPayload):
    """Client-side task polling options carried on EngineAPIRequest.call_options.
    Never serialized to the wire (transport only sends EngineAPIRequest.body())."""

    task_timeout: float | None = None
    task_api_timeout: int | None = None


class ParallelSearchResultItem(EngineResponsePayload):
    url: str
    title: str | None = None
    publish_date: str | None = None
    excerpts: list[str] | None = None


class ParallelSearchResponse(EngineResponsePayload):
    search_id: str | None = None
    session_id: str
    results: list[ParallelSearchResultItem] = Field(default_factory=list)
    warnings: list[JsonValue] | None = None
    usage: JsonValue = None


class ParallelExtractResultItem(EngineResponsePayload):
    url: str
    title: str | None = None
    publish_date: str | None = None
    excerpts: list[str] | None = None
    full_content: str | None = None


class ParallelExtractError(EngineResponsePayload):
    url: str | None = None
    error_type: str | None = None
    http_status_code: int | None = None
    content: str | None = None


class ParallelExtractResponse(EngineResponsePayload):
    extract_id: str | None = None
    session_id: str
    results: list[ParallelExtractResultItem] = Field(default_factory=list)
    errors: list[ParallelExtractError]
    warnings: list[JsonValue] | None = None


class ParallelTaskRun(EngineResponsePayload):
    """The 202 create response, also embedded as `run` in the completed result."""

    run_id: str
    interaction_id: str | None = None
    status: str | None = None
    is_active: bool | None = None
    processor: str | None = None


class ParallelCitation(EngineResponsePayload):
    url: str | None = None
    title: str | None = None
    excerpts: list[str] | None = None


class ParallelFieldBasis(EngineResponsePayload):
    field: str | None = None
    reasoning: str | None = None
    confidence: str | None = None
    citations: list[ParallelCitation] | None = None


class ParallelTaskOutput(EngineResponsePayload):
    type: str | None = None
    content: JsonValue = None
    basis: list[ParallelFieldBasis] | None = None


class ParallelTaskRunResult(EngineResponsePayload):
    """The 200 completed long-poll response."""

    run: ParallelTaskRun
    output: ParallelTaskOutput | None = None


class ParallelSourceItem(EngineResponsePayload):
    """Internal result item built by the task route (parallel:// pseudo-sources) and fed
    to ParallelSearchResult alongside regular ParallelSearchResultItem entries."""

    url: str
    title: str | None = None
    excerpts: list[str] | None = None


ParallelSearchAPIRequest = EngineAPIRequest[ParallelSearchRequest, EngineRequestPayload]
ParallelExtractAPIRequest = EngineAPIRequest[ParallelExtractRequest, EngineRequestPayload]
ParallelTaskRunCreateAPIRequest = EngineAPIRequest[
    ParallelTaskRunCreateRequest, ParallelTaskPollOptions
]
