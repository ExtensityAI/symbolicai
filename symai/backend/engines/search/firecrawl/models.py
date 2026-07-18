"""Firecrawl v2 search/scrape wire models.

Docs: https://docs.firecrawl.dev/api-reference/endpoint/search
      https://docs.firecrawl.dev/api-reference/endpoint/scrape
Verified against the official v2 OpenAPI spec on API_PINNED.
"""

from __future__ import annotations

import warnings
from typing import Literal

from pydantic import Field, JsonValue

from symai.backend.request import EngineRequestPayload, EngineResponsePayload

API_PINNED = "2026-07-18"

FIRECRAWL_API_BASE = "https://api.firecrawl.dev/v2"


class FirecrawlLocation(EngineRequestPayload):
    country: str | None = None
    languages: list[str] | None = None


class FirecrawlScrapeOptions(EngineRequestPayload):
    formats: list[str | dict[str, JsonValue]] | None = None
    only_main_content: bool | None = Field(default=None, alias="onlyMainContent")
    include_tags: list[str] | None = Field(default=None, alias="includeTags")
    exclude_tags: list[str] | None = Field(default=None, alias="excludeTags")
    max_age: int | None = Field(default=None, alias="maxAge")
    wait_for: int | None = Field(default=None, alias="waitFor")
    mobile: bool | None = None
    proxy: str | None = None
    store_in_cache: bool | None = Field(default=None, alias="storeInCache")
    timeout: int | None = None  # milliseconds
    location: FirecrawlLocation | None = None


class FirecrawlSearchRequest(EngineRequestPayload):
    query: str
    limit: int | None = Field(default=None, ge=1, le=100)
    sources: list[Literal["web", "news", "images"]] | None = None
    categories: list[str] | None = None
    tbs: str | None = None
    # NOTE: location is a plain string on /search (e.g. "San Francisco,California,United
    # States") but an object inside scrapeOptions and on /scrape — do not unify the two.
    location: str | None = None
    country: str | None = None
    timeout: int | None = None  # milliseconds
    ignore_invalid_urls: bool | None = Field(default=None, alias="ignoreInvalidURLs")
    scrape_options: FirecrawlScrapeOptions | None = Field(default=None, alias="scrapeOptions")


class FirecrawlScrapeRequest(EngineRequestPayload):
    url: str
    formats: list[str | dict[str, JsonValue]] | None = None
    only_main_content: bool | None = Field(default=None, alias="onlyMainContent")
    include_tags: list[str] | None = Field(default=None, alias="includeTags")
    exclude_tags: list[str] | None = Field(default=None, alias="excludeTags")
    max_age: int | None = Field(default=None, alias="maxAge")
    store_in_cache: bool | None = Field(default=None, alias="storeInCache")
    timeout: int | None = None  # milliseconds
    proxy: str | None = None
    location: FirecrawlLocation | None = None
    headers: dict[str, str] | None = None
    wait_for: int | None = Field(default=None, alias="waitFor")
    mobile: bool | None = None
    actions: list[dict[str, JsonValue]] | None = None


class FirecrawlSearchResultItem(EngineResponsePayload):
    url: str | None = None
    title: str | None = None
    description: str | None = None
    snippet: str | None = None
    markdown: str | None = None


class FirecrawlSearchData(EngineResponsePayload):
    web: list[FirecrawlSearchResultItem] | None = None
    news: list[FirecrawlSearchResultItem] | None = None
    images: list[FirecrawlSearchResultItem] | None = None


class FirecrawlSearchResponse(EngineResponsePayload):
    # NOTE: error envelopes arrive either as {error: str} (HTTP error, handled by the
    # transport) or {success: false, code, error} with HTTP 200 — hence the tolerant
    # error/code fields alongside success.
    success: bool
    data: FirecrawlSearchData | None = None
    warning: str | None = None
    id: str | None = None
    credits_used: float | None = Field(default=None, alias="creditsUsed")
    error: str | None = None
    code: str | int | None = None


with warnings.catch_warnings():
    # NOTE: `json` is the wire key for extracted structured data. It shadows the
    # deprecated BaseModel.json method, so the shadow warning is suppressed locally
    # (the old SDK-based module filtered the same class of warning from firecrawl-py).
    warnings.filterwarnings("ignore", message='Field name "json".*shadows an attribute')

    class FirecrawlScrapeData(EngineResponsePayload):
        markdown: str | None = None
        html: str | None = None
        raw_html: str | None = Field(default=None, alias="rawHtml")
        json: JsonValue | None = None
        links: list[str] | None = None
        metadata: dict[str, JsonValue] | None = None
        warning: str | None = None


class FirecrawlScrapeResponse(EngineResponsePayload):
    success: bool
    data: FirecrawlScrapeData | None = None
    error: str | None = None
    code: str | int | None = None
