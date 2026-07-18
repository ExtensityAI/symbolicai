from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any

from symai.backend.base import Engine
from symai.backend.engines.search.firecrawl.models import (
    FIRECRAWL_API_BASE,
    FirecrawlScrapeOptions,
    FirecrawlScrapeRequest,
    FirecrawlScrapeResponse,
    FirecrawlSearchRequest,
    FirecrawlSearchResponse,
)
from symai.backend.engines.search.utils import Citation, CitationResult, normalize_url
from symai.backend.request import EngineAPIRequest
from symai.backend.settings import SYMAI_CONFIG
from symai.backend.transport import DEFAULT_RETRIES, execute_engine_api_request
from symai.symbol import Result
from symai.utils import silence_noisy_loggers

silence_noisy_loggers()

logger = logging.getLogger(__name__)


class FirecrawlSearchResult(CitationResult, Result):
    def __init__(
        self, value: dict[str, Any] | Any, max_chars_per_result: int | None = None, **kwargs
    ) -> None:
        raw_dict = value.model_dump() if hasattr(value, "model_dump") else value
        super().__init__(raw_dict, **kwargs)
        self._citations = []
        self._max_chars_per_result = max_chars_per_result
        try:
            text, citations = self._build_text_and_citations(raw_dict)
            self._value = text
            self._citations = citations
        except Exception as e:
            self._value = None
            msg = f"Failed to parse Firecrawl search response: {e}"
            raise ValueError(msg) from e

    def _build_text_and_citations(self, data: dict[str, Any]) -> tuple[str, list[Citation]]:
        # NOTE: the v2 wire fixes one content field per source — web items carry
        # `description` (plus `markdown` when scrape options requested it), news items
        # `snippet`, and images no content field at all (title + url only).
        items = [
            (item, content_field)
            for source, content_field in (
                ("web", "description"),
                ("news", "snippet"),
                ("images", None),
            )
            for item in data.get(source) or []
        ]

        if not items:
            return "", []

        parts = []
        citations = []
        cursor = 0
        seen_urls = set()
        cid = 0

        for item, content_field in items:
            raw_url = item.get("url")
            if not raw_url:
                continue

            url = normalize_url(raw_url)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            cid += 1

            title = item.get("title") or ""
            markdown = item.get("markdown")
            if markdown:
                content = markdown
                if self._max_chars_per_result and len(content) > self._max_chars_per_result:
                    content = content[: self._max_chars_per_result] + "..."
                result_text = f"{title}\n{url}\n{content}"
            else:
                result_text = f"{title}\n{url}"
                description = item.get(content_field) if content_field else None
                if description:
                    if self._max_chars_per_result and len(description) > self._max_chars_per_result:
                        description = description[: self._max_chars_per_result] + "..."
                    result_text += f"\n{description}"

            if parts:
                parts.append("\n\n")
                cursor += 2

            parts.append(result_text)
            cursor += len(result_text)

            marker = f"[{cid}]"
            start = cursor
            parts.append(marker)
            cursor += len(marker)

            citations.append(Citation(id=cid, title=title, url=url, start=start, end=cursor))

        text = "".join(parts)
        return text, citations


class FirecrawlExtractResult(Result):
    """Result wrapper for Firecrawl scrape API responses."""

    def __init__(self, value: Any, **kwargs) -> None:
        raw_dict = value.model_dump() if hasattr(value, "model_dump") else value
        super().__init__(raw_dict, **kwargs)
        try:
            self._value = self._extract_content(raw_dict)
        except Exception as e:
            self._value = None
            msg = f"Failed to parse Firecrawl scrape response: {e}"
            raise ValueError(msg) from e

    def _extract_content(self, data: dict[str, Any]) -> str:
        content = data.get("markdown") or data.get("html") or data.get("raw_html")
        if content:
            return str(content)
        json_data = data.get("json")
        if json_data:
            return json.dumps(json_data, indent=2)
        return ""

    def __str__(self) -> str:
        try:
            return str(self._value or "")
        except Exception:
            return ""

    def _repr_html_(self) -> str:
        try:
            return f"<pre>{self._value or ''}</pre>"
        except Exception:
            return "<pre></pre>"


class FirecrawlEngine(Engine):
    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        self.api_key = api_key or self.config.get("SEARCH_ENGINE_API_KEY")
        self.model = self.config.get("SEARCH_ENGINE_MODEL")
        self.name = self.__class__.__name__
        self.transport_client = None
        self._max_chars_per_result = None
        self._final_url = None

        if api_key is None and self.id() != "search":
            return

        if not self.api_key:
            msg = "Firecrawl API key not found. Set SEARCH_ENGINE_API_KEY in config or environment."
            raise ValueError(msg)

    def id(self) -> str:
        if (
            self.config.get("SEARCH_ENGINE_API_KEY")
            and str(self.config.get("SEARCH_ENGINE_MODEL", "")).lower() == "firecrawl"
        ):
            return "search"
        return super().id()

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "SEARCH_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["SEARCH_ENGINE_API_KEY"]
        if "SEARCH_ENGINE_MODEL" in kwargs:
            self.model = kwargs["SEARCH_ENGINE_MODEL"]

    def forward(self, argument):
        request = self.build_request(argument)
        response = self.call_request(request)
        return self.parse_response(response)

    def build_request(self, argument) -> EngineAPIRequest:
        kwargs = argument.kwargs
        url = argument.prop.url or kwargs.get("url")
        if url:
            operation = "scrape"
            payload = self._build_scrape_payload(str(url), kwargs)
        else:
            operation = "search"
            payload = self._build_search_payload(argument, kwargs)

        return EngineAPIRequest(
            provider="firecrawl",
            operation=operation,
            payload=payload,
            method="POST",
            url=f"{FIRECRAWL_API_BASE}/{operation}",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.client_timeout,
        )

    def _build_search_payload(self, argument, kwargs) -> FirecrawlSearchRequest:
        raw_query = argument.prop.prepared_input
        if raw_query is None:
            raw_query = argument.prop.query
        query = str(raw_query or "").strip() if raw_query else ""
        if not query:
            msg = "FirecrawlEngine.forward requires at least one non-empty query or url."
            raise ValueError(msg)

        # NOTE: max_chars_per_result steers result parsing (per-result truncation), not the
        # wire payload; stash it for parse_response, which only receives the response.
        self._max_chars_per_result = kwargs.get("max_chars_per_result")

        search_kwargs = {}
        for key in ("limit", "location", "tbs", "sources", "categories", "timeout"):
            if key in kwargs:
                search_kwargs[key] = kwargs[key]

        scrape_opts = {}
        for key in ("formats", "proxy", "only_main_content", "include_tags", "exclude_tags"):
            if key in kwargs:
                scrape_opts[key] = kwargs[key]
        if "scrape_location" in kwargs:
            scrape_opts["location"] = kwargs["scrape_location"]

        if scrape_opts:
            search_kwargs["scrape_options"] = FirecrawlScrapeOptions.model_validate(scrape_opts)

        return FirecrawlSearchRequest.model_validate({"query": query, **search_kwargs})

    def _build_scrape_payload(self, url: str, kwargs) -> FirecrawlScrapeRequest:
        normalized_url = normalize_url(url)
        # NOTE: final_url is parse-time metadata (the normalized input url); stash it for
        # parse_response, which only receives the response.
        self._final_url = normalized_url

        scrape_kwargs: dict[str, Any] = {"formats": kwargs.get("formats", ["markdown"])}
        for key in (
            "only_main_content",
            "timeout",
            "proxy",
            "location",
            "max_age",
            "store_in_cache",
            "actions",
            "headers",
            "include_tags",
            "exclude_tags",
            "wait_for",
            "mobile",
        ):
            if key in kwargs:
                scrape_kwargs[key] = kwargs[key]

        return FirecrawlScrapeRequest.model_validate({"url": normalized_url, **scrape_kwargs})

    def call_request(
        self, request: EngineAPIRequest
    ) -> FirecrawlSearchResponse | FirecrawlScrapeResponse:
        max_retries = (
            self.client_max_retries if self.client_max_retries is not None else DEFAULT_RETRIES
        )
        response = execute_engine_api_request(
            request,
            client=self.transport_client,
            max_retries=max_retries,
        )
        model = (
            FirecrawlScrapeResponse if request.operation == "scrape" else FirecrawlSearchResponse
        )
        result = model.model_validate(response.json())
        if not result.success:
            msg = f"Failed to call Firecrawl {request.operation.capitalize()} API: {result.error or 'unknown error'}"
            raise ValueError(msg)
        return result

    def parse_response(self, response: FirecrawlSearchResponse | FirecrawlScrapeResponse):
        data = response.data.model_dump(exclude_none=True) if response.data is not None else {}
        if isinstance(response, FirecrawlScrapeResponse):
            return [FirecrawlExtractResult(data)], {
                "raw_output": response,
                "final_url": self._final_url,
            }
        return [FirecrawlSearchResult(data, max_chars_per_result=self._max_chars_per_result)], {
            "raw_output": response
        }

    def prepare(self, argument):
        url = argument.kwargs.get("url") or argument.prop.url
        if url:
            argument.prop.prepared_input = str(url)
            return

        query = argument.prop.query
        if isinstance(query, list):
            argument.prop.prepared_input = " ".join(str(q) for q in query if q)
            return

        argument.prop.prepared_input = str(query or "").strip()
