import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from firecrawl import Firecrawl
from firecrawl.v2.types import ScrapeOptions

from ....symbol import Result
from ....utils import UserMessage
from ...base import Engine
from ...settings import SYMAI_CONFIG

logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

TRACKING_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
}


@dataclass
class Citation:
    id: int
    title: str
    url: str
    start: int
    end: int

    def __hash__(self):
        return hash((self.url,))


class FirecrawlSearchResult(Result):
    def __init__(
        self, value: dict[str, Any] | Any, max_chars_per_result: int | None = None, **kwargs
    ) -> None:
        raw_dict = value.model_dump() if hasattr(value, "model_dump") else value
        super().__init__(raw_dict, **kwargs)
        self._citations: list[Citation] = []
        self._max_chars_per_result = max_chars_per_result
        try:
            text, citations = self._build_text_and_citations(raw_dict)
            self._value = text
            self._citations = citations
        except Exception as e:
            self._value = None
            UserMessage(f"Failed to parse Firecrawl search response: {e}", raise_with=ValueError)

    def _build_text_and_citations(self, data: dict[str, Any]) -> tuple[str, list[Citation]]:
        results = []
        for source in ["web", "news", "images"]:
            source_data = data.get(source) or []
            results.extend(source_data)

        if not results:
            return "", []

        parts = []
        citations = []
        cursor = 0

        for idx, item in enumerate(results, 1):
            # Handle both SearchResultWeb (url/title at top level) and Document (url/title in metadata)
            metadata = item.get("metadata") or {}
            url = item.get("url") or metadata.get("url") or metadata.get("source_url") or ""
            title = item.get("title") or metadata.get("title") or ""

            if not url:
                continue

            # Check if this is a scraped result (has markdown content)
            markdown = item.get("markdown", "")
            if markdown:
                content = markdown
                if self._max_chars_per_result and len(content) > self._max_chars_per_result:
                    content = content[: self._max_chars_per_result] + "..."
                result_text = f"{title}\n{url}\n{content}"
            else:
                description = (
                    item.get("description")
                    or item.get("snippet")
                    or metadata.get("description")
                    or ""
                )
                result_text = f"{title}\n{url}"
                if description:
                    if self._max_chars_per_result and len(description) > self._max_chars_per_result:
                        description = description[: self._max_chars_per_result] + "..."
                    result_text += f"\n{description}"

            if parts:
                parts.append("\n\n")
                cursor += 2

            parts.append(result_text)
            cursor += len(result_text)

            marker = f"[{idx}]"
            start = cursor
            parts.append(marker)
            cursor += len(marker)

            citations.append(Citation(id=idx, title=title, url=url, start=start, end=cursor))

        text = "".join(parts)
        return text, citations

    def __str__(self) -> str:
        if isinstance(self._value, str) and self._value:
            return self._value
        try:
            return json.dumps(self.raw, indent=2)
        except TypeError:
            return str(self.raw)

    def _repr_html_(self) -> str:
        if isinstance(self._value, str) and self._value:
            return f"<pre>{self._value}</pre>"
        try:
            return f"<pre>{json.dumps(self.raw, indent=2)}</pre>"
        except Exception:
            return f"<pre>{self.raw!s}</pre>"

    def get_citations(self) -> list[Citation]:
        return self._citations


class FirecrawlExtractResult(Result):
    """Result wrapper for Firecrawl scrape API responses."""

    def __init__(self, value: Any, **kwargs) -> None:
        raw_dict = value.model_dump() if hasattr(value, "model_dump") else value
        super().__init__(raw_dict, **kwargs)
        try:
            self._value = self._extract_content(raw_dict)
        except Exception as e:
            self._value = None
            UserMessage(f"Failed to parse Firecrawl scrape response: {e}", raise_with=ValueError)

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

        if not self.api_key:
            UserMessage(
                "Firecrawl API key not found. Set SEARCH_ENGINE_API_KEY in config or environment.",
                raise_with=ValueError,
            )

        try:
            self.client = Firecrawl(api_key=self.api_key)
        except Exception as e:
            UserMessage(f"Failed to initialize Firecrawl client: {e}", raise_with=ValueError)

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

    def _normalize_url(self, url: str) -> str:
        parts = urlsplit(url)
        filtered_query = [
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k not in TRACKING_KEYS and not k.lower().startswith("utm_")
        ]
        query = urlencode(filtered_query, doseq=True)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))

    def _search(self, query: str, kwargs: dict[str, Any]):
        if not query:
            UserMessage(
                "FirecrawlEngine._search requires a non-empty query.", raise_with=ValueError
            )

        max_chars_per_result = kwargs.get("max_chars_per_result")

        # Build search kwargs
        search_kwargs = {}
        if "limit" in kwargs:
            search_kwargs["limit"] = kwargs["limit"]
        if "location" in kwargs:
            search_kwargs["location"] = kwargs["location"]
        if "tbs" in kwargs:
            search_kwargs["tbs"] = kwargs["tbs"]
        if "sources" in kwargs:
            search_kwargs["sources"] = kwargs["sources"]
        if "categories" in kwargs:
            search_kwargs["categories"] = kwargs["categories"]
        if "timeout" in kwargs:
            search_kwargs["timeout"] = kwargs["timeout"]

        # Build scrape options for search results content
        scrape_opts = {}
        if "formats" in kwargs:
            scrape_opts["formats"] = kwargs["formats"]
        if "proxy" in kwargs:
            scrape_opts["proxy"] = kwargs["proxy"]
        if "only_main_content" in kwargs:
            scrape_opts["only_main_content"] = kwargs["only_main_content"]
        if "scrape_location" in kwargs:
            scrape_opts["location"] = kwargs["scrape_location"]
        if "include_tags" in kwargs:
            scrape_opts["include_tags"] = kwargs["include_tags"]
        if "exclude_tags" in kwargs:
            scrape_opts["exclude_tags"] = kwargs["exclude_tags"]

        if scrape_opts:
            search_kwargs["scrape_options"] = ScrapeOptions(**scrape_opts)

        try:
            result = self.client.search(query, **search_kwargs)
        except Exception as e:
            UserMessage(f"Failed to call Firecrawl Search API: {e}", raise_with=ValueError)

        raw = result.model_dump() if hasattr(result, "model_dump") else result
        return [FirecrawlSearchResult(result, max_chars_per_result=max_chars_per_result)], {
            "raw_output": raw
        }

    def _extract(self, url: str, kwargs: dict[str, Any]):
        normalized_url = self._normalize_url(url)

        # Build scrape kwargs
        scrape_kwargs = {"formats": kwargs.get("formats", ["markdown"])}
        if "only_main_content" in kwargs:
            scrape_kwargs["only_main_content"] = kwargs["only_main_content"]
        if "timeout" in kwargs:
            scrape_kwargs["timeout"] = kwargs["timeout"]
        if "proxy" in kwargs:
            scrape_kwargs["proxy"] = kwargs["proxy"]
        if "location" in kwargs:
            scrape_kwargs["location"] = kwargs["location"]
        if "max_age" in kwargs:
            scrape_kwargs["max_age"] = kwargs["max_age"]
        if "store_in_cache" in kwargs:
            scrape_kwargs["store_in_cache"] = kwargs["store_in_cache"]
        if "actions" in kwargs:
            scrape_kwargs["actions"] = kwargs["actions"]
        if "headers" in kwargs:
            scrape_kwargs["headers"] = kwargs["headers"]
        if "include_tags" in kwargs:
            scrape_kwargs["include_tags"] = kwargs["include_tags"]
        if "exclude_tags" in kwargs:
            scrape_kwargs["exclude_tags"] = kwargs["exclude_tags"]
        if "wait_for" in kwargs:
            scrape_kwargs["wait_for"] = kwargs["wait_for"]
        if "mobile" in kwargs:
            scrape_kwargs["mobile"] = kwargs["mobile"]

        try:
            result = self.client.scrape(normalized_url, **scrape_kwargs)
        except Exception as e:
            UserMessage(f"Failed to call Firecrawl Scrape API: {e}", raise_with=ValueError)

        raw = result.model_dump() if hasattr(result, "model_dump") else result
        return [FirecrawlExtractResult(result)], {"raw_output": raw, "final_url": normalized_url}

    def forward(self, argument):
        kwargs = argument.kwargs
        url = argument.prop.url or kwargs.get("url")
        if url:
            return self._extract(str(url), kwargs)

        raw_query = argument.prop.prepared_input
        if raw_query is None:
            raw_query = argument.prop.query

        query = str(raw_query or "").strip() if raw_query else ""
        if not query:
            UserMessage(
                "FirecrawlEngine.forward requires at least one non-empty query or url.",
                raise_with=ValueError,
            )

        return self._search(query, kwargs)

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
