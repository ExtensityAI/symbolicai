import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ....symbol import Result
from ....utils import UserMessage
from ...base import Engine
from ...settings import SYMAI_CONFIG

logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

try:
    from parallel import Parallel
    logging.getLogger("parallel").setLevel(logging.ERROR)
except ImportError as exc:
    msg = (
        "parallel-web SDK is not installed. Install with 'pip install parallel-web' "
        "or add it to your environment."
    )
    UserMessage(msg)
    raise RuntimeError(msg) from exc


TRACKING_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
}


def _item_to_mapping(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        try:
            return dict(item.model_dump())
        except TypeError:
            return dict(item.model_dump(mode="python"))
    if hasattr(item, "dict"):
        return dict(item.dict())
    if hasattr(item, "__dict__"):
        return deepcopy({k: v for k, v in item.__dict__.items() if not k.startswith("_")})
    return {}


@dataclass
class Citation:
    id: int
    title: str
    url: str
    start: int
    end: int

    def __hash__(self):
        return hash((self.url,))


class SearchResult(Result):
    """Normalized search result with inline citation markers.

    Produces a flattened text where each cited excerpt is cleaned of stray
    square-bracket artifacts, concatenated, and annotated inline with a
    "[<id>]" suffix. Citations are exposed via ``get_citations()`` and URLs
    are normalized (no utm_ params, no fragments).
    """

    def __init__(self, value: dict[str, Any] | Any, **kwargs) -> None:
        super().__init__(value, **kwargs)
        if isinstance(value, dict) and value.get("error"):
            UserMessage(value["error"], raise_with=ValueError)
        self._citations: list[Citation] = []
        try:
            results = self._coerce_results(value)
            text, citations = self._build_text_and_citations(results)
            self._value = text
            self._citations = citations
        except Exception as e:
            self._value = None
            UserMessage(f"Failed to parse Parallel search response: {e}", raise_with=ValueError)

    def _coerce_results(self, raw: Any) -> list[dict[str, Any]]:
        if raw is None:
            return []
        results = raw.get("results", []) if isinstance(raw, dict) else getattr(raw, "results", None)
        if not results:
            return []
        coerced: list[dict[str, Any]] = []
        for item in results:
            if item is None:
                continue
            coerced.append(_item_to_mapping(item))
        return coerced

    def _normalize_url(self, u: str) -> str:
        parts = urlsplit(u)
        scheme = parts.scheme.lower() if parts.scheme else "https"
        netloc = parts.netloc.lower()
        path = parts.path.rstrip("/") or "/"
        q = []
        for k, v in parse_qsl(parts.query, keep_blank_values=True):
            kl = k.lower()
            if kl in TRACKING_KEYS or kl.startswith("utm_"):
                continue
            q.append((k, v))
        query = urlencode(q, doseq=True)
        fragment = ""
        return urlunsplit((scheme, netloc, path, query, fragment))

    def _strip_markdown_links(self, text: str) -> str:
        pattern_paren = re.compile(r"\(\s*\[[^\]]+\]\(https?://[^)]+\)\s*\)")
        text = pattern_paren.sub("", text)
        pattern_bare = re.compile(r"\[[^\]]+\]\(https?://[^)]+\)")
        text = pattern_bare.sub("", text)
        pattern_empty_paren = re.compile(r"\(\s*\)")
        text = pattern_empty_paren.sub("", text)
        pattern_commas_only = re.compile(r"\(\s*(,\s*)+\)")
        text = pattern_commas_only.sub("", text)
        return re.sub(r"\s{2,}", " ", text).strip()

    def _strip_square_brackets(self, text: str) -> str:
        def _replace(match) -> str:
            inner = match.group(1)
            return inner or ""

        cleaned = re.sub(r"\[([^\]]*)\]", _replace, text)
        return cleaned.replace("[", "").replace("]", "")

    def _build_text_and_citations(self, results: list[dict[str, Any]]):
        pieces: list[str] = []
        citations: list[Citation] = []
        cursor = 0
        seen_urls: set[str] = set()
        cid = 1

        for item in results:
            url = str(item.get("url", "")).strip()
            if not url:
                continue
            nu = self._normalize_url(url)
            if nu in seen_urls:
                continue
            seen_urls.add(nu)

            title = str(item.get("title") or "").strip() or urlsplit(nu).netloc
            excerpts = item.get("excerpts") or []
            excerpt_parts: list[str] = []
            for ex in excerpts:
                if not isinstance(ex, str):
                    continue
                cleaned = self._strip_markdown_links(ex)
                cleaned = self._strip_square_brackets(cleaned)
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                if cleaned:
                    excerpt_parts.append(cleaned)

            combined_excerpt = " ".join(excerpt_parts).strip()
            if not combined_excerpt:
                continue

            if pieces:
                pieces.append(" ")
                cursor += 1

            pieces.append(combined_excerpt)
            cursor += len(combined_excerpt)

            pieces.append(" ")
            cursor += 1

            marker = f"[{cid}]"
            start = cursor
            end = start + len(marker)
            pieces.append(marker)
            cursor += len(marker)

            citations.append(Citation(id=cid, title=title, url=nu, start=start, end=end))
            cid += 1

        text = "".join(pieces).strip() if pieces else ""
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


class ExtractResult(Result):
    """Result wrapper for Parallel Extract API responses."""

    def __init__(self, value: dict[str, Any] | Any, **kwargs) -> None:
        super().__init__(value, **kwargs)
        try:
            results = self._coerce_results(value)
            content_parts: list[str] = []
            for r in results:
                excerpts = r.get("excerpts") or []
                full = r.get("full_content")
                if full:
                    content_parts.append(str(full))
                elif excerpts:
                    content_parts.extend([s for s in excerpts if isinstance(s, str) and s.strip()])
            self._value = "\n\n".join(content_parts).strip()
        except Exception as e:
            self._value = None
            UserMessage(f"Failed to parse Parallel extract response: {e}", raise_with=ValueError)

    def _coerce_results(self, raw: Any) -> list[dict[str, Any]]:
        if raw is None:
            return []
        results = raw.get("results", []) if isinstance(raw, dict) else getattr(raw, "results", None)
        if not results:
            return []
        coerced: list[dict[str, Any]] = []
        for item in results:
            if item is None:
                continue
            coerced.append(_item_to_mapping(item))
        return coerced

    def __str__(self) -> str:
        try:
            return str(self._value or "")
        except Exception:
            return ""

    def _repr_html_(self) -> str:
        try:
            return f"<pre>{(self._value or '').strip()}</pre>"
        except Exception:
            return "<pre></pre>"


class ParallelEngine(Engine):
    MAX_INCLUDE_DOMAINS = 10

    def __init__(self):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        self.api_key = self.config.get("SEARCH_ENGINE_API_KEY")
        self.model = self.config.get("SEARCH_ENGINE_MODEL")
        self.name = self.__class__.__name__

        try:
            self.client = Parallel(api_key=self.api_key)
        except Exception as e:
            UserMessage(f"Failed to initialize Parallel client: {e}", raise_with=ValueError)

    def id(self) -> str:
        # Register as a search engine when configured with the 'parallel' model token
        if self.config.get("SEARCH_ENGINE_API_KEY") and str(self.config.get("SEARCH_ENGINE_MODEL", "")).lower() == "parallel":
            return "search"
        return super().id()

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "SEARCH_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["SEARCH_ENGINE_API_KEY"]
        if "SEARCH_ENGINE_MODEL" in kwargs:
            self.model = kwargs["SEARCH_ENGINE_MODEL"]

    def _extract_netloc(self, raw: str | None) -> str | None:
        if not isinstance(raw, str):
            return None
        s = raw.strip()
        if not s:
            return None
        parts = urlsplit(s if "://" in s else f"//{s}")
        netloc = parts.netloc or parts.path
        netloc = netloc.split("@", 1)[-1]
        netloc = netloc.split(":", 1)[0]
        netloc = netloc.strip(".").strip().lower()
        return netloc or None

    def _normalize_include_domains(self, domains: list[str] | None) -> list[str]:
        if not isinstance(domains, list):
            return []
        seen: set[str] = set()
        out: list[str] = []
        for d in domains:
            netloc = self._extract_netloc(d)
            if not netloc or netloc in seen:
                continue
            if not self._is_valid_domain(netloc):
                # Skip strings that are not apex domains or bare TLD patterns
                continue
            seen.add(netloc)
            out.append(netloc)
            if len(out) >= self.MAX_INCLUDE_DOMAINS:
                break
        return out

    def _is_valid_domain(self, s: str) -> bool:
        """Validate apex domains or bare extension filters.

        Accepts:
        - Apex/sub domains like "example.com", "www.arstechnica.com"
        - Bare extension patterns like ".gov", ".co.uk"
        Rejects:
        - Values without a dot (e.g., "tomshardware")
        - Schemes, paths, or ports (filtered earlier by _extract_netloc)
        """
        if not s:
            return False
        if s.startswith('.'):
            # Allow bare domain extensions like .gov or .co.uk
            remainder = s[1:]
            return bool((remainder and '.' in remainder) or remainder.isalpha())
        # Require at least one dot and valid label characters
        label_re = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
        parts = s.split('.')
        if len(parts) < 2:
            return False
        return all(label_re.fullmatch(p or "") for p in parts)

    def _search(self, query: str, kwargs: dict[str, Any]):
        mode = kwargs.get("mode") or "one-shot"
        max_results = kwargs.get("max_results", 10)
        max_chars_per_result = kwargs.get("max_chars_per_result", 10000)
        include = self._normalize_include_domains(kwargs.get("allowed_domains"))
        source_policy = {"include_domains": include} if include else None
        search_queries = kwargs.get("search_queries")
        objective = kwargs.get("objective") or query

        try:
            result = self.client.beta.search(
                objective=objective,
                search_queries=search_queries,
                max_results=max_results,
                max_chars_per_result=max_chars_per_result,
                mode=mode,
                source_policy=source_policy,
            )
        except Exception as e:
            UserMessage(f"Failed to call Parallel Search API: {e}", raise_with=ValueError)
        return [SearchResult(result)], {"raw_output": result}

    def _extract(self, url: str, kwargs: dict[str, Any]):
        excerpts = kwargs.get("excerpts", True)
        full_content = kwargs.get("full_content", False)
        objective = kwargs.get("objective")
        try:
            result = self.client.beta.extract(
                urls=[url],
                objective=objective,
                excerpts=excerpts,
                full_content=full_content,
            )
        except Exception as e:
            UserMessage(f"Failed to call Parallel Extract API: {e}", raise_with=ValueError)
        return [ExtractResult(result)], {"raw_output": result, "final_url": url}

    def forward(self, argument):
        kwargs = argument.kwargs
        # Route based on presence of URL vs Query
        url = getattr(argument.prop, "url", None) or kwargs.get("url")
        if url:
            return self._extract(str(url), kwargs)

        query = getattr(argument.prop, "prepared_input", None) or getattr(argument.prop, "query", None)
        query = str(query or "").strip()
        if not query:
            UserMessage("ParallelEngine.forward requires a non-empty query or url.", raise_with=ValueError)
        return self._search(query, kwargs)

    def prepare(self, argument):
        # For scraping: store URL directly. For search: pass through query string.
        url = argument.kwargs.get("url") or getattr(argument.prop, "url", None)
        if url:
            argument.prop.prepared_input = str(url)
            return
        query = getattr(argument.prop, "query", None)
        argument.prop.prepared_input = str(query or "")
