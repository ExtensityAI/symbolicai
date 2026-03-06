import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import tldextract

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
    from parallel.resources.task_run import build_task_spec_param
    from parallel.types.beta import WebSearchResult

    logging.getLogger("parallel").setLevel(logging.ERROR)
except ImportError:
    Parallel = None
    WebSearchResult = None


TRACKING_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
}

# --- Pre-compiled regex patterns ---
# Matches Markdown links like "[label](https://example.com "title")" and captures the label and URL.
_RE_MARKDOWN_LINK = re.compile(
    r"\[(?P<label>[^\]]+)\]\((?P<url>https?://[^)\s]+)(?:\s+\"[^\"]*\")?\)"
)
# Matches empty parentheses left over after stripping markdown links.
_RE_EMPTY_PARENS = re.compile(r"\(\s*\)")
# Matches parentheses containing only commas or whitespace remnants.
_RE_COMMA_PARENS = re.compile(r"\(\s*(,\s*)+\)")
# Strips all square bracket characters, preserving inner text.
_RE_SQUARE_BRACKETS = re.compile(r"[\[\]]")
# Collapses consecutive spaces/tabs down to a single space.
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
# Shrinks runs of three or more newlines to a double newline.
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
# Replaces non-safe characters in source identifiers with hyphens.
_RE_UNSAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9._:-]+")
# Converts non-lowercase-alphanumeric chars to hyphens for URL slugs.
_RE_SLUG = re.compile(r"[^a-z0-9]+")


@dataclass
class Citation:
    id: int
    title: str
    url: str
    start: int
    end: int

    def __hash__(self):
        return hash((self.url,))


class ParallelSearchResult(Result):
    def __init__(self, value, **kwargs) -> None:
        super().__init__(value, **kwargs)
        self._citations: list[Citation] = []
        # value is either:
        #   - SearchResult (from beta.search) with .results: list[WebSearchResult]
        #   - list[WebSearchResult] (from _task path)
        items = value.results if hasattr(value, "results") else value
        text, citations = self._build_text_and_citations(items)
        self._value = text
        self._citations = citations

    def _normalize_url(self, url: str) -> str:
        parts = urlsplit(url)
        scheme = parts.scheme.lower() if parts.scheme else "https"
        netloc = parts.netloc.lower()
        path = parts.path.rstrip("/") or "/"
        filtered_query = [
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k not in TRACKING_KEYS and not k.lower().startswith("utm_")
        ]
        query = urlencode(filtered_query, doseq=True)
        return urlunsplit((scheme, netloc, path, query, ""))

    def _sanitize_excerpt(self, text: str) -> str:
        cleaned = _RE_MARKDOWN_LINK.sub(lambda m: (m.group("label") or "").strip(), text)
        cleaned = _RE_EMPTY_PARENS.sub("", cleaned)
        cleaned = _RE_COMMA_PARENS.sub("", cleaned)
        cleaned = _RE_SQUARE_BRACKETS.sub("", cleaned)
        cleaned = _RE_MULTI_SPACE.sub(" ", cleaned)
        cleaned = _RE_MULTI_NEWLINE.sub("\n\n", cleaned)
        return cleaned.strip()

    def _build_text_and_citations(self, results):
        pieces = []
        citations = []
        cursor = 0
        seen_urls = set()
        cid = 1
        separator = "\n\n---\n\n"

        for item in results:
            url = item.url
            if not url:
                continue
            normalized_url = self._normalize_url(url)
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)

            title = item.title or urlsplit(normalized_url).netloc
            excerpts = item.excerpts or []
            excerpt_parts = [p for ex in excerpts if (p := self._sanitize_excerpt(ex))]
            if not excerpt_parts:
                continue

            combined_excerpt = "\n\n".join(excerpt_parts)
            raw_id = urlsplit(normalized_url).netloc or normalized_url
            source_id = _RE_UNSAFE_ID_CHARS.sub("-", raw_id).strip("-") or f"source-{cid}"
            block_body = f"{source_id}\n\n{combined_excerpt}"

            if pieces:
                pieces.append(separator)
                cursor += len(separator)

            opening_tag = "<source>\n"
            pieces.append(opening_tag)
            cursor += len(opening_tag)

            pieces.append(block_body)
            cursor += len(block_body)

            closing_tag = "\n</source>"
            pieces.append(closing_tag)
            cursor += len(closing_tag)

            marker = f"[{cid}]"
            start = cursor
            pieces.append(marker)
            cursor += len(marker)

            citations.append(
                Citation(id=cid, title=title, url=normalized_url, start=start, end=cursor)
            )
            cid += 1

        text = "".join(pieces)
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


class ParallelExtractResult(Result):
    """Result wrapper for Parallel Extract API responses (ExtractResponse)."""

    def __init__(self, value, **kwargs) -> None:
        super().__init__(value, **kwargs)
        # value is an ExtractResponse with .results: list[ExtractResult]
        content_parts = []
        for r in value.results:
            if r.full_content is not None:
                content_parts.append(r.full_content)
            elif r.excerpts:
                content_parts.extend(r.excerpts)
        self._value = "\n\n".join(content_parts)

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


class ParallelEngine(Engine):
    MAX_INCLUDE_DOMAINS = 10
    MAX_EXCLUDE_DOMAINS = 10

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        self.api_key = api_key or self.config.get("SEARCH_ENGINE_API_KEY")
        self.model = self.config.get("SEARCH_ENGINE_MODEL")
        self.name = self.__class__.__name__

        if Parallel is None:
            UserMessage(
                "parallel-web SDK is not installed. Install with 'pip install parallel-web' "
                "or add it to your environment.",
                raise_with=ValueError,
            )

        try:
            self.client = Parallel(api_key=self.api_key)
        except Exception as e:
            UserMessage(f"Failed to initialize Parallel client: {e}", raise_with=ValueError)

    def id(self) -> str:
        if self.api_key and self.model == "parallel":
            return "search"
        return super().id()

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "SEARCH_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["SEARCH_ENGINE_API_KEY"]
        if "SEARCH_ENGINE_MODEL" in kwargs:
            self.model = kwargs["SEARCH_ENGINE_MODEL"]

    def _normalize_domains(self, domains: list[str] | None, max_count: int) -> list[str]:
        if not isinstance(domains, list):
            return []
        seen = set()
        out = []
        for d in domains:
            fqdn = tldextract.extract(d).fqdn
            if not fqdn or fqdn in seen:
                continue
            seen.add(fqdn)
            out.append(fqdn)
            if len(out) >= max_count:
                break
        return out

    def _coerce_search_queries(self, value: Any) -> list[str]:  # called from forward + prepare
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, list):
            cleaned = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    cleaned.append(text)
            return cleaned
        text = str(value).strip()
        return [text] if text else []

    def _build_source_policy(self, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        include = self._normalize_domains(kwargs.get("allowed_domains"), self.MAX_INCLUDE_DOMAINS)
        exclude = self._normalize_domains(kwargs.get("excluded_domains"), self.MAX_EXCLUDE_DOMAINS)
        if not include and not exclude:
            return None
        policy = {}
        if include:
            policy["include_domains"] = include
        if exclude:
            policy["exclude_domains"] = exclude
        return policy

    def _search(self, queries: list[str], kwargs: dict[str, Any]):
        if not queries:
            UserMessage(
                "ParallelEngine._search requires at least one query.", raise_with=ValueError
            )

        mode = kwargs.get("mode") or "one-shot"
        max_results = kwargs.get("max_results", 10)
        max_chars_per_result = kwargs.get("max_chars_per_result", 15000)
        excerpts = {"max_chars_per_result": max_chars_per_result}
        source_policy = self._build_source_policy(kwargs)
        objective = kwargs.get("objective")

        try:
            result = self.client.beta.search(
                objective=objective,
                search_queries=queries,
                max_results=max_results,
                excerpts=excerpts,
                mode=mode,
                source_policy=source_policy,
            )
        except Exception as e:
            UserMessage(f"Failed to call Parallel Search API: {e}", raise_with=ValueError)
        return [ParallelSearchResult(result)], {"raw_output": result}

    def _task(self, queries: list[str], kwargs: dict[str, Any]):
        processor = kwargs.get("processor")
        if not processor or not str(processor).strip():
            UserMessage(
                "ParallelEngine.task requires a non-empty processor.", raise_with=ValueError
            )

        task_input = (
            queries[0]
            if len(queries) == 1
            else "\n\n".join(f"{i}. {q}" for i, q in enumerate(queries, start=1))
        )

        source_policy = self._build_source_policy(kwargs)
        output_schema = (
            kwargs.get("task_output_schema")
            or kwargs.get("task_output")
            or kwargs.get("output_schema")
            or kwargs.get("output")
        )

        create_kwargs = {"input": task_input, "processor": str(processor).strip()}
        metadata = kwargs.get("metadata")
        if isinstance(metadata, dict):
            create_kwargs["metadata"] = metadata
        if source_policy is not None:
            create_kwargs["source_policy"] = source_policy
        if output_schema is not None:
            try:
                create_kwargs["task_spec"] = build_task_spec_param(output_schema, task_input)
            except Exception as exc:
                UserMessage(f"Invalid task output schema: {exc}", raise_with=ValueError)

        try:
            run = self.client.task_run.create(**create_kwargs)
        except Exception as e:
            UserMessage(f"Failed to create Parallel task: {e}", raise_with=ValueError)

        result_kwargs = {}
        timeout = kwargs.get("task_timeout") or kwargs.get("timeout")
        api_timeout = kwargs.get("task_api_timeout") or kwargs.get("api_timeout")
        if timeout is not None:
            result_kwargs["timeout"] = timeout
        if api_timeout is not None:
            try:
                result_kwargs["api_timeout"] = int(api_timeout)
            except (TypeError, ValueError) as exc:
                UserMessage(f"api_timeout must be numeric: {exc}", raise_with=ValueError)

        try:
            task_result = self.client.task_run.result(run.run_id, **result_kwargs)
        except Exception as e:
            UserMessage(f"Failed to fetch Parallel task result: {e}", raise_with=ValueError)

        output = task_result.output
        items, prefix = self._task_output_to_items(output)
        wrapped = ParallelSearchResult(items)
        if prefix:
            offset = len(prefix) + (2 if wrapped._value else 0)
            for c in wrapped._citations:
                c.start += offset
                c.end += offset
            wrapped._value = prefix + ("\n\n" + wrapped._value if wrapped._value else "")
        wrapped.raw = task_result
        return [wrapped], {
            "raw_output": task_result,
            "task_output": output.content if output else None,
            "task_output_type": output.type if output else None,
        }

    def _task_output_to_items(self, output) -> tuple[list, str]:
        """Flatten TaskRunTextOutput/TaskRunJsonOutput into WebSearchResult list and prefix text."""
        if output is None:
            return [], ""

        basis_items = output.basis or []
        prefix_parts = []
        if basis_items:
            first = basis_items[0]
            if first.reasoning:
                prefix_parts.append(f"<reasoning>\n{first.reasoning}\n</reasoning>")
            if isinstance(output.content, str) and output.content:
                prefix_parts.append(f"<answer>\n{output.content}\n</answer>")
            if first.confidence:
                prefix_parts.append(
                    f"<answer_confidence>\n{first.confidence}\n</answer_confidence>"
                )

        items = []
        for idx, basis in enumerate(basis_items):
            reasoning = basis.reasoning or ""
            field_title = basis.field or ""
            if not field_title.strip():
                field_title = "Parallel Task Output"
            citations = basis.citations or []
            if not citations:
                if not reasoning:
                    continue
                citations = [None]

            slug = _RE_SLUG.sub("-", field_title.lower()).strip("-") or "field"
            basis_url = f"parallel://task-output/{idx:04d}-{slug}"
            for citation in citations:
                if citation is None:
                    items.append(
                        WebSearchResult(url=basis_url, title=field_title, excerpts=[reasoning])
                    )
                else:
                    items.append(
                        WebSearchResult(
                            url=citation.url or basis_url,
                            title=citation.title or field_title,
                            excerpts=citation.excerpts or ([reasoning] if reasoning else []),
                        )
                    )

        if not items:
            content = output.content
            if isinstance(content, str):
                snippet = content
            elif isinstance(content, (dict, list)):
                snippet = json.dumps(content, ensure_ascii=False)
            else:
                snippet = str(content or "")
            if not snippet:
                snippet = (
                    " ".join(b.reasoning for b in basis_items if b.reasoning)
                    or "Parallel task output"
                )
            items.append(
                WebSearchResult(
                    url="parallel://task-output",
                    title="Parallel Task Output",
                    excerpts=[snippet],
                )
            )

        return items, "\n\n".join(prefix_parts)

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
        return [ParallelExtractResult(result)], {"raw_output": result, "final_url": url}

    def forward(self, argument):
        kwargs = argument.kwargs
        # Route based on presence of URL vs Query
        url = argument.prop.url or kwargs.get("url")
        if url:
            return self._extract(str(url), kwargs)

        raw_query = argument.prop.prepared_input
        if raw_query is None:
            raw_query = argument.prop.query
        search_queries = self._coerce_search_queries(raw_query)
        if not search_queries:
            UserMessage(
                "ParallelEngine.forward requires at least one non-empty query or url.",
                raise_with=ValueError,
            )
        processor = kwargs.get("processor")
        if processor is not None:
            return self._task(search_queries, kwargs)
        return self._search(search_queries, kwargs)

    def prepare(self, argument):
        # For scraping: store URL directly. For search: pass through query string.
        url = argument.kwargs.get("url") or argument.prop.url
        if url:
            argument.prop.prepared_input = str(url)
            return
        query = argument.prop.query
        if isinstance(query, list):
            argument.prop.prepared_input = self._coerce_search_queries(query)
            return
        argument.prop.prepared_input = str(query or "").strip()
