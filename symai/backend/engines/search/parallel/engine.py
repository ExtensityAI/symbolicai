import json
import logging
import re
import time
from copy import deepcopy
from urllib.parse import urlsplit

from symai.backend.base import Engine
from symai.backend.engines.search.parallel.models import (
    PARALLEL_API_BASE,
    SEARCH_MODE_ALIASES,
    ParallelExcerptSettings,
    ParallelExtractRequest,
    ParallelExtractResponse,
    ParallelFetchPolicy,
    ParallelFullContentSettings,
    ParallelSearchRequest,
    ParallelSearchResponse,
    ParallelSourceItem,
    ParallelSourcePolicy,
    ParallelTaskOutput,
    ParallelTaskOutputSchema,
    ParallelTaskRun,
    ParallelTaskRunCreateRequest,
    ParallelTaskRunResult,
    ParallelTaskSpec,
)
from symai.backend.engines.search.utils import (
    Citation,
    CitationResultMixin,
    normalize_domains,
    normalize_url,
)
from symai.backend.request import EngineAPIRequest
from symai.backend.settings import SYMAI_CONFIG
from symai.backend.transport import (
    DEFAULT_RETRIES,
    build_engine_api_error,
    default_engine_api_client,
    execute_engine_api_request,
)
from symai.symbol import Result

logger = logging.getLogger(__name__)


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


class ParallelSearchResult(CitationResultMixin, Result):
    def __init__(self, value, **kwargs) -> None:
        super().__init__(value, **kwargs)
        self._citations = []
        # value is either:
        #   - ParallelSearchResponse (from search) with .results: list[ParallelSearchResultItem]
        #   - list[ParallelSourceItem] (from _task path)
        items = value.results if hasattr(value, "results") else value
        text, citations = self._build_text_and_citations(items)
        self._value = text
        self._citations = citations

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
            normalized_url = normalize_url(url)
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)

            title = item.title or urlsplit(normalized_url).hostname or ""
            excerpts = item.excerpts or []
            excerpt_parts = [p for ex in excerpts if (p := self._sanitize_excerpt(ex))]
            if not excerpt_parts:
                continue

            combined_excerpt = "\n\n".join(excerpt_parts)
            raw_id = urlsplit(normalized_url).hostname or normalized_url
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


class ParallelExtractResult(Result):
    """Result wrapper for Parallel Extract API responses (ParallelExtractResponse)."""

    def __init__(self, value, **kwargs) -> None:
        super().__init__(value, **kwargs)
        # value is a ParallelExtractResponse with .results: list[ParallelExtractResultItem]
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
    # Overall seconds a task route waits for completion before giving up.
    DEFAULT_TASK_TIMEOUT = 600
    # Per-request long-poll seconds sent as the `timeout` query param on GET result.
    DEFAULT_TASK_POLL_TIMEOUT = 600
    # Extra client-side seconds beyond the wire long-poll so the server always answers first.
    POLL_TIMEOUT_BUFFER = 30

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        self.api_key = api_key or self.config.get("SEARCH_ENGINE_API_KEY")
        self.model = self.config.get("SEARCH_ENGINE_MODEL")
        self.name = self.__class__.__name__

        if api_key is None and self.id() != "search":
            return
        self.transport_client = None

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

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _max_retries(self) -> int:
        return self.client_max_retries if self.client_max_retries is not None else DEFAULT_RETRIES

    @staticmethod
    def _map_search_mode(mode) -> str:
        # NOTE: default keeps the legacy behavior (old kwarg default "advanced" -> "agentic").
        if not mode:
            return "agentic"
        normalized = str(mode).strip().lower()
        return SEARCH_MODE_ALIASES.get(normalized, normalized)

    def _coerce_search_queries(self, value) -> list[str]:  # called from forward + prepare
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

    def _build_source_policy(self, kwargs: dict) -> ParallelSourcePolicy | None:
        include = normalize_domains(kwargs.get("allowed_domains"), self.MAX_INCLUDE_DOMAINS)
        exclude = normalize_domains(kwargs.get("excluded_domains"), self.MAX_EXCLUDE_DOMAINS)
        after_date = kwargs.get("after_date")
        if not include and not exclude and not after_date:
            return None
        return ParallelSourcePolicy(
            include_domains=include or None,
            exclude_domains=exclude or None,
            after_date=after_date,
        )

    @staticmethod
    def _coerce_fetch_policy(value) -> ParallelFetchPolicy | None:
        if value is None:
            return None
        if isinstance(value, ParallelFetchPolicy):
            return value
        if isinstance(value, dict):
            return ParallelFetchPolicy.model_validate(value)
        msg = f"fetch_policy must be a dict of wire fields, got {type(value).__name__}."
        raise ValueError(msg)

    @staticmethod
    def _build_task_spec(kwargs: dict) -> ParallelTaskSpec | None:
        output_schema = (
            kwargs.get("task_output_schema")
            or kwargs.get("task_output")
            or kwargs.get("output_schema")
            or kwargs.get("output")
        )
        if output_schema is None:
            return None
        if isinstance(output_schema, dict):
            # A dict output schema is a JSON schema for structured task output.
            return ParallelTaskSpec(
                output_schema=ParallelTaskOutputSchema(type="json", json_schema=output_schema)
            )
        if isinstance(output_schema, str):
            # A string output schema is a natural-language description of the text output.
            return ParallelTaskSpec(
                output_schema=ParallelTaskOutputSchema(type="text", description=output_schema)
            )
        msg = (
            "Invalid task output schema: expected a dict (JSON schema) or a str "
            f"(text description), got {type(output_schema).__name__}."
        )
        raise ValueError(msg)

    def _search(self, queries: list[str], kwargs: dict):
        if not queries:
            msg = "ParallelEngine._search requires at least one query."
            raise ValueError(msg)

        payload = ParallelSearchRequest(
            mode=self._map_search_mode(kwargs.get("mode")),
            objective=kwargs.get("objective"),
            search_queries=queries,
            max_results=kwargs.get("max_results", 10),
            excerpts=ParallelExcerptSettings(
                max_chars_per_result=kwargs.get("max_chars_per_result", 15000),
                max_chars_total=kwargs.get("max_chars_total"),
            ),
            location=kwargs.get("location"),
            source_policy=self._build_source_policy(kwargs),
            fetch_policy=self._coerce_fetch_policy(kwargs.get("fetch_policy")),
            session_id=kwargs.get("session_id"),
            client_model=kwargs.get("client_model"),
        )
        request = EngineAPIRequest(
            provider="parallel",
            operation="search.create",
            payload=payload,
            method="POST",
            url=f"{PARALLEL_API_BASE}/v1beta/search",
            headers=self._headers(),
            timeout=self.client_timeout,
        )
        response = execute_engine_api_request(
            request,
            client=self.transport_client,
            max_retries=self._max_retries(),
        )
        result = ParallelSearchResponse.model_validate(response.json())
        return [ParallelSearchResult(result)], {"raw_output": result}

    def _task(self, queries: list[str], kwargs: dict):
        processor = kwargs.get("processor")
        if not processor or not str(processor).strip():
            msg = "ParallelEngine.task requires a non-empty processor."
            raise ValueError(msg)

        task_input = (
            queries[0]
            if len(queries) == 1
            else "\n\n".join(f"{i}. {q}" for i, q in enumerate(queries, start=1))
        )

        metadata = kwargs.get("metadata")
        mcp_servers = kwargs.get("mcp_servers")
        if mcp_servers:
            mcp_servers = [
                {"type": "url", **server} if isinstance(server, dict) else server
                for server in mcp_servers
            ]

        payload = ParallelTaskRunCreateRequest(
            processor=str(processor).strip(),
            input=task_input,
            task_spec=self._build_task_spec(kwargs),
            metadata=metadata if isinstance(metadata, dict) else None,
            source_policy=self._build_source_policy(kwargs),
            previous_interaction_id=kwargs.get("previous_interaction_id"),
            mcp_servers=mcp_servers or None,
        )
        request = EngineAPIRequest(
            provider="parallel",
            operation="task_run.create",
            payload=payload,
            method="POST",
            url=f"{PARALLEL_API_BASE}/v1/tasks/runs",
            headers=self._headers(),
            timeout=self.client_timeout,
        )
        # NOTE: task run creation answers 202 Accepted; the transport only raises on
        # is_error (4xx/5xx), so 202 passes through as success.
        response = execute_engine_api_request(
            request,
            client=self.transport_client,
            max_retries=self._max_retries(),
        )
        run = ParallelTaskRun.model_validate(response.json())

        task_result = self._poll_task_result(run.run_id, kwargs)
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

    def _poll_task_result(self, run_id: str, kwargs: dict) -> ParallelTaskRunResult:
        try:
            deadline_seconds = float(
                kwargs.get("task_timeout") or kwargs.get("timeout") or self.DEFAULT_TASK_TIMEOUT
            )
        except (TypeError, ValueError) as exc:
            msg = f"task_timeout must be numeric: {exc}"
            raise ValueError(msg) from exc

        api_timeout = kwargs.get("task_api_timeout") or kwargs.get("api_timeout")
        if api_timeout is not None:
            try:
                poll_timeout = int(api_timeout)
            except (TypeError, ValueError) as exc:
                msg = f"api_timeout must be numeric: {exc}"
                raise ValueError(msg) from exc
        else:
            poll_timeout = self.DEFAULT_TASK_POLL_TIMEOUT

        deadline = time.monotonic() + deadline_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                msg = f"Parallel task run {run_id} did not complete within {deadline_seconds}s."
                raise ValueError(msg)
            wire_timeout = max(1, min(poll_timeout, int(remaining)))
            # NOTE: the result endpoint long-polls via GET with only the `timeout`
            # query param. EngineAPIRequest always serializes a JSON body (even `{}`),
            # and the API's edge rejects GET-with-body as a 400 malformed request, so
            # the poll goes through a raw httpx GET on the shared pooled client.
            client = (
                default_engine_api_client()
                if self.transport_client is None
                else self.transport_client
            )
            response = client.get(
                f"{PARALLEL_API_BASE}/v1/tasks/runs/{run_id}/result",
                headers=self._headers(),
                params={"timeout": wire_timeout},
                timeout=wire_timeout + self.POLL_TIMEOUT_BUFFER,
            )
            if response.status_code == 408:
                continue  # still active; poll again until the overall deadline
            if response.status_code == 404:
                msg = f"Parallel task run {run_id} failed or is unknown."
                raise ValueError(msg)
            if response.is_error:
                raise build_engine_api_error(response)
            return ParallelTaskRunResult.model_validate(response.json())

    def _task_output_to_items(
        self, output: ParallelTaskOutput | None
    ) -> tuple[list[ParallelSourceItem], str]:
        """Flatten a task output into ParallelSourceItem entries and prefix text."""
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
                        ParallelSourceItem(url=basis_url, title=field_title, excerpts=[reasoning])
                    )
                else:
                    items.append(
                        ParallelSourceItem(
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
                ParallelSourceItem(
                    url="parallel://task-output",
                    title="Parallel Task Output",
                    excerpts=[snippet],
                )
            )

        return items, "\n\n".join(prefix_parts)

    def _extract(self, url: str, kwargs: dict):
        max_chars_per_result = kwargs.get("max_chars_per_result")
        max_chars_total = kwargs.get("max_chars_total")
        if max_chars_per_result or max_chars_total:
            excerpts: bool | ParallelExcerptSettings | None = ParallelExcerptSettings(
                max_chars_per_result=max_chars_per_result,
                max_chars_total=max_chars_total,
            )
        else:
            excerpts = None  # wire default: excerpts enabled

        full_content = kwargs.get("full_content", False)
        if isinstance(full_content, dict):
            full_content_value: bool | ParallelFullContentSettings | None = (
                ParallelFullContentSettings.model_validate(full_content)
            )
        elif full_content:
            full_content_value = True
        else:
            full_content_value = None  # wire default: full_content disabled

        payload = ParallelExtractRequest(
            urls=[url],
            objective=kwargs.get("objective"),
            search_queries=self._coerce_search_queries(kwargs.get("search_queries")) or None,
            fetch_policy=self._coerce_fetch_policy(kwargs.get("fetch_policy")),
            excerpts=excerpts,
            full_content=full_content_value,
            session_id=kwargs.get("session_id"),
            client_model=kwargs.get("client_model"),
        )
        request = EngineAPIRequest(
            provider="parallel",
            operation="extract.create",
            payload=payload,
            method="POST",
            url=f"{PARALLEL_API_BASE}/v1beta/extract",
            headers=self._headers(),
            timeout=self.client_timeout,
        )
        response = execute_engine_api_request(
            request,
            client=self.transport_client,
            max_retries=self._max_retries(),
        )
        result = ParallelExtractResponse.model_validate(response.json())
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
            msg = "ParallelEngine.forward requires at least one non-empty query or url."
            raise ValueError(msg)
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
