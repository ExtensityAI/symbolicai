import json
import logging
import re
import time
from copy import deepcopy
from urllib.parse import urlsplit

from symai.backend.base import Engine
from symai.backend.engines.search.parallel.models import (
    PARALLEL_API_BASE,
    PARALLEL_EXTRACT_PATH,
    PARALLEL_SEARCH_PATH,
    PARALLEL_TASK_RUNS_PATH,
    ParallelExcerptSettings,
    ParallelExtractAdvancedSettings,
    ParallelExtractRequest,
    ParallelExtractResponse,
    ParallelFetchPolicy,
    ParallelFullContentSettings,
    ParallelMCPServer,
    ParallelSearchAdvancedSettings,
    ParallelSearchRequest,
    ParallelSearchResponse,
    ParallelSourceItem,
    ParallelSourcePolicy,
    ParallelTaskAdvancedSettings,
    ParallelTaskOutput,
    ParallelTaskOutputSchema,
    ParallelTaskPollOptions,
    ParallelTaskRun,
    ParallelTaskRunCreateRequest,
    ParallelTaskRunResult,
    ParallelTaskSpec,
)
from symai.backend.engines.search.utils import (
    Citation,
    CitationResult,
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


class ParallelSearchResult(CitationResult, Result):
    def __init__(self, value, **kwargs) -> None:
        super().__init__(value, **kwargs)
        self._citations = []
        # value is either:
        #   - ParallelSearchResponse (from search) with .results: list[ParallelSearchResultItem]
        #   - list[ParallelSourceItem] (from the task route)
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
    # Overall seconds the task route waits for completion before giving up.
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

    def forward(self, argument):
        request = self.build_request(argument)
        response = self.call_request(request)
        return self.parse_response(response)

    def build_request(self, argument) -> EngineAPIRequest:
        kwargs = argument.kwargs
        url = argument.prop.url or kwargs.get("url")
        if url is not None:
            return self._build_extract_request(url, kwargs)

        raw_query = argument.prop.prepared_input
        if raw_query is None:
            raw_query = argument.prop.query
        search_queries = self._normalize_queries(raw_query)
        if not any(q.strip() for q in search_queries):
            msg = "ParallelEngine requires at least one non-empty query or a url."
            raise ValueError(msg)

        if kwargs.get("processor") is not None:
            return self._build_task_create_request(search_queries, kwargs)
        return self._build_search_request(search_queries, kwargs)

    def call_request(self, request: EngineAPIRequest):
        response = execute_engine_api_request(
            request,
            client=self.transport_client,
            max_retries=self._max_retries(),
        )
        if request.operation == "search":
            return ParallelSearchResponse.model_validate(response.json())
        if request.operation == "extract":
            return ParallelExtractResponse.model_validate(response.json())
        # operation == "task_create": creation answers 202 Accepted; the transport only
        # raises on is_error (4xx/5xx), so 202 passes through as success.
        run = ParallelTaskRun.model_validate(response.json())
        return self._poll_task_result(run.run_id, request.call_options)

    def parse_response(self, response):
        if isinstance(response, ParallelSearchResponse):
            return [ParallelSearchResult(response)], {"raw_output": response}
        if isinstance(response, ParallelExtractResponse):
            final_url = response.results[0].url if response.results else None
            return [ParallelExtractResult(response)], {
                "raw_output": response,
                "final_url": final_url,
            }
        # ParallelTaskRunResult from the task route.
        output = response.output
        items, prefix = self._task_output_to_items(output)
        wrapped = ParallelSearchResult(items)
        if prefix:
            offset = len(prefix) + (2 if wrapped._value else 0)
            for c in wrapped._citations:
                c.start += offset
                c.end += offset
            wrapped._value = prefix + ("\n\n" + wrapped._value if wrapped._value else "")
        wrapped.raw = response
        return [wrapped], {
            "raw_output": response,
            "task_output": output.content if output else None,
            "task_output_type": output.type if output else None,
        }

    def prepare(self, argument):
        # For scraping: store the URL directly. For search: normalize to a list[str] of queries.
        url = argument.kwargs.get("url") or argument.prop.url
        if url is not None:
            if not isinstance(url, str):
                msg = f"url must be a str, got {type(url).__name__}."
                raise TypeError(msg)
            argument.prop.prepared_input = url
            return
        argument.prop.prepared_input = self._normalize_queries(argument.prop.query)

    # --- request builders ---

    def _build_search_request(self, queries: list[str], kwargs: dict) -> EngineAPIRequest:
        payload = ParallelSearchRequest(
            mode=kwargs.get("mode", "advanced"),
            objective=kwargs.get("objective"),
            search_queries=queries,
            max_chars_total=kwargs.get("max_chars_total"),
            session_id=kwargs.get("session_id"),
            client_model=kwargs.get("client_model"),
            advanced_settings=self._build_search_advanced_settings(kwargs),
        )
        return EngineAPIRequest(
            provider="parallel",
            operation="search",
            payload=payload,
            method="POST",
            url=f"{PARALLEL_API_BASE}{PARALLEL_SEARCH_PATH}",
            headers=self._headers(),
            timeout=self.client_timeout,
        )

    def _build_extract_request(self, url, kwargs: dict) -> EngineAPIRequest:
        if not isinstance(url, str):
            msg = f"url must be a str, got {type(url).__name__}."
            raise TypeError(msg)
        payload = ParallelExtractRequest(
            urls=[url],
            objective=kwargs.get("objective"),
            search_queries=(
                self._normalize_queries(kwargs["search_queries"])
                if kwargs.get("search_queries") is not None
                else None
            ),
            max_chars_total=kwargs.get("max_chars_total"),
            session_id=kwargs.get("session_id"),
            client_model=kwargs.get("client_model"),
            advanced_settings=self._build_extract_advanced_settings(kwargs),
        )
        return EngineAPIRequest(
            provider="parallel",
            operation="extract",
            payload=payload,
            method="POST",
            url=f"{PARALLEL_API_BASE}{PARALLEL_EXTRACT_PATH}",
            headers=self._headers(),
            timeout=self.client_timeout,
        )

    def _build_task_create_request(self, queries: list[str], kwargs: dict) -> EngineAPIRequest:
        processor = kwargs.get("processor")
        if not isinstance(processor, str) or not processor.strip():
            msg = "ParallelEngine task route requires a non-empty 'processor' string."
            raise ValueError(msg)

        task_input = (
            queries[0]
            if len(queries) == 1
            else "\n\n".join(f"{i}. {q}" for i, q in enumerate(queries, start=1))
        )

        metadata = kwargs.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            msg = f"metadata must be a dict of str, got {type(metadata).__name__}."
            raise TypeError(msg)

        mcp_servers = kwargs.get("mcp_servers")
        if mcp_servers is not None:
            mcp_servers = [
                server
                if isinstance(server, ParallelMCPServer)
                else ParallelMCPServer.model_validate(server)
                for server in mcp_servers
            ]

        location = kwargs.get("location")
        payload = ParallelTaskRunCreateRequest(
            processor=processor.strip(),
            input=task_input,
            task_spec=self._build_task_spec(kwargs),
            metadata=metadata,
            source_policy=self._build_source_policy(kwargs),
            advanced_settings=(
                ParallelTaskAdvancedSettings(location=location) if location else None
            ),
            previous_interaction_id=kwargs.get("previous_interaction_id"),
            mcp_servers=mcp_servers,
        )
        return EngineAPIRequest(
            provider="parallel",
            operation="task_create",
            payload=payload,
            call_options=self._build_task_poll_options(kwargs),
            method="POST",
            url=f"{PARALLEL_API_BASE}{PARALLEL_TASK_RUNS_PATH}",
            headers=self._headers(),
            timeout=self.client_timeout,
        )

    # --- payload section builders ---

    @staticmethod
    def _normalize_queries(value) -> list[str]:
        # Trust boundary: a query is a str or a list[str]; anything else is a caller bug.
        if isinstance(value, str):
            return [value]
        if isinstance(value, list) and all(isinstance(q, str) for q in value):
            return value
        msg = f"query must be a str or list[str], got {type(value).__name__}."
        raise TypeError(msg)

    def _build_search_advanced_settings(
        self, kwargs: dict
    ) -> ParallelSearchAdvancedSettings | None:
        source_policy = self._build_source_policy(kwargs)
        fetch_policy = self._build_fetch_policy(kwargs.get("fetch_policy"))
        excerpt_settings = None
        if kwargs.get("max_chars_per_result") is not None:
            excerpt_settings = ParallelExcerptSettings(
                max_chars_per_result=kwargs["max_chars_per_result"]
            )
        location = kwargs.get("location")
        max_results = kwargs.get("max_results")
        if (
            source_policy is None
            and fetch_policy is None
            and excerpt_settings is None
            and location is None
            and max_results is None
        ):
            return None
        return ParallelSearchAdvancedSettings(
            source_policy=source_policy,
            fetch_policy=fetch_policy,
            excerpt_settings=excerpt_settings,
            location=location,
            max_results=max_results,
        )

    def _build_extract_advanced_settings(
        self, kwargs: dict
    ) -> ParallelExtractAdvancedSettings | None:
        fetch_policy = self._build_fetch_policy(kwargs.get("fetch_policy"))
        excerpt_settings = None
        if kwargs.get("max_chars_per_result") is not None:
            excerpt_settings = ParallelExcerptSettings(
                max_chars_per_result=kwargs["max_chars_per_result"]
            )
        full_content = kwargs.get("full_content", False)
        if (
            fetch_policy is None
            and excerpt_settings is None
            and (full_content is False or full_content is None)
        ):
            return None
        return ParallelExtractAdvancedSettings(
            fetch_policy=fetch_policy,
            excerpt_settings=excerpt_settings,
            full_content=self._build_full_content(full_content),
        )

    @staticmethod
    def _build_full_content(value) -> bool | ParallelFullContentSettings:
        if isinstance(value, bool):
            return value
        if isinstance(value, dict):
            return ParallelFullContentSettings.model_validate(value)
        msg = f"full_content must be a bool or a dict of wire fields, got {type(value).__name__}."
        raise TypeError(msg)

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
    def _build_fetch_policy(value) -> ParallelFetchPolicy | None:
        if value is None:
            return None
        if isinstance(value, ParallelFetchPolicy):
            return value
        if isinstance(value, dict):
            return ParallelFetchPolicy.model_validate(value)
        msg = f"fetch_policy must be a dict of wire fields, got {type(value).__name__}."
        raise TypeError(msg)

    @staticmethod
    def _build_task_spec(kwargs: dict) -> ParallelTaskSpec | None:
        # NOTE: output_schema aliases mirror the old SDK-based engine's fallback chain.
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
            "Invalid output_schema: expected a dict (JSON schema) or a str "
            f"(text description), got {type(output_schema).__name__}."
        )
        raise TypeError(msg)

    @staticmethod
    def _build_task_poll_options(kwargs: dict) -> ParallelTaskPollOptions:
        # NOTE: `timeout` / `api_timeout` are the old SDK-based engine's aliases for
        # the task-poll knobs; the explicit task_* names win when both are given.
        task_timeout = kwargs.get("task_timeout")
        if task_timeout is None:
            task_timeout = kwargs.get("timeout")
        if task_timeout is not None and (
            isinstance(task_timeout, bool) or not isinstance(task_timeout, (int, float))
        ):
            msg = f"task_timeout must be a number of seconds, got {type(task_timeout).__name__}."
            raise TypeError(msg)
        task_api_timeout = kwargs.get("task_api_timeout")
        if task_api_timeout is None:
            task_api_timeout = kwargs.get("api_timeout")
        if task_api_timeout is not None and (
            isinstance(task_api_timeout, bool) or not isinstance(task_api_timeout, int)
        ):
            msg = f"task_api_timeout must be whole seconds, got {type(task_api_timeout).__name__}."
            raise TypeError(msg)
        return ParallelTaskPollOptions(
            task_timeout=task_timeout,
            task_api_timeout=task_api_timeout,
        )

    # --- task result polling ---

    def _poll_task_result(
        self, run_id: str, options: ParallelTaskPollOptions | None
    ) -> ParallelTaskRunResult:
        deadline_seconds = float(
            options.task_timeout if options and options.task_timeout else self.DEFAULT_TASK_TIMEOUT
        )
        poll_timeout = int(
            options.task_api_timeout
            if options and options.task_api_timeout
            else self.DEFAULT_TASK_POLL_TIMEOUT
        )

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
                f"{PARALLEL_API_BASE}{PARALLEL_TASK_RUNS_PATH}/{run_id}/result",
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

    # --- transport helpers ---

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _max_retries(self) -> int:
        return self.client_max_retries if self.client_max_retries is not None else DEFAULT_RETRIES
