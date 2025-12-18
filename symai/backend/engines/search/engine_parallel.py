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
    from parallel.resources.task_run import build_task_spec_param

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
    def __init__(self, value: dict[str, Any] | Any, **kwargs) -> None:
        super().__init__(value, **kwargs)
        if isinstance(value, dict) and value.get("error"):
            UserMessage(value["error"], raise_with=ValueError)
        self._citations: list[Citation] = []
        try:
            results = self._coerce_results(value)
            task_meta = self._extract_task_metadata(value)
            text, citations = self._build_text_and_citations(results, task_meta=task_meta)
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
        coerced = []
        for item in results:
            if item is None:
                continue
            coerced.append(_item_to_mapping(item))
        return coerced

    def _extract_task_metadata(self, raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        task_output = raw.get("task_output")
        if task_output is None:
            return None
        output_value = task_output.get("output") if isinstance(task_output, dict) else None
        return {
            "reasoning": raw.get("task_reasoning"),
            "answer": output_value,
            "confidence": raw.get("task_confidence"),
        }

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

    def _strip_markdown_links(self, text: str) -> str:
        # Matches Markdown links like "[label](https://example.com "title")" and captures only the label.
        pattern = re.compile(
            r"\[(?P<label>[^\]]+)\]\((?P<url>https?://[^)\s]+)(?:\s+\"[^\"]*\")?\)"
        )

        def _replacement(match: re.Match) -> str:
            label = match.group("label") or ""
            return label.strip()

        cleaned = pattern.sub(_replacement, text)
        # Remove lingering empty parentheses that previously wrapped the stripped links.
        cleaned = re.sub(r"\(\s*\)", "", cleaned)
        # Remove parentheses that contain only commas or whitespace remnants.
        return re.sub(r"\(\s*(,\s*)+\)", "", cleaned)

    def _strip_square_brackets(self, text: str) -> str:
        def _replacement(match: re.Match) -> str:
            return match.group(1) or ""

        # Replace bracketed fragments with their inner text so literal '[' or ']' do not leak into the output.
        return re.sub(r"\[([^\]]*)\]", _replacement, text).replace("[", "").replace("]", "")

    def _sanitize_excerpt(self, text: str) -> str:
        cleaned = self._strip_markdown_links(text)
        cleaned = self._strip_square_brackets(cleaned)
        # Collapse consecutive spaces/tabs down to a single space for readability.
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        # Shrink runs of three or more blank lines to a double newline spacer.
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _build_text_and_citations(
        self, results: list[dict[str, Any]], *, task_meta: dict[str, Any] | None = None
    ):
        pieces = []
        citations = []
        cursor = 0

        if task_meta:
            reasoning = task_meta.get("reasoning")
            answer = task_meta.get("answer")
            confidence = task_meta.get("confidence")

            if reasoning:
                block = f"<reasoning>\n{reasoning}\n</reasoning>"
                pieces.append(block)
                cursor += len(block)

            if answer:
                if pieces:
                    pieces.append("\n\n")
                    cursor += 2
                block = f"<answer>\n{answer}\n</answer>"
                pieces.append(block)
                cursor += len(block)

            if confidence:
                if pieces:
                    pieces.append("\n\n")
                    cursor += 2
                block = f"<answer_confidence>\n{confidence}\n</answer_confidence>"
                pieces.append(block)
                cursor += len(block)

        seen_urls = set()
        cid = 1
        separator = "\n\n---\n\n"

        for item in results:
            url = str(item.get("url") or "")
            if not url:
                continue
            normalized_url = self._normalize_url(url)
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)

            title = str(item.get("title") or "") or urlsplit(normalized_url).netloc
            excerpts = item.get("excerpts") or []
            excerpt_parts = [self._sanitize_excerpt(ex) for ex in excerpts]
            excerpt_parts = [p for p in excerpt_parts if p]
            if not excerpt_parts:
                continue

            combined_excerpt = "\n\n".join(excerpt_parts)
            source_id = self._coerce_source_identifier(
                item, url=normalized_url, fallback=f"source-{cid}"
            )
            block_body = combined_excerpt
            if source_id:
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

    def _coerce_source_identifier(self, item: dict[str, Any], *, url: str, fallback: str) -> str:
        for key in ("source_id", "sourceId", "sourceID", "id"):
            candidate = self._sanitize_source_identifier(item.get(key))
            if candidate:
                return candidate

        split_url = urlsplit(url)
        derived = split_url.netloc or split_url.path or url
        candidate = self._sanitize_source_identifier(derived)
        if candidate:
            return candidate
        return fallback

    def _sanitize_source_identifier(self, raw: Any) -> str:
        if raw is None:
            return ""
        text = str(raw).strip()
        if not text:
            return ""
        # Replace any character outside [A-Za-z0-9._:-] with hyphens so IDs are safe for tag embedding.
        sanitized = re.sub(r"[^A-Za-z0-9._:-]+", "-", text)
        sanitized = sanitized.strip("-")
        return sanitized or ""

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
            content_parts = []
            for r in results:
                full = r.get("full_content")
                if full is not None:
                    content_parts.append(full)
                else:
                    excerpts = r.get("excerpts") or []
                    content_parts.extend(excerpts)
            self._value = "\n\n".join(content_parts)
        except Exception as e:
            self._value = None
            UserMessage(f"Failed to parse Parallel extract response: {e}", raise_with=ValueError)

    def _coerce_results(self, raw: Any) -> list[dict[str, Any]]:
        if raw is None:
            return []
        results = raw.get("results", []) if isinstance(raw, dict) else getattr(raw, "results", None)
        if not results:
            return []
        coerced = []
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

        try:
            self.client = Parallel(api_key=self.api_key)
        except Exception as e:
            UserMessage(f"Failed to initialize Parallel client: {e}", raise_with=ValueError)

    def id(self) -> str:
        # Register as a search engine when configured with the 'parallel' model token
        if (
            self.config.get("SEARCH_ENGINE_API_KEY")
            and str(self.config.get("SEARCH_ENGINE_MODEL", "")).lower() == "parallel"
        ):
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
        seen = set()
        out = []
        for d in domains:
            netloc = self._extract_netloc(d)
            if not netloc or netloc in seen:
                continue
            if not self._is_valid_domain(netloc):
                continue
            seen.add(netloc)
            out.append(netloc)
            if len(out) >= self.MAX_INCLUDE_DOMAINS:
                break
        return out

    def _normalize_exclude_domains(self, domains: list[str] | None) -> list[str]:
        if not isinstance(domains, list):
            return []
        seen = set()
        out = []
        for d in domains:
            netloc = self._extract_netloc(d)
            if not netloc or netloc in seen:
                continue
            if not self._is_valid_domain(netloc):
                continue
            seen.add(netloc)
            out.append(netloc)
            if len(out) >= self.MAX_EXCLUDE_DOMAINS:
                break
        return out

    def _coerce_search_queries(self, value: Any) -> list[str]:
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
        if s.startswith("."):
            # Allow bare domain extensions like .gov or .co.uk
            remainder = s[1:]
            return bool((remainder and "." in remainder) or remainder.isalpha())
        # Require at least one dot and valid label characters
        # Matches a single DNS label: 1-63 chars, alphanumeric at both ends, hyphens allowed internally.
        label_re = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
        parts = s.split(".")
        if len(parts) < 2:
            return False
        return all(label_re.fullmatch(p or "") for p in parts)

    def _search(self, queries: list[str], kwargs: dict[str, Any]):
        if not queries:
            UserMessage(
                "ParallelEngine._search requires at least one query.", raise_with=ValueError
            )

        mode = kwargs.get("mode") or "one-shot"
        max_results = kwargs.get("max_results", 10)
        max_chars_per_result = kwargs.get("max_chars_per_result", 15000)
        excerpts = {"max_chars_per_result": max_chars_per_result}
        include = self._normalize_include_domains(kwargs.get("allowed_domains"))
        exclude = self._normalize_exclude_domains(kwargs.get("excluded_domains"))
        source_policy = None
        if include or exclude:
            source_policy = {}
            if include:
                source_policy["include_domains"] = include
            if exclude:
                source_policy["exclude_domains"] = exclude
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
        return [SearchResult(result)], {"raw_output": result}

    def _task(self, queries: list[str], kwargs: dict[str, Any]):
        processor_name = self._coerce_processor(kwargs.get("processor"))
        task_input = self._compose_task_input(queries)

        include = self._normalize_include_domains(kwargs.get("allowed_domains"))
        exclude = self._normalize_exclude_domains(kwargs.get("excluded_domains"))
        source_policy = None
        if include or exclude:
            source_policy = {}
            if include:
                source_policy["include_domains"] = include
            if exclude:
                source_policy["exclude_domains"] = exclude
        metadata = self._coerce_metadata(kwargs.get("metadata"))

        output_schema = (
            kwargs.get("task_output_schema")
            or kwargs.get("task_output")
            or kwargs.get("output_schema")
            or kwargs.get("output")
        )
        task_spec_param = self._build_task_spec(output_schema, task_input)
        timeout, api_timeout = self._collect_task_timeouts(kwargs)

        run = self._create_task_run(
            task_input=task_input,
            processor=processor_name,
            metadata=metadata,
            source_policy=source_policy,
            task_spec=task_spec_param,
        )
        result = self._fetch_task_result(run.run_id, timeout=timeout, api_timeout=api_timeout)

        payload = self._task_result_to_search_payload(result)
        return [SearchResult(payload)], {
            "raw_output": result,
            "task_output": payload.get("task_output"),
            "task_output_type": payload.get("task_output_type"),
        }

    def _coerce_processor(self, processor: Any) -> str:
        if processor is None:
            UserMessage("ParallelEngine.task requires a processor.", raise_with=ValueError)
        value = processor.strip() if isinstance(processor, str) else str(processor).strip()
        if not value:
            UserMessage(
                "ParallelEngine.task requires a non-empty processor.", raise_with=ValueError
            )
        return value

    def _compose_task_input(self, queries: list[str]) -> str:
        if not queries:
            UserMessage(
                "ParallelEngine.task requires at least one query input.", raise_with=ValueError
            )
        if len(queries) == 1:
            return queries[0]
        return "\n\n".join(f"{idx}. {q}" for idx, q in enumerate(queries, start=1))

    def _coerce_metadata(self, metadata: Any) -> dict[str, Any] | None:
        if metadata is None or isinstance(metadata, dict):
            return metadata
        return None

    def _build_task_spec(self, output_schema: Any, task_input: str):
        if output_schema is None:
            return None
        try:
            return build_task_spec_param(output_schema, task_input)
        except Exception as exc:
            UserMessage(f"Invalid task output schema: {exc}", raise_with=ValueError)

    def _collect_task_timeouts(self, kwargs: dict[str, Any]) -> tuple[Any, int | None]:
        timeout = kwargs.get("task_timeout") or kwargs.get("timeout")
        api_timeout = kwargs.get("task_api_timeout") or kwargs.get("api_timeout")
        if api_timeout is None:
            return timeout, None
        try:
            return timeout, int(api_timeout)
        except (TypeError, ValueError) as exc:
            UserMessage(f"api_timeout must be numeric: {exc}", raise_with=ValueError)

    def _create_task_run(
        self,
        *,
        task_input: str,
        processor: str,
        metadata: dict[str, Any] | None,
        source_policy: dict[str, Any] | None,
        task_spec: Any,
    ):
        task_kwargs = {
            "input": task_input,
            "processor": processor,
        }
        if metadata is not None:
            task_kwargs["metadata"] = metadata
        if source_policy is not None:
            task_kwargs["source_policy"] = source_policy
        if task_spec is not None:
            task_kwargs["task_spec"] = task_spec

        try:
            return self.client.task_run.create(**task_kwargs)
        except Exception as e:
            UserMessage(f"Failed to create Parallel task: {e}", raise_with=ValueError)

    def _fetch_task_result(self, run_id: str, *, timeout: Any, api_timeout: int | None):
        result_kwargs = {}
        if api_timeout is not None:
            result_kwargs["api_timeout"] = api_timeout
        if timeout is not None:
            result_kwargs["timeout"] = timeout
        try:
            return self.client.task_run.result(run_id, **result_kwargs)
        except Exception as e:
            UserMessage(f"Failed to fetch Parallel task result: {e}", raise_with=ValueError)

    def _task_result_to_search_payload(self, task_result: Any) -> dict[str, Any]:
        payload = {"results": []}
        output = task_result.output
        if output is None:
            return payload

        basis_items = output.basis or []
        for idx, basis in enumerate(basis_items):
            payload["results"].extend(self._basis_to_results(basis, basis_index=idx))

        if not payload["results"]:
            payload["results"].append(self._task_fallback_result(output, basis_items))

        payload["task_output"] = output.content
        payload["task_output_type"] = output.type

        if basis_items:
            first_basis = basis_items[0]
            payload["task_reasoning"] = first_basis.reasoning
            payload["task_confidence"] = first_basis.confidence

        return payload

    def _basis_to_results(self, basis: Any, *, basis_index: int) -> list[dict[str, Any]]:
        reasoning = basis.reasoning or ""
        field_title = basis.field or ""
        if not field_title.strip():
            field_title = "Parallel Task Output"
        citations = basis.citations or []
        if not citations:
            if not reasoning:
                return []
            citations = [None]

        results = []
        # Convert field titles to lowercase slugs by swapping non-alphanumerics for hyphens.
        slug = re.sub(r"[^a-z0-9]+", "-", field_title.lower()).strip("-") or "field"
        basis_url = f"parallel://task-output/{basis_index:04d}-{slug}"
        for citation in citations:
            if citation is None:
                url = basis_url
                title = field_title
                excerpts = [reasoning]
            else:
                url = str(citation.url or "")
                title = str(citation.title or field_title)
                excerpts = citation.excerpts or []
                if not excerpts and reasoning:
                    excerpts = [reasoning]
            results.append(
                {
                    "url": url or basis_url,
                    "title": title or field_title,
                    "excerpts": excerpts or ([reasoning] if reasoning else []),
                }
            )
        return results

    def _task_fallback_result(self, output: Any, basis_items: list[Any]) -> dict[str, Any]:
        content = output.content
        if isinstance(content, str):
            snippet = content
        elif isinstance(content, (dict, list)):
            snippet = json.dumps(content, ensure_ascii=False)
        else:
            snippet = str(content or "")
        if not snippet:
            extra_reasoning = []
            for basis in basis_items:
                raw_value = basis.reasoning or ""
                if isinstance(raw_value, str):
                    extra_reasoning.append(raw_value)
                else:
                    extra_reasoning.append(str(raw_value))
            snippet = " ".join(r for r in extra_reasoning if r) or "Parallel task output"
        return {
            "url": "parallel://task-output",
            "title": "Parallel Task Output",
            "excerpts": [snippet],
        }

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
