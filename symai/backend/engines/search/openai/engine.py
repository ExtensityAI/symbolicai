from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING

from symai.backend.base import Engine
from symai.backend.engines.neurosymbolic.openai.models import (
    SUPPORTED_CHAT_MODELS,
    SUPPORTED_REASONING_MODELS,
)
from symai.backend.engines.search.openai.models import (
    OPENAI_RESPONSES_URL,
    OpenAISearchRequestPayload,
    OpenAISearchResponse,
    OpenAISearchTool,
)
from symai.backend.engines.search.utils import (
    CitationResultMixin,
    insert_citation_markers,
    normalize_domains,
)
from symai.backend.request import EngineAPIRequest
from symai.backend.settings import SYMAI_CONFIG
from symai.backend.transport import DEFAULT_RETRIES, execute_engine_api_request
from symai.symbol import Result
from symai.utils import silence_noisy_loggers

if TYPE_CHECKING:
    from pydantic import JsonValue

silence_noisy_loggers()

logger = logging.getLogger(__name__)


class OpenAISearchResult(CitationResultMixin, Result):
    def __init__(self, value, **kwargs) -> None:
        super().__init__(value, **kwargs)
        if value.get("error"):
            msg = value["error"]
            raise ValueError(msg)
        try:
            text, annotations = self._extract_text_and_annotations(value)
            if text is None:
                self._value = None
                self._citations = []
                return
            self._value, self._citations = insert_citation_markers(text, annotations)

        except Exception as e:
            self._value = None
            msg = f"Failed to parse response: {e}"
            raise ValueError(msg) from e

    def _extract_text_and_annotations(self, value):
        segments = []
        global_annotations = []
        pos = 0
        for output in value.get("output", []) or []:
            if output.get("type") != "message" or not output.get("content"):
                continue
            for content in output.get("content", []) or []:
                seg_text = content.get("text") or ""
                if not isinstance(seg_text, str):
                    continue
                for ann in content.get("annotations") or []:
                    if ann.get("type") == "url_citation" and ann.get("url"):
                        start = ann.get("start_index", 0)
                        end = ann.get("end_index", 0)
                        global_annotations.append(
                            {
                                "type": "url_citation",
                                "url": ann.get("url"),
                                "title": (ann.get("title") or "").strip(),
                                "start_index": pos + int(start),
                                "end_index": pos + int(end),
                            }
                        )
                segments.append(seg_text)
                pos += len(seg_text)

        built_text = "".join(segments) if segments else None
        # Prefer top-level output_text if present AND segments are empty (no way to compute indices)
        if not built_text and isinstance(value.get("output_text"), str):
            return value.get("output_text"), []
        return built_text, global_annotations


class GPTXSearchEngine(Engine):
    MAX_ALLOWED_DOMAINS = 20

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        client_timeout: float | None = None,
    ):
        super().__init__(client_timeout=client_timeout)
        self.name = self.__class__.__name__
        self.config = deepcopy(SYMAI_CONFIG)
        if api_key is not None and model is not None:
            self.config["SEARCH_ENGINE_API_KEY"] = api_key
            self.config["SEARCH_ENGINE_MODEL"] = model
        self.api_key = self.config.get("SEARCH_ENGINE_API_KEY")
        self.model = self.config.get(
            "SEARCH_ENGINE_MODEL", "gpt-4.1"
        )  # Default to gpt-4.1 as per docs
        self.transport_client = None

    def id(self) -> str:
        if (
            self.config.get("SEARCH_ENGINE_API_KEY")
            and self.config.get("SEARCH_ENGINE_MODEL")
            in SUPPORTED_CHAT_MODELS + SUPPORTED_REASONING_MODELS
        ):
            return "search"
        return super().id()  # default to unregistered

    def _normalize_allowed_domains(self, domains: list[str] | None) -> list[str]:
        return normalize_domains(domains, self.MAX_ALLOWED_DOMAINS)

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "SEARCH_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["SEARCH_ENGINE_API_KEY"]
            # NOTE: auth headers are built per request on the shared transport, so a key
            # change only needs the cached transport handle dropped, not a client rebuild.
            self.transport_client = None
        if "SEARCH_ENGINE_MODEL" in kwargs:
            self.model = kwargs["SEARCH_ENGINE_MODEL"]

    def forward(self, argument):
        messages = argument.prop.prepared_input
        kwargs = argument.kwargs

        tool_definition: dict[str, JsonValue] = {"type": "web_search"}
        user_location = kwargs.get("user_location")
        if user_location:
            tool_definition["user_location"] = user_location

        allowed_domains = self._normalize_allowed_domains(kwargs.get("allowed_domains"))
        if allowed_domains:
            tool_definition["filters"] = {"allowed_domains": allowed_domains}

        self.model = kwargs.get(
            "model", self.model
        )  # Important for MetadataTracker to work correctly

        is_reasoning = self.model in SUPPORTED_REASONING_MODELS
        payload = OpenAISearchRequestPayload(
            model=self.model,
            input=messages,
            tools=[OpenAISearchTool.model_validate(tool_definition)],
            # force the use of web search tool for non-reasoning models
            tool_choice="auto" if is_reasoning else {"type": "web_search"},
            reasoning=kwargs.get("reasoning", {"effort": "low", "summary": "auto"})
            if is_reasoning
            else None,
        )
        request = EngineAPIRequest(
            provider="openai",
            operation="responses.create",
            payload=payload,
            method="POST",
            url=OPENAI_RESPONSES_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.client_timeout,
        )
        max_retries = (
            self.client_max_retries if self.client_max_retries is not None else DEFAULT_RETRIES
        )
        response = execute_engine_api_request(
            request,
            client=self.transport_client,
            max_retries=max_retries,
        )
        search_response = OpenAISearchResponse.model_validate(response.json())

        res = OpenAISearchResult(search_response.model_dump(exclude_none=True))

        metadata = {"raw_output": search_response}
        output = [res]

        return output, metadata

    def prepare(self, argument):
        system_message = (
            "You are a helpful AI assistant. Be precise and informative."
            if argument.kwargs.get("system_message") is None
            else argument.kwargs.get("system_message")
        )

        res = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{argument.prop.query}"},
        ]
        argument.prop.prepared_input = res
