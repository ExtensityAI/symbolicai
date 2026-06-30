import logging
from copy import deepcopy

import httpx
from openai import OpenAI

from symai.backend.base import Engine
from symai.backend.engines.search.utils import (
    CitationResultMixin,
    insert_citation_markers,
    normalize_domains,
)
from symai.backend.mixin import OPENAI_CHAT_MODELS, OPENAI_REASONING_MODELS
from symai.backend.settings import SYMAI_CONFIG
from symai.symbol import Result

logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

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
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        if api_key is not None and model is not None:
            self.config["SEARCH_ENGINE_API_KEY"] = api_key
            self.config["SEARCH_ENGINE_MODEL"] = model
        self.api_key = self.config.get("SEARCH_ENGINE_API_KEY")
        self.model = self.config.get(
            "SEARCH_ENGINE_MODEL", "gpt-4.1"
        )  # Default to gpt-4.1 as per docs
        self.client_timeout = client_timeout
        self.name = self.__class__.__name__

        if api_key is None and model is None and self.id() != "search":
            return

        try:
            if self.client_timeout is not None:
                # Socket-level timeout so a hung web_search call raises instead of blocking
                # the caller indefinitely (the search engine otherwise has no timeout). Keep
                # max_retries low so the timeout actually terminates the request rather than
                # being retried away by the SDK's default retry loop.
                self.client = OpenAI(
                    api_key=self.api_key,
                    timeout=httpx.Timeout(self.client_timeout, connect=10.0),
                    max_retries=1,
                )
            else:
                self.client = OpenAI(api_key=self.api_key)
        except Exception as e:
            msg = f"Failed to initialize OpenAI client: {e}"
            raise ValueError(msg) from e

    def id(self) -> str:
        if (
            self.config.get("SEARCH_ENGINE_API_KEY")
            and self.config.get("SEARCH_ENGINE_MODEL")
            in OPENAI_CHAT_MODELS + OPENAI_REASONING_MODELS
        ):
            return "search"
        return super().id()  # default to unregistered

    def _normalize_allowed_domains(self, domains: list[str] | None) -> list[str]:
        return normalize_domains(domains, self.MAX_ALLOWED_DOMAINS)

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "SEARCH_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["SEARCH_ENGINE_API_KEY"]
        if "SEARCH_ENGINE_MODEL" in kwargs:
            self.model = kwargs["SEARCH_ENGINE_MODEL"]

    def forward(self, argument):
        messages = argument.prop.prepared_input
        kwargs = argument.kwargs

        tool_definition = {"type": "web_search"}
        user_location = kwargs.get("user_location")
        if user_location:
            tool_definition["user_location"] = user_location

        allowed_domains = self._normalize_allowed_domains(kwargs.get("allowed_domains"))
        if allowed_domains:
            tool_definition["filters"] = {"allowed_domains": allowed_domains}

        self.model = kwargs.get(
            "model", self.model
        )  # Important for MetadataTracker to work correctly

        payload = {
            "model": self.model,
            "input": messages,
            "tools": [tool_definition],
            "tool_choice": {"type": "web_search"}
            if self.model not in OPENAI_REASONING_MODELS
            else "auto",  # force the use of web search tool for non-reasoning models
        }

        if self.model in OPENAI_REASONING_MODELS:
            reasoning = kwargs.get("reasoning", {"effort": "low", "summary": "auto"})
            payload["reasoning"] = reasoning

        try:
            res = self.client.responses.create(**payload)
            res = OpenAISearchResult(res.dict())
        except Exception as e:
            msg = f"Failed to make request: {e}"
            raise ValueError(msg) from e

        metadata = {"raw_output": res.raw}
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
