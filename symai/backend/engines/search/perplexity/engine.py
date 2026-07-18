import logging
import re
from copy import deepcopy
from urllib.parse import urlsplit

from symai.backend.base import Engine
from symai.backend.engines.search.perplexity.models import (
    PERPLEXITY_CHAT_COMPLETIONS_URL,
    PerplexityRequestPayload,
    PerplexityResponse,
)
from symai.backend.engines.search.utils import Citation, CitationResult, normalize_url
from symai.backend.request import EngineAPIRequest
from symai.backend.settings import SYMAI_CONFIG
from symai.backend.transport import DEFAULT_RETRIES, execute_engine_api_request
from symai.symbol import Result
from symai.utils import silence_noisy_loggers

silence_noisy_loggers()

logger = logging.getLogger(__name__)

# Inline source markers in the response content, e.g. "...2-1[1]." — n is a 1-based
# index into the response's top-level `citations` URL list.
_MARKER_RE = re.compile(r"\[(\d+)\]")


class PerplexitySearchResult(CitationResult, Result):
    def __init__(self, value, **kwargs) -> None:
        super().__init__(value, **kwargs)
        if value.get("error"):
            msg = value["error"]
            raise ValueError(msg)
        try:
            self._value = value["choices"][0]["message"]["content"]
            self._citations = self._extract_citations(self._value, value.get("citations") or [])
        except Exception as e:
            self._value = None
            self._citations = []
            msg = f"Failed to parse response: {e}"
            raise ValueError(msg) from e

    @staticmethod
    def _extract_citations(text: str, urls: list[str]) -> list[Citation]:
        # The wire inlines [n] markers in the content; n is a 1-based index into the
        # top-level citations list of URLs. The text (and thus its markers) is kept
        # unchanged, so each citation's id equals its marker number and its span covers
        # the marker's first occurrence.
        citations = {}
        for m in _MARKER_RE.finditer(text):
            n = int(m.group(1))
            if n < 1 or n > len(urls) or n in citations:
                continue
            url = normalize_url(urls[n - 1])
            title = urlsplit(url).hostname or ""
            citations[n] = Citation(id=n, title=title, url=url, start=m.start(), end=m.end())
        return [citations[n] for n in sorted(citations)]


class PerplexityEngine(Engine):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        if api_key is not None and model is not None:
            self.config["SEARCH_ENGINE_API_KEY"] = api_key
            self.config["SEARCH_ENGINE_MODEL"] = model
        self.api_key = self.config.get("SEARCH_ENGINE_API_KEY")
        self.model = self.config.get("SEARCH_ENGINE_MODEL")
        self.transport_client = None
        self.name = self.__class__.__name__

    def id(self) -> str:
        if self.config.get("SEARCH_ENGINE_API_KEY") and self.config.get(
            "SEARCH_ENGINE_MODEL"
        ).startswith("sonar"):
            return "search"
        return super().id()  # default to unregistered

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
        messages = argument.prop.prepared_input
        kwargs = argument.kwargs

        payload = PerplexityRequestPayload(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", None),
            temperature=kwargs.get("temperature", 0.2),
            top_p=kwargs.get("top_p", 0.9),
            top_k=kwargs.get("top_k", 0),
            presence_penalty=kwargs.get("presence_penalty", 0),
            frequency_penalty=kwargs.get("frequency_penalty", 1),
            response_format=kwargs.get("response_format", None),
            search_domain_filter=kwargs.get("search_domain_filter", []),
            return_images=kwargs.get("return_images", False),
            return_related_questions=kwargs.get("return_related_questions", False),
            search_recency_filter=kwargs.get("search_recency_filter", "month"),
            web_search_options=kwargs.get("web_search_options", None),
        )
        return EngineAPIRequest(
            provider="perplexity",
            operation="chat.completions.create",
            payload=payload,
            method="POST",
            url=PERPLEXITY_CHAT_COMPLETIONS_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    def call_request(self, request: EngineAPIRequest) -> PerplexityResponse:
        max_retries = (
            self.client_max_retries if self.client_max_retries is not None else DEFAULT_RETRIES
        )
        response = execute_engine_api_request(
            request, client=self.transport_client, max_retries=max_retries
        )
        return PerplexityResponse.model_validate(response.json())

    def parse_response(self, response: PerplexityResponse):
        res = PerplexitySearchResult(response.model_dump())
        metadata = {"raw_output": response}
        return [res], metadata

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
