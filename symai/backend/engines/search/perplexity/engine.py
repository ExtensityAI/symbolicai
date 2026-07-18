import json
import logging

from symai.backend.base import Engine
from symai.backend.engines.search.perplexity.models import (
    PERPLEXITY_CHAT_COMPLETIONS_URL,
    PerplexityRequestPayload,
    PerplexityResponse,
)
from symai.backend.request import EngineAPIRequest
from symai.backend.settings import SYMAI_CONFIG
from symai.backend.transport import DEFAULT_RETRIES, execute_engine_api_request
from symai.symbol import Result
from symai.utils import silence_noisy_loggers

silence_noisy_loggers()

logger = logging.getLogger(__name__)


class PerplexitySearchResult(Result):
    def __init__(self, value, **kwargs) -> None:
        super().__init__(value, **kwargs)
        if value.get("error"):
            msg = value["error"]
            raise ValueError(msg)
        try:
            self._value = value["choices"][0]["message"]["content"]
        except Exception as e:
            self._value = None
            msg = f"Failed to parse response: {e}"
            raise ValueError(msg) from e

    def __str__(self) -> str:
        try:
            return json.dumps(self.raw, indent=2)
        except TypeError:
            return str(self.raw)

    def _repr_html_(self) -> str:
        try:
            return f"<pre>{json.dumps(self.raw, indent=2)}</pre>"
        except TypeError:
            return f"<pre>{self.raw!s}</pre>"


class PerplexityEngine(Engine):
    def __init__(self):
        super().__init__()
        self.config = SYMAI_CONFIG
        self.api_key = self.config["SEARCH_ENGINE_API_KEY"]
        self.model = self.config["SEARCH_ENGINE_MODEL"]
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
        request = EngineAPIRequest(
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
        max_retries = (
            self.client_max_retries if self.client_max_retries is not None else DEFAULT_RETRIES
        )
        response = execute_engine_api_request(request, max_retries=max_retries)
        perplexity_response = PerplexityResponse.model_validate(response.json())

        res = PerplexitySearchResult(perplexity_response.model_dump())

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
