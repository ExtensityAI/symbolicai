import json
import logging

import requests

from ....symbol import Result
from ....utils import CustomUserWarning
from ...base import Engine
from ...settings import SYMAI_CONFIG

logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class SearchResult(Result):
    def __init__(self, value, **kwargs) -> None:
        super().__init__(value, **kwargs)
        if value.get('error'):
            CustomUserWarning(value['error'], raise_with=ValueError)
        try:
            self._value = value['choices'][0]['message']['content']
        except Exception as e:
            self._value = None
            CustomUserWarning(f"Failed to parse response: {e}", raise_with=ValueError)

    def __str__(self) -> str:
        try:
            return json.dumps(self.raw, indent=2)
        except TypeError:
            return str(self.raw)

    def _repr_html_(self) -> str:
        try:
            return f"<pre>{json.dumps(self.raw, indent=2)}</pre>"
        except TypeError:
            return f"<pre>{str(self.raw)}</pre>"


class PerplexityEngine(Engine):
    def __init__(self):
        super().__init__()
        self.config = SYMAI_CONFIG
        self.api_key = self.config['SEARCH_ENGINE_API_KEY']
        self.model = self.config['SEARCH_ENGINE_MODEL']
        self.name = self.__class__.__name__

    def id(self) -> str:
        if self.config.get('SEARCH_ENGINE_API_KEY') and self.config.get('SEARCH_ENGINE_MODEL').startswith("sonar"):
            return 'search'
        return super().id()  # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'SEARCH_ENGINE_API_KEY' in kwargs:
            self.api_key = kwargs['SEARCH_ENGINE_API_KEY']
        if 'SEARCH_ENGINE_MODEL' in kwargs:
            self.model = kwargs['SEARCH_ENGINE_MODEL']

    def forward(self, argument):
        messages  = argument.prop.prepared_input
        kwargs    = argument.kwargs

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get('max_tokens', None),
            "temperature": kwargs.get('temperature', 0.2),
            "top_p": kwargs.get('top_p', 0.9),
            "search_domain_filter": kwargs.get('search_domain_filter', []),
            "return_images": kwargs.get('return_images', False),
            "return_related_questions": kwargs.get('return_related_questions', False),
            "search_recency_filter": kwargs.get('search_recency_filter', "month"),
            "top_k": kwargs.get('top_k', 0),
            "stream": kwargs.get('stream', False),
            "presence_penalty": kwargs.get('presence_penalty', 0),
            "frequency_penalty": kwargs.get('frequency_penalty', 1),
            "response_format": kwargs.get('response_format', None)
        }
        web_search_options = kwargs.get('web_search_options', None)
        if web_search_options is not None:
            payload["web_search_options"] = web_search_options

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        res = requests.post("https://api.perplexity.ai/chat/completions", json=payload, headers=headers)
        res = SearchResult(res.json())

        metadata = {"raw_output": res.raw}
        output   = [res]

        return output, metadata

    def prepare(self, argument):
        system_message = "You are a helpful AI assistant. Be precise and informative." if argument.kwargs.get('system_message') is None else argument.kwargs.get('system_message')

        res = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": f"{argument.prop.query}"
            }
        ]
        argument.prop.prepared_input = res
