import json
import logging
import requests
from copy import deepcopy
from dataclasses import dataclass

from ....symbol import Result
from ....utils import CustomUserWarning
from ...base import Engine
from ...settings import SYMAI_CONFIG
from ...mixin import OPENAI_CHAT_MODELS, OPENAI_REASONING_MODELS

logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


@dataclass
class Citation:
    id: str
    title: str
    url: str
    start: int
    end: int

    def __hash__(self):
        return hash((self.url, ))


class SearchResult(Result):
    def __init__(self, value, **kwargs) -> None:
        super().__init__(value, **kwargs)
        if value.get('error'):
            CustomUserWarning(value['error'], raise_with=ValueError)
        try:
            for output in value.get('output', []):
                if output.get('type') == 'message' and output.get('content'):
                    annotations = output['content'][0].get('annotations', [])
                    citations = []
                    for n, annotation in enumerate(annotations):
                        if annotation.get('type') == 'url_citation':
                            citation = Citation(
                                id=f'[{n + 1}]',
                                start=annotation.get('start_index'),
                                end=annotation.get('end_index'),
                                title=annotation.get('title', ''),
                                url=annotation.get('url', ''),
                            )
                            if citation not in citations:
                                citations.append(citation)
            self._value = output['content'][0]['text']
            delta = 0
            for citation in citations:
                self._value = self._value[:citation.start - delta] + citation.id + self._value[citation.end - delta:]
                delta += (citation.end - citation.start) - len(citation.id)
            self._citations = citations

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
        except Exception as e:
            return f"<pre>{str(self.raw)}</pre>"

    def get_citations(self) -> list[Citation]:
        return self._citations


class GPTXSearchEngine(Engine):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.config = deepcopy(SYMAI_CONFIG)
        if api_key is not None and model is not None:
            self.config['SEARCH_ENGINE_API_KEY'] = api_key
            self.config['SEARCH_ENGINE_MODEL']   = model
        self.api_key = self.config.get('SEARCH_ENGINE_API_KEY')
        self.model = self.config.get('SEARCH_ENGINE_MODEL', 'gpt-4.1') # Default to gpt-4.1 as per docs
        self.name = self.__class__.__name__

    def id(self) -> str:
        if self.config.get('SEARCH_ENGINE_API_KEY') and \
            self.config.get('SEARCH_ENGINE_MODEL') in OPENAI_CHAT_MODELS + OPENAI_REASONING_MODELS:
            return 'search'
        return super().id()  # default to unregistered

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'SEARCH_ENGINE_API_KEY' in kwargs:
            self.api_key = kwargs['SEARCH_ENGINE_API_KEY']
        if 'SEARCH_ENGINE_MODEL' in kwargs:
            self.model = kwargs['SEARCH_ENGINE_MODEL']

    def forward(self, argument):
        messages = argument.prop.prepared_input
        kwargs = argument.kwargs

        tool_definition = {"type": "web_search_preview"}
        user_location = kwargs.get('user_location')
        if user_location:
            tool_definition['user_location'] = user_location
        search_context_size = kwargs.get('search_context_size')
        if search_context_size:
            tool_definition['search_context_size'] = search_context_size

        self.model = kwargs.get('model', self.model) # Important for MetadataTracker to work correctly
        payload = {
            "model": self.model,
            "input": messages,
            "tools": [tool_definition],
            "tool_choice": {"type": "web_search_preview"} # force the use of web search tool
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v1" # Required for some beta features, might be useful
        }
        api_url = "https://api.openai.com/v1/responses"

        res = requests.post(api_url, json=payload, headers=headers)
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
