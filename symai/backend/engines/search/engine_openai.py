import hashlib
import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from openai import OpenAI

from ....symbol import Result
from ....utils import CustomUserWarning
from ...base import Engine
from ...mixin import OPENAI_CHAT_MODELS, OPENAI_REASONING_MODELS
from ...settings import SYMAI_CONFIG

logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


TRACKING_KEYS = {
    "utm_source" # so far I've only seen this one
}

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
            text, annotations = self._extract_text_and_annotations(value)
            if text is None:
                self._value = None
                self._citations = []
                return
            replaced_text, ordered = self._replace_links_with_citations(text, annotations, id_mode="sequential")
            self._value = replaced_text
            self._citations = [
                Citation(id=cid, title=title, url=url, start=0, end=0)
                for cid, title, url in ordered
            ]

        except Exception as e:
            self._value = None
            CustomUserWarning(f"Failed to parse response: {e}", raise_with=ValueError)

    def _extract_text(self, value) -> str | None:
        text = None
        for output in value.get('output', []):
            if output.get('type') == 'message' and output.get('content'):
                content0 = output['content'][0]
                if 'text' in content0 and content0['text']:
                    text = content0['text']
        return text

    def _extract_text_and_annotations(self, value):
        text = None
        annotations = []
        for output in value.get('output', []):
            if output.get('type') != 'message' or not output.get('content'):
                continue
            for content in output.get('content', []) or []:
                if 'text' in content and content['text']:
                    text = content['text']
                anns = content.get('annotations', []) or []
                for ann in anns:
                    if ann.get('type') == 'url_citation':
                        annotations.append(ann)
        return text, annotations

    def _normalize_url(self, u: str) -> str:
        parts = urlsplit(u)
        scheme = parts.scheme.lower()
        netloc = parts.netloc.lower()
        path = parts.path.rstrip('/') or '/'
        q = []
        for k, v in parse_qsl(parts.query, keep_blank_values=True):
            kl = k.lower()
            if kl in TRACKING_KEYS or kl.startswith('utm_'):
                continue
            q.append((k, v))
        query = urlencode(q, doseq=True)
        fragment = ''
        return urlunsplit((scheme, netloc, path, query, fragment))

    def _make_title_map(self, annotations):
        m = {}
        for a in annotations or []:
            url = a.get('url')
            if not url:
                continue
            nu = self._normalize_url(url)
            title = (a.get('title') or '').strip()
            if nu not in m and title:
                m[nu] = title
        return m

    def _hostname(self, u: str) -> str:
        return urlsplit(u).netloc

    def _short_hash_id(self, nu: str, length=6) -> str:
        return hashlib.sha1(nu.encode('utf-8')).hexdigest()[:length]

    def _replace_links_with_citations(self, text: str, annotations, id_mode: str = 'sequential'):
        title_map = self._make_title_map(annotations)
        id_map = {}
        ordered = []  # list of ("[n]", title, normalized_url)
        next_id = 1

        pattern = re.compile(r"\[([^\]]*?)\]\((https?://[^\s)]+)\)")

        def _get_id(nu: str) -> str:
            nonlocal next_id
            if id_mode == 'hash':
                return self._short_hash_id(nu)
            if nu not in id_map:
                id_map[nu] = str(next_id)
                t = title_map.get(nu) or self._hostname(nu)
                ordered.append((f"[{id_map[nu]}]", t, nu))
                next_id += 1
            return id_map[nu]

        def _repl(m):
            link_text, url = m.group(1), m.group(2)
            nu = self._normalize_url(url)
            cid = _get_id(nu)
            title = title_map.get(nu)
            if not title:
                lt = (link_text or '').strip()
                title = lt if (' ' in lt) else self._hostname(nu)
            return f"[{cid}] ({title})"

        replaced = pattern.sub(_repl, text)
        return replaced, ordered

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
        try:
            self.client = OpenAI(api_key=self.api_key)
        except Exception as e:
            CustomUserWarning(f"Failed to initialize OpenAI client: {e}", raise_with=ValueError)

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
            "tool_choice": {"type": "web_search_preview"} if self.model not in OPENAI_REASONING_MODELS else "auto" # force the use of web search tool for non-reasoning models
        }

        try:
            res = self.client.responses.create(**payload)
            res = SearchResult(res.dict())
        except Exception as e:
            CustomUserWarning(f"Failed to make request: {e}", raise_with=ValueError)

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
