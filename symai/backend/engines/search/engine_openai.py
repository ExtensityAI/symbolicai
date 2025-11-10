import hashlib
import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from openai import OpenAI

from ....symbol import Result
from ....utils import UserMessage
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
    id: int
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
            UserMessage(value['error'], raise_with=ValueError)
        try:
            text, annotations = self._extract_text_and_annotations(value)
            if text is None:
                self._value = None
                self._citations = []
                return
            replaced_text, ordered, starts_ends = self._insert_citation_markers(text, annotations)
            self._value = replaced_text
            self._citations = [
                Citation(id=cid, title=title, url=url, start=starts_ends[cid][0], end=starts_ends[cid][1])
                for cid, title, url in ordered
            ]

        except Exception as e:
            self._value = None
            UserMessage(f"Failed to parse response: {e}", raise_with=ValueError)

    def _extract_text(self, value) -> str | None:
        if isinstance(value.get('output_text'), str) and value.get('output_text'):
            return value.get('output_text')
        text = None
        for output in value.get('output', []):
            if output.get('type') == 'message' and output.get('content'):
                content0 = output['content'][0]
                if content0.get('text'):
                    text = content0['text']
        return text

    def _extract_text_and_annotations(self, value):
        segments = []
        global_annotations = []
        pos = 0
        for output in value.get('output', []) or []:
            if output.get('type') != 'message' or not output.get('content'):
                continue
            for content in output.get('content', []) or []:
                seg_text = content.get('text') or ''
                if not isinstance(seg_text, str):
                    continue
                for ann in (content.get('annotations') or []):
                    if ann.get('type') == 'url_citation' and ann.get('url'):
                        start = ann.get('start_index', 0)
                        end = ann.get('end_index', 0)
                        global_annotations.append({
                            'type': 'url_citation',
                            'url': ann.get('url'),
                            'title': (ann.get('title') or '').strip(),
                            'start_index': pos + int(start),
                            'end_index': pos + int(end),
                        })
                segments.append(seg_text)
                pos += len(seg_text)

        built_text = ''.join(segments) if segments else None
        # Prefer top-level output_text if present AND segments are empty (no way to compute indices)
        if not built_text and isinstance(value.get('output_text'), str):
            return value.get('output_text'), []
        return built_text, global_annotations

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

    def _insert_citation_markers(self, text: str, annotations):
        title_map = self._make_title_map(annotations)
        id_map: dict[str, int] = {}
        first_span: dict[int, tuple[int, int]] = {}
        ordered: list[tuple[int, str, str]] = []  # (id, title, normalized_url)
        next_id = 1

        url_anns = [a for a in annotations or [] if a.get('type') == 'url_citation' and a.get('url')]
        url_anns.sort(key=lambda a: int(a.get('start_index', 0)))

        pieces: list[str] = []
        cursor = 0
        out_len = 0  # length of output built so far (after cleaning and prior markers)

        def _get_id(nu: str) -> int:
            nonlocal next_id
            if nu not in id_map:
                cid = next_id
                id_map[nu] = cid
                title = title_map.get(nu) or self._hostname(nu)
                ordered.append((cid, title, nu))
                next_id += 1
            return id_map[nu]

        for ann in url_anns:
            start = int(ann.get('start_index', 0))
            end = int(ann.get('end_index', 0))
            if end <= cursor:
                continue  # skip overlapping or backwards spans
            url = ann.get('url')
            nu = self._normalize_url(url)
            cid = _get_id(nu)
            title = title_map.get(nu) or self._hostname(nu)

            prefix = text[cursor:start]
            prefix_clean = self._strip_markdown_links(prefix)
            pieces.append(prefix_clean)
            out_len += len(prefix_clean)

            span_text = text[start:end]
            span_clean = self._strip_markdown_links(span_text)
            span_end_out = out_len + len(span_clean)
            pieces.append(span_clean)
            out_len = span_end_out

            marker = f"[{cid}] ({title})\n"
            marker_start_out = out_len
            marker_end_out = out_len + len(marker)
            if cid not in first_span:
                first_span[cid] = (marker_start_out, marker_end_out)
            pieces.append(marker)
            out_len = marker_end_out
            cursor = end

        tail_clean = self._strip_markdown_links(text[cursor:])
        pieces.append(tail_clean)
        replaced = ''.join(pieces)

        starts_ends = {cid: first_span.get(cid, (0, 0)) for cid, _, _ in ordered}
        return replaced, ordered, starts_ends

    def _strip_markdown_links(self, text: str) -> str:
        # Remove ([text](http...)) including surrounding parentheses
        pattern_paren = re.compile(r"\(\s*\[[^\]]+\]\(https?://[^)]+\)\s*\)")
        text = pattern_paren.sub('', text)
        # Remove bare [text](http...)
        pattern_bare = re.compile(r"\[[^\]]+\]\(https?://[^)]+\)")
        text = pattern_bare.sub('', text)
        # Remove parentheses that became empty or contain only commas/whitespace like (, , )
        pattern_empty_paren = re.compile(r"\(\s*\)")
        text = pattern_empty_paren.sub('', text)
        pattern_commas_only = re.compile(r"\(\s*(,\s*)+\)")
        text = pattern_commas_only.sub('', text)
        # Collapse potential double spaces resulting from removals
        return re.sub(r"\s{2,}", " ", text).strip()

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


class GPTXSearchEngine(Engine):
    MAX_ALLOWED_DOMAINS = 20

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
            UserMessage(f"Failed to initialize OpenAI client: {e}", raise_with=ValueError)

    def id(self) -> str:
        if self.config.get('SEARCH_ENGINE_API_KEY') and \
            self.config.get('SEARCH_ENGINE_MODEL') in OPENAI_CHAT_MODELS + OPENAI_REASONING_MODELS:
            return 'search'
        return super().id()  # default to unregistered

    def _extract_netloc(self, raw_domain: str | None) -> str | None:
        if not isinstance(raw_domain, str):
            return None
        candidate = raw_domain.strip()
        if not candidate:
            return None
        parsed = urlsplit(candidate if '://' in candidate else f"//{candidate}")
        netloc = parsed.netloc or parsed.path
        if not netloc:
            return None
        if '@' in netloc:
            netloc = netloc.split('@', 1)[1]
        if ':' in netloc:
            netloc = netloc.split(':', 1)[0]
        netloc = netloc.strip('.').strip()
        if not netloc:
            return None
        return netloc.lower()

    def _normalize_allowed_domains(self, domains: list[str] | None) -> list[str]:
        if not domains or not isinstance(domains, list):
            return []

        normalized: list[str] = []
        seen: set[str] = set()
        for domain in domains:
            netloc = self._extract_netloc(domain)
            if not netloc or netloc in seen:
                continue
            # Validate that netloc is actually a valid domain
            if not self._is_domain(netloc):
                continue
            normalized.append(netloc)
            seen.add(netloc)
            if len(normalized) >= self.MAX_ALLOWED_DOMAINS:
                break
        return normalized

    def _is_domain(self, s: str) -> bool:
        _label_re = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
        if not s:
            return False
        host = s.strip().rstrip(".")
        # If the input might be a URL, extract the hostname via urllib:
        if "://" in host or "/" in host or "@" in host:
            host = urlsplit(host if "://" in host else f"//{host}").hostname or ""
        if not host:
            return False
        try:
            host_ascii = host.encode("idna").decode("ascii")
        except Exception:
            return False
        if len(host_ascii) > 253:
            return False
        labels = host_ascii.split(".")
        if len(labels) < 2:  # require a dot (reject "google")
            return False
        return all(_label_re.fullmatch(lbl or "") for lbl in labels)

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if 'SEARCH_ENGINE_API_KEY' in kwargs:
            self.api_key = kwargs['SEARCH_ENGINE_API_KEY']
        if 'SEARCH_ENGINE_MODEL' in kwargs:
            self.model = kwargs['SEARCH_ENGINE_MODEL']

    def forward(self, argument):
        messages = argument.prop.prepared_input
        kwargs = argument.kwargs

        tool_definition = {"type": "web_search"}
        user_location = kwargs.get('user_location')
        if user_location:
            tool_definition['user_location'] = user_location

        allowed_domains = self._normalize_allowed_domains(kwargs.get('allowed_domains'))
        if allowed_domains:
            tool_definition['filters'] = {
                'allowed_domains': allowed_domains
            }

        self.model = kwargs.get('model', self.model) # Important for MetadataTracker to work correctly

        payload = {
            "model": self.model,
            "input": messages,
            "tools": [tool_definition],
            "tool_choice": {"type": "web_search"} if self.model not in OPENAI_REASONING_MODELS else "auto" # force the use of web search tool for non-reasoning models
        }

        if self.model in OPENAI_REASONING_MODELS:
            reasoning = kwargs.get('reasoning', { "effort": "low", "summary": "auto" })
            payload['reasoning'] = reasoning

        try:
            res = self.client.responses.create(**payload)
            res = SearchResult(res.dict())
        except Exception as e:
            UserMessage(f"Failed to make request: {e}", raise_with=ValueError)

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
