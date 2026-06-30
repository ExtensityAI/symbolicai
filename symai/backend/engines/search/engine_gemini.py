import logging
from copy import deepcopy

import httpx
from google import genai
from google.genai import types

from symai.backend.base import Engine
from symai.backend.engines.search.utils import CitationResultMixin, insert_citation_markers
from symai.backend.mixin.google import GoogleMixin
from symai.backend.settings import SYMAI_CONFIG
from symai.symbol import Result

logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("google_genai").propagate = False
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


SUPPORTED_SEARCH_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
]
DEFAULT_SEARCH_MODEL = "gemini-3.5-flash"

# Gemini grounding returns Vertex AI redirect URLs (vertexaisearch.cloud.google.com).
# Real hostnames arrive in the annotation `title`; the redirect is followed on demand.
_REDIRECT_HOST = "vertexaisearch.cloud.google.com"


class GeminiSearchResult(CitationResultMixin, Result):
    def __init__(self, value, *, resolve_urls: bool = True, resolve_timeout: float = 10.0) -> None:
        super().__init__(value)
        self._resolve_urls = resolve_urls
        self._resolve_timeout = resolve_timeout
        self._resolved_cache = {}
        self._citations = []
        self._value = None
        try:
            text, annotations = self._extract_text_and_annotations()
            if text is None:
                self._value = None
                self._citations = []
                return
            if self._resolve_urls:
                self._resolve_annotation_urls(annotations)
            self._value, self._citations = insert_citation_markers(text, annotations)
        except Exception as e:
            self._value = None
            self._citations = []
            msg = f"Failed to parse response: {e}"
            raise ValueError(msg) from e

    def _extract_text_and_annotations(self):
        segments = []
        global_annotations = []
        pos = 0
        # NOTE: Gemini responses carry the answer in `model_output` steps; each content
        # block has its own `text` and inline `annotations` whose indices are relative to
        # that block's text, so we accumulate a running offset across all blocks.
        for step in self.raw.get("steps", []) or []:
            if step.get("type") != "model_output":
                continue
            for content in step.get("content", []) or []:
                seg_text = content.get("text") or ""
                if not isinstance(seg_text, str):
                    continue
                for ann in content.get("annotations", []) or []:
                    if ann.get("type") == "url_citation" and ann.get("url"):
                        start = int(ann.get("start_index", 0) or 0)
                        end = int(ann.get("end_index", 0) or 0)
                        global_annotations.append(
                            {
                                "type": "url_citation",
                                "url": ann.get("url"),
                                "title": (ann.get("title") or "").strip(),
                                "start_index": pos + start,
                                "end_index": pos + end,
                            }
                        )
                segments.append(seg_text)
                pos += len(seg_text)

        built_text = "".join(segments) if segments else None
        # Fall back to the convenience field when no model_output text could be assembled
        if not built_text and isinstance(self.raw.get("output_text"), str):
            return self.raw.get("output_text"), []
        return built_text, global_annotations

    def _resolve_annotation_urls(self, annotations: list[dict]) -> None:
        cache = {}
        for ann in annotations:
            url = ann.get("url")
            if not url:
                continue
            if url not in cache:
                cache[url] = self._resolve_redirect(url)
            ann["url"] = cache[url]

    def _resolve_redirect(self, url: str) -> str:
        if not url or _REDIRECT_HOST not in url:
            return url
        if url in self._resolved_cache:
            return self._resolved_cache[url]
        resolved = url
        try:
            # NOTE: The grounding endpoint 30x-redirects to the real source URL; a HEAD is
            # enough because we only need the final Location, not the body.
            with httpx.Client(follow_redirects=True, timeout=self._resolve_timeout) as client:
                response = client.head(url, headers={"User-Agent": "symai/1.0"})
                if response.url and str(response.url) != url:
                    resolved = str(response.url)
        except Exception:
            # Best effort: keep the opaque redirect URL if resolution fails
            resolved = url
        self._resolved_cache[url] = resolved
        return resolved


class GeminiSearchEngine(Engine, GoogleMixin):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        client_timeout: float | None = None,
        client_max_retries: int | None = None,
    ):
        super().__init__(client_timeout=client_timeout, client_max_retries=client_max_retries)
        self.config = deepcopy(SYMAI_CONFIG)
        # In case we use EngineRepository.register to inject the api_key and model => dynamically change the engine at runtime
        if api_key is not None and model is not None:
            self.config["SEARCH_ENGINE_API_KEY"] = api_key
            self.config["SEARCH_ENGINE_MODEL"] = model
        self.api_key = self.config.get("SEARCH_ENGINE_API_KEY")
        self.model = self.config.get("SEARCH_ENGINE_MODEL", DEFAULT_SEARCH_MODEL)
        self.name = self.__class__.__name__

        if api_key is None and model is None and self.id() != "search":
            return  # do not initialize if not the active search engine

        try:
            self.client = self._build_client()
        except Exception as e:
            msg = f"Failed to initialize Gemini client: {e}"
            raise ValueError(msg) from e

    def id(self) -> str:
        model = self.config.get("SEARCH_ENGINE_MODEL")
        if model and model in SUPPORTED_SEARCH_MODELS:
            return "search"
        return super().id()  # default to unregistered

    def _build_client(self) -> genai.Client:
        http_opts_kwargs = {}
        if self.client_timeout is not None:
            # NOTE: google-genai takes timeout in milliseconds (int), not seconds
            http_opts_kwargs["timeout"] = int(self.client_timeout * 1000)
        if self.client_max_retries is not None:
            # NOTE: attempts includes the original request; our contract counts retries
            # after the original, so add 1
            http_opts_kwargs["retry_options"] = types.HttpRetryOptions(
                attempts=self.client_max_retries + 1
            )
        return genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(**http_opts_kwargs),
        )

    def command(self, *args, **kwargs):
        super().command(*args, **kwargs)
        if "SEARCH_ENGINE_API_KEY" in kwargs:
            self.api_key = kwargs["SEARCH_ENGINE_API_KEY"]
            try:
                self.client = self._build_client()
            except Exception as e:
                msg = f"Failed to re-initialize Gemini client: {e}"
                raise ValueError(msg) from e
        if "SEARCH_ENGINE_MODEL" in kwargs:
            self.model = kwargs["SEARCH_ENGINE_MODEL"]

    def forward(self, argument):
        system_instruction, user_input = argument.prop.prepared_input
        kwargs = argument.kwargs

        self.model = kwargs.get(
            "model", self.model
        )  # Important for MetadataTracker to work correctly

        create_kwargs = {
            "model": self.model,
            "input": user_input,
            "tools": [{"type": "google_search"}],
        }
        if system_instruction:
            create_kwargs["system_instruction"] = system_instruction

        interaction = None
        try:
            interaction = self.client.interactions.create(**create_kwargs)
        except Exception as e:
            msg = f"Failed to make request: {e}"
            raise ValueError(msg) from e

        resolve_urls = kwargs.get("resolve_urls", True)
        res = GeminiSearchResult(interaction.model_dump(), resolve_urls=resolve_urls)
        metadata = {"raw_output": interaction}
        output = [res]

        return output, metadata

    def prepare(self, argument):
        system_message = (
            "You are a helpful AI assistant. Be precise and informative."
            if argument.kwargs.get("system_message") is None
            else argument.kwargs.get("system_message")
        )

        # NOTE: Gemini's interactions API has no `tool_choice`/force-grounding flag (unlike
        # OpenAI's web_search), so the only lever left is a system-instruction nudge. This
        # engine is a search engine, so search takes priority over the model's own knowledge;
        # we always prepend the nudge to force grounded answers.
        system_message = (
            f"{system_message}\n\n"
            "You must always issue a Google Search to ground your answer before responding."
        )

        argument.prop.prepared_input = (system_message, f"{argument.prop.query}")
