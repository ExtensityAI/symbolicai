import json
import re
from urllib.parse import urlparse

import pytest

from symai.backend.engines.search.gemini import GeminiSearchEngine
from symai.backend.engines.search.gemini.models import (
    API_PINNED,
    GEMINI_API_BASE,
    GeminiInteractionResponse,
)
from symai.components import MetadataTracker
from symai.utils import RuntimeInfo
from tests.engines.search.interface import MOCK_QUERY, SearchEngineTestInterface

pytestmark = pytest.mark.searchengine

REDIRECT_HOST = "vertexaisearch.cloud.google.com"
GROUNDING_NUDGE = "You must always issue a Google Search to ground your answer before responding."


def _answer_text_and_annotations():
    text = (
        "Spain won the UEFA Euro 2024 final, beating England 2-1 in Berlin. "
        "Mikel Oyarzabal scored the decisive goal in the 86th minute."
    )
    annotations = [
        {
            "type": "url_citation",
            # NOTE: Gemini grounding returns opaque Vertex redirect URLs; the real
            # hostname arrives in the annotation title.
            "url": f"https://{REDIRECT_HOST}/grounding-api-redirect/AWhns0v1q2w3e4r5t6y7u8i9",
            "title": "uefa.com",
            "start_index": 0,
            "end_index": text.index(" in Berlin"),
        },
        {
            "type": "url_citation",
            "url": f"https://{REDIRECT_HOST}/grounding-api-redirect/BXiots1w2e3r4t5y6u7i8o9p",
            "title": "bbc.com",
            "start_index": text.index("Mikel"),
            "end_index": len(text),
        },
    ]
    return text, annotations


class TestGeminiSearchEngine(SearchEngineTestInterface):
    engine_cls = GeminiSearchEngine
    response_cls = GeminiInteractionResponse
    default_model = "gemini-3.5-flash"
    wire_url = f"{GEMINI_API_BASE}/interactions"
    auth_header_name = "x-goog-api-key"
    auth_header_prefix = ""
    api_pinned = API_PINNED
    api_pinned_module = "symai.backend.engines.search.gemini.models"
    keys_log_section = "google"
    keys_log_pattern = r'"(AIzaSy[^"]+)"'

    def mock_forward_kwargs(self):
        # NOTE: resolve_urls=True (the engine default) would HEAD the provider's redirect
        # URLs over the network; the mock suite stays hermetic by disabling resolution.
        # Resolution itself is covered by test_live_resolve_urls.
        return {"resolve_urls": False}

    def mock_response_json(self):
        text, annotations = _answer_text_and_annotations()
        return {
            "id": "interaction_0c3f5a1b9d2e4f80",
            "model": "gemini-3.5-flash",
            "steps": [
                {
                    "type": "google_search_call",
                    "content": [{"type": "query", "text": "UEFA Euro 2024 final winner score"}],
                },
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": text, "annotations": annotations}],
                },
            ],
            "usage": {
                "total_input_tokens": 412,
                "total_output_tokens": 96,
                "total_tokens": 508,
                "total_cached_tokens": 0,
                "total_thought_tokens": 0,
                "grounding_tool_count": [{"type": "google_search", "count": 1}],
            },
        }

    def response_dropping_required(self, payload):
        # GeminiInteractionStep.type is the one required response field.
        payload["steps"] = [{"content": []}]
        return payload

    def expected_request_body_subset(self):
        return {
            "model": self.default_model,
            "input": MOCK_QUERY,
            "tools": [{"type": "google_search"}],
        }

    def expected_prepared_input(self, query):
        return (
            "You are a helpful AI assistant. Be precise and informative.\n\n" + GROUNDING_NUDGE,
            query,
        )

    def test_mock_resolve_urls_disabled_keeps_redirect_urls(self):
        _api, output, _metadata = self.forward_through_mock()

        for citation in output[0].get_citations():
            assert REDIRECT_HOST in citation.url

    def test_build_request_keeps_resolve_urls_off_the_wire(self):
        engine = self.make_engine()
        argument = self.make_argument(kwargs={"resolve_urls": False})
        engine.prepare(argument)

        body = engine.build_request(argument).body()

        # resolve_urls steers result parsing (redirect resolution), not the wire payload
        assert "resolve_urls" not in json.dumps(body)

    @pytest.mark.engine_live
    @pytest.mark.parametrize("model", ["gemini-3.5-flash", "gemini-3.1-flash-lite"])
    def test_live_citations_and_formatting(self, engine_api_mode, model):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(api_key)
        argument = self.make_argument(
            query="Who won the UEFA Euro 2024 final and what was the score? (with citations)",
            kwargs={"model": model},
        )
        engine.prepare(argument)
        output, _metadata = engine.forward(argument)
        res = output[0]

        # 1) No leftover markdown link patterns or empty parentheses artifacts
        assert not re.search(r"\[[^\]]+\]\(https?://[^)]+\)", res.value)
        assert "(, , )" not in res.value
        assert "()" not in res.value

        # 2) Citations exist with integer ids and normalized URLs (no utm_ params)
        citations = res.get_citations()
        assert isinstance(citations, list) and len(citations) >= 1
        seen_ids = set()
        for c in citations:
            assert isinstance(c.id, int)
            assert c.id not in seen_ids
            seen_ids.add(c.id)
            assert "utm_" not in c.url

            # Slice should match the marker format "[id] (title)\n"
            slice_text = res.value[c.start : c.end]
            assert slice_text.startswith(f"[{c.id}] (")
            assert slice_text.endswith(")\n")
            assert slice_text == f"[{c.id}] ({c.title})\n"

        # 3) Formatting: at least one marker pattern with newline is present
        assert re.search(r"\[\d+\] \([^)]+\)\n", res.value)

    @pytest.mark.engine_live
    @pytest.mark.parametrize("model", ["gemini-3.5-flash", "gemini-3.1-flash-lite"])
    def test_live_resolve_urls(self, engine_api_mode, model):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(api_key)
        argument = self.make_argument(
            query="What is the capital of France and its longest river? (with citations)",
            kwargs={"model": model, "resolve_urls": True},
        )
        engine.prepare(argument)
        output, _metadata = engine.forward(argument)
        res = output[0]

        citations = res.get_citations()
        assert len(citations) >= 1
        for c in citations:
            # Resolved citations must point at real sources, not the opaque Vertex redirect
            assert REDIRECT_HOST not in c.url
            netloc = urlparse(c.url).netloc
            assert netloc, f"Resolved URL has no netloc: {c.url}"
            # The redirect title carries the real hostname, which should match the resolved host
            assert c.title, "Citation title is empty"

    @pytest.mark.engine_live
    @pytest.mark.parametrize("model", ["gemini-3.5-flash", "gemini-3.1-pro-preview"])
    def test_live_metadata_tracker_runtimeinfo(self, engine_api_mode, model):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(api_key)
        argument = self.make_argument(
            query="Who won the UEFA Euro 2024 final? (with citations)",
            kwargs={"model": model},
        )
        engine.prepare(argument)
        with MetadataTracker() as tracker:
            output, _metadata = engine.forward(argument)
        res = output[0]

        assert res.value is not None
        assert len(tracker.metadata) == 1

        (_entry_id, engine_name, tracked_model), _ = next(iter(tracker.metadata.items()))
        assert engine_name == "GeminiSearchEngine"
        assert tracked_model == model

        usage_per_engine = RuntimeInfo.from_tracker(tracker, 0.0)
        info = usage_per_engine.get((engine_name, model))
        assert info is not None
        assert info.prompt_tokens > 0
        assert info.cached_tokens >= 0
