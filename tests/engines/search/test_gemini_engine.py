import os
import re
from urllib.parse import urlparse

import pytest

from symai.backend.engines.search.engine_gemini import (
    DEFAULT_SEARCH_MODEL,
    SUPPORTED_SEARCH_MODELS,
    GeminiSearchEngine,
)
from symai.backend.settings import SYMAI_CONFIG
from symai.components import MetadataTracker
from symai.extended.interfaces.gemini_search import gemini_search
from symai.functional import EngineRepository
from symai.utils import RuntimeInfo

GEMINI_API_KEY = SYMAI_CONFIG.get("SEARCH_ENGINE_API_KEY")

pytestmark = [
    pytest.mark.searchengine,
    pytest.mark.skipif(
        not GEMINI_API_KEY,
        reason="SEARCH_ENGINE_API_KEY not set; live test skipped",
    ),
]


def register_engine(model):
    engine = GeminiSearchEngine(api_key=GEMINI_API_KEY, model=model)
    EngineRepository.register("search", engine, allow_engine_override=True)
    return engine


@pytest.mark.parametrize("model", ["gemini-3.5-flash", "gemini-3.1-flash-lite"])
def test_gemini_search_citations_and_formatting_live(model):
    register_engine(model)

    # A query that reliably triggers grounding with multiple sources
    query = "Who won the UEFA Euro 2024 final and what was the score? (with citations)"
    res = gemini_search()(query, model=model)

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


@pytest.mark.parametrize("model", ["gemini-3.5-flash", "gemini-3.1-flash-lite"])
def test_gemini_search_resolve_urls_live(model):
    register_engine(model)

    query = "What is the capital of France and its longest river? (with citations)"
    res = gemini_search()(query, model=model, resolve_urls=True)

    citations = res.get_citations()
    assert len(citations) >= 1
    for c in citations:
        # Resolved citations must point at real sources, not the opaque Vertex redirect
        assert "vertexaisearch.cloud.google.com" not in c.url
        netloc = urlparse(c.url).netloc
        assert netloc, f"Resolved URL has no netloc: {c.url}"
        # The redirect title carries the real hostname, which should match the resolved host
        assert c.title, "Citation title is empty"


@pytest.mark.parametrize("model", ["gemini-3.5-flash", "gemini-3.1-pro-preview"])
def test_gemini_search_metadata_tracker_runtimeinfo_live(model):
    register_engine(model)

    query = "Who won the UEFA Euro 2024 final? (with citations)"
    with MetadataTracker() as tracker:
        res = gemini_search()(query, model=model)

    assert res.value is not None
    assert len(tracker.metadata) == 1

    (_entry_id, engine_name, tracked_model), _ = next(iter(tracker.metadata.items()))
    assert engine_name == "GeminiSearchEngine"
    assert tracked_model == model

    usage_per_engine = RuntimeInfo.from_tracker(tracker, 0.0)
    info = usage_per_engine.get((engine_name, model))
    assert info is not None
    assert info.prompt_tokens > 0
    assert info.completion_tokens > 0
    # Gemini's total_token_count includes thought/reasoning tokens, so total >=
    # prompt + completion (equality holds only when there are no thoughts)
    assert info.total_tokens >= info.prompt_tokens + info.completion_tokens
    assert info.total_calls == 1
    # reasoning_tokens come from Gemini's total_thought_tokens (may be 0 for non-thinking turns)
    assert info.reasoning_tokens >= 0
    assert info.cached_tokens >= 0

