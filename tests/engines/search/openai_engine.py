import os
import re

import pytest

from symai.backend.engines.search.engine_openai import GPTXSearchEngine
from symai.backend.settings import SYMAI_CONFIG
from symai.extended.interfaces.openai_search import openai_search
from symai.functional import EngineRepository


def _get_api_key():
    return (
        os.environ.get("OPENAI_API_KEY")
        or SYMAI_CONFIG.get("SEARCH_ENGINE_API_KEY")
        or os.environ.get("SEARCH_ENGINE_API_KEY")
    )


@pytest.mark.parametrize("model", ["gpt-4.1-mini", "gpt-5-mini"])
def test_openai_search_citations_and_formatting_live(model):
    api_key = _get_api_key()
    if not api_key:
        pytest.skip("OPENAI_API_KEY/SEARCH_ENGINE_API_KEY not set; live test skipped")

    # Register a fresh engine instance with the provided API key and target model
    engine = GPTXSearchEngine(api_key=api_key, model=model)
    EngineRepository.register("search", engine, allow_engine_override=True)

    # Keep the query stable but realistic to elicit citations
    query = "President of Romania 2025 inauguration timeline and partner (with citations)"
    search = openai_search()
    res = search(query, model=model, search_context_size="medium")

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

        # Slice should match the marker format; allow small whitespace variance before newline
        slice_text = res.value[c.start:c.end]
        assert slice_text.startswith(f"[{c.id}] (")
        assert slice_text.endswith(")\n")
        # Optional stronger check including title
        assert slice_text == f"[{c.id}] ({c.title})\n"

    # 3) Formatting: At least one marker pattern with newline is present
    assert re.search(r"\[\d+\] \([^)]+\)\n", res.value)
