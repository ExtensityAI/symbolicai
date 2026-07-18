"""Cross-provider Citation contract.

Every search engine returns a Result whose ``.value`` is the marker-annotated
answer text and whose ``.get_citations()`` yields the shared ``Citation``
dataclass. This suite pins that surface down per provider: same object, same
invariants, regardless of how the provider formats its wire response.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from symai.backend.engines.search.gemini import GeminiSearchEngine
from symai.backend.engines.search.openai import OpenAISearchEngine
from symai.backend.engines.search.parallel import ParallelEngine
from symai.backend.engines.search.perplexity import PerplexityEngine
from symai.backend.engines.search.utils import Citation, normalize_url

pytestmark = [
    pytest.mark.searchengine,
    pytest.mark.skipif(
        not Path("api_keys.log").is_file(),
        reason="api_keys.log not present; live test skipped",
    ),
]

QUERY = "Who won the UEFA Euro 2024 final and what was the score?"


def load_key(provider: str, pattern: str) -> str | None:
    raw = Path("api_keys.log").read_text()
    section = re.search(rf"^{provider}:\n((?:\s+.*\n)+)", raw, re.MULTILINE)
    if not section:
        return None
    match = re.search(pattern, section.group(1))
    return match.group(1) if match else None


def assert_citation_contract(result) -> None:
    text = result.value
    assert isinstance(text, str) and text.strip(), "result value must be non-empty text"

    citations = result.get_citations()
    assert isinstance(citations, list) and citations, "expected at least one citation"
    assert all(isinstance(c, Citation) for c in citations)

    ids = [c.id for c in citations]
    # NOTE: ids correspond to the [id] markers in the text; providers that assign
    # their own marker numbers (Perplexity) may skip ids, so the invariant is
    # 1-based strictly increasing, not contiguity.
    assert ids[0] >= 1, "ids must be 1-based"
    assert ids == sorted(set(ids)), "ids must be unique and increasing"

    for citation in citations:
        assert 0 <= citation.start <= citation.end <= len(text), (
            f"citation span out of bounds: {citation}"
        )
        assert citation.url == normalize_url(citation.url), (
            f"citation url not normalized: {citation.url}"
        )
        assert f"[{citation.id}]" in text, f"marker [{citation.id}] missing from text"


def forward(engine, query: str, **kwargs):
    argument = SimpleNamespace(
        prop=SimpleNamespace(query=query, url=None, prepared_input=None),
        kwargs=kwargs,
    )
    engine.prepare(argument)
    output, metadata = engine.forward(argument)
    return output[0], metadata


def test_openai_citation_contract():
    api_key = load_key("openai", r'"(sk-proj-[^"]+)"')
    if not api_key:
        pytest.skip("openai key not found in api_keys.log")
    engine = OpenAISearchEngine(api_key=api_key, model="gpt-4.1-mini")
    result, _ = forward(engine, QUERY)
    assert_citation_contract(result)


def test_gemini_citation_contract():
    api_key = load_key("google", r'"(AIzaSy[^"]+)"')
    if not api_key:
        pytest.skip("google key not found in api_keys.log")
    engine = GeminiSearchEngine(api_key=api_key, model="gemini-3.1-flash-lite")
    result, _ = forward(engine, QUERY)
    assert_citation_contract(result)


def test_perplexity_citation_contract():
    api_key = load_key("perplexity", r'"(pplx-[^"]+)"\s*# office')
    if not api_key:
        pytest.skip("perplexity key not found in api_keys.log")
    engine = PerplexityEngine()
    engine.api_key = api_key
    engine.model = "sonar"
    result, _ = forward(engine, QUERY)
    assert_citation_contract(result)


def test_parallel_citation_contract():
    api_key = load_key("parallel", r'"(xxxx[^"]+)"')
    if not api_key:
        pytest.skip("parallel key not found in api_keys.log")
    engine = ParallelEngine(api_key=api_key)
    result, _ = forward(engine, QUERY, max_results=3)
    assert_citation_contract(result)
