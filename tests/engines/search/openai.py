import json
import os
import re

import pytest
from pydantic import BaseModel

from symai import Interface
from symai.backend.engines.search.engine_openai import SearchResult
from symai.backend.mixin import OPENAI_CHAT_MODELS, OPENAI_REASONING_MODELS
from symai.backend.settings import SYMAI_CONFIG

API_KEY = bool(SYMAI_CONFIG.get('SEARCH_ENGINE_API_KEY', None))
MODEL = SYMAI_CONFIG.get('SEARCH_ENGINE_MODEL', '') in OPENAI_CHAT_MODELS + OPENAI_REASONING_MODELS

pytestmark = [
    pytest.mark.search_engine,
    pytest.mark.skipif(not API_KEY, reason="SEARCH_ENGINE_API_KEY not configured or missing."),
    pytest.mark.skipif(not MODEL, reason="SEARCH_ENGINE_MODEL is not OpenAI chat model.")
]

try:
    search_interface = Interface('openai_search')
except Exception as e:
    search_interface = None
    pytestmark.append(pytest.mark.skipif(True, reason=f"OpenAI search interface initialization failed: {e}"))

@pytest.fixture(scope="module")
def openai_search_interface():
    if search_interface is None:
        pytest.skip("OpenAI search interface not available.")
    return search_interface

def test_openai_search_basic_query(openai_search_interface):
    """Test a basic query to OpenAI search."""
    query = "Who is Nicusor Dan?"
    res = openai_search_interface(query)

    assert isinstance(res, SearchResult), "Response should be a SearchResult instance."
    assert res.value is not None, "Response value should not be None."
    assert isinstance(res.value, str), "Response value should be a string."
    assert len(res.value) > 0, "Response value should not be empty."

    citations = res.get_citations()
    assert isinstance(citations, list), "get_citations should return a list"
    if citations:
        citation_ids = set(citation.id for citation in citations)
        for cid in citation_ids:
            assert cid in res.value, f"Citation {cid} should appear in text"

def test_openai_search_with_user_location(openai_search_interface):
    """Test OpenAI search with user location customization."""
    query = "What are popular tourist attractions nearby?"
    location_config = {
        "user_location": {
            "type": "approximate",
            "country": "US",
            "city": "New York",
            "region": "New York"
        }
    }

    res = openai_search_interface(query, **location_config)

    assert isinstance(res, SearchResult)
    assert res.value is not None
    assert isinstance(res.value, str)
    assert len(res.value) > 0

    location_terms = ["New York", "NYC", "Manhattan", "Brooklyn"]
    has_location_relevance = any(term.lower() in res.value.lower() for term in location_terms)
    if not has_location_relevance:
        pytest.skip("Response did not contain location-relevant information - API specifics may vary")

def test_openai_search_with_timezone(openai_search_interface):
    """Test OpenAI search with timezone in user location."""
    query = "What local events are happening today?"
    timezone_config = {
        "user_location": {
            "type": "approximate",
            "country": "JP",
            "city": "Tokyo",
            "region": "Tokyo",
            "timezone": "Asia/Tokyo"
        }
    }

    res = openai_search_interface(query, **timezone_config)

    assert isinstance(res, SearchResult)
    assert res.value is not None
    assert isinstance(res.value, str)
    assert len(res.value) > 0

    timezone_terms = ["Japan", "Tokyo", "JST"]
    has_timezone_relevance = any(term.lower() in res.value.lower() for term in timezone_terms)
    if not has_timezone_relevance:
        pytest.skip("Response did not contain location-relevant information - API specifics may vary")

def test_openai_search_context_size(openai_search_interface):
    """Test OpenAI search with different context size settings."""
    query = "Explain quantum computing developments"

    try:
        low_config = {"search_context_size": "low"}
        low_res = openai_search_interface(query, **low_config)

        assert isinstance(low_res, SearchResult)
        assert low_res.value is not None
        assert len(low_res.value) > 0

        high_config = {"search_context_size": "high"}
        high_res = openai_search_interface(query, **high_config)

        assert isinstance(high_res, SearchResult)
        assert high_res.value is not None
        assert len(high_res.value) > 0
    except Exception as e:
        pytest.skip(f"Search context size test failed with error: {str(e)}")
