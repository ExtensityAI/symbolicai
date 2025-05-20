import json
import os
import re

import pytest
from pydantic import BaseModel

from symai import Interface
from symai.backend.engines.search.engine_perplexity import SearchResult
from symai.backend.settings import SYMAI_CONFIG

API_KEY = bool(SYMAI_CONFIG.get('SEARCH_ENGINE_API_KEY', None))
MODEL = SYMAI_CONFIG.get('SEARCH_ENGINE_MODEL', '').startswith('sonar')

pytestmark = [
    pytest.mark.searchengine,
    pytest.mark.skipif(not API_KEY, reason="SEARCH_ENGINE_API_KEY not configured or missing."),
    pytest.mark.skipif(not MODEL, reason="SEARCH_ENGINE_MODEL is not 'sonar' or sonar-compatible.")
]

try:
    search_interface = Interface('perplexity')
except Exception as e:
    search_interface = None
    pytestmark.append(pytest.mark.skipif(True, reason=f"Perplexity interface initialization failed: {e}"))


@pytest.fixture(scope="module")
def perplexity_interface():
    if search_interface is None:
        pytest.skip("Perplexity interface not available.")
    return search_interface


def test_perplexity_basic_query(perplexity_interface):
    """Test a basic query to Perplexity."""
    query = "What is the capital of France?"
    res = perplexity_interface(query)
    assert isinstance(res, SearchResult), "Response should be a SearchResult instance."
    assert res._value is not None, "Response value should not be None."
    assert isinstance(res._value, str), "Response value should be a string."
    assert len(res._value) > 0, "Response value should not be empty."
    assert "Paris" in res._value, "Response for capital of France should contain 'Paris'."


def test_perplexity_custom_system_message(perplexity_interface):
    """Test Perplexity with a custom system message."""
    query = "Explain black holes."
    system_message = "Explain like I'm five years old."
    res = perplexity_interface(query, system_message=system_message)
    assert isinstance(res, SearchResult)
    assert res._value is not None and len(res._value) > 0


def test_perplexity_max_tokens(perplexity_interface):
    """Test Perplexity with max_tokens parameter."""
    query = "Tell me a very short story about a robot."
    max_tokens_val = 30
    res = perplexity_interface(query, max_tokens=max_tokens_val)
    assert isinstance(res, SearchResult)
    assert res._value is not None and len(res._value) > 0
    # A rough check for length; tokenization varies.
    # This is a heuristic, not a strict guarantee.
    assert len(res._value.split()) < max_tokens_val * 3, f"Response seems too long for max_tokens={max_tokens_val}"


def test_perplexity_temperature(perplexity_interface):
    """Test Perplexity with different temperature settings."""
    query = "What are some creative names for a pet cat?"

    res = perplexity_interface(query, temperature=0.2)
    assert isinstance(res, SearchResult)
    assert res._value is not None and len(res._value) > 0


def test_perplexity_top_p(perplexity_interface):
    """Test Perplexity with top_p parameter."""
    query = "Suggest a good plot for a mystery novel."
    res = perplexity_interface(query, top_p=0.6)
    assert isinstance(res, SearchResult)
    assert res._value is not None and len(res._value) > 0


def test_perplexity_search_domain_filter(perplexity_interface):
    """Test Perplexity with search_domain_filter."""
    query = "What is the main idea of Albert Einstein's theory of relativity, from wikipedia.org?"
    # Engine default is ["perplexity.ai"], override it.
    res = perplexity_interface(query, search_domain_filter=["wikipedia.org"])
    assert isinstance(res, SearchResult)
    assert res._value is not None and len(res._value) > 0


def test_perplexity_return_images_parameter_handling(perplexity_interface):
    """Test Perplexity's handling of the return_images parameter."""
    query = "Show me pictures of the aurora borealis."

    res_with_images = perplexity_interface(query, return_images=True)
    assert isinstance(res_with_images, SearchResult)
    assert res_with_images._value is not None # Textual part
    assert res_with_images.raw is not None # Full API response
    # Actual image data or links would be in res_with_images.raw, format dependent on Perplexity API.

    res_without_images = perplexity_interface(query, return_images=False)
    assert isinstance(res_without_images, SearchResult)
    assert res_without_images._value is not None


def test_perplexity_return_related_questions_parameter_handling(perplexity_interface):
    """Test Perplexity's handling of the return_related_questions parameter."""
    query = "What are the benefits of meditation?"

    res_with_related = perplexity_interface(query, return_related_questions=True)
    assert isinstance(res_with_related, SearchResult)
    assert res_with_related._value is not None # Textual part
    assert res_with_related.raw is not None # Full API response
    # Related questions, if returned, would be in res_with_related.raw.

    res_without_related = perplexity_interface(query, return_related_questions=False)
    assert isinstance(res_without_related, SearchResult)
    assert res_without_related._value is not None


def test_perplexity_search_recency_filter(perplexity_interface):
    """Test Perplexity with search_recency_filter."""
    query = "What are the latest developments in space exploration this week?"
    res = perplexity_interface(query, search_recency_filter="week") # Default is "month"
    assert isinstance(res, SearchResult)
    assert res._value is not None and len(res._value) > 0


def test_perplexity_top_k(perplexity_interface):
    """Test Perplexity with top_k parameter."""
    query = "Describe a fantasy creature."
    res = perplexity_interface(query, top_k=5) # API default is 0 (disabled)
    assert isinstance(res, SearchResult)
    assert res._value is not None and len(res._value) > 0


def test_perplexity_presence_penalty(perplexity_interface):
    """Test Perplexity with presence_penalty parameter."""
    query = "Summarize the plot of 'Hamlet' ensuring to cover diverse aspects without dwelling on one."
    res = perplexity_interface(query, presence_penalty=0.8) # API default is 0
    assert isinstance(res, SearchResult)
    assert res._value is not None and len(res._value) > 0


def test_perplexity_frequency_penalty(perplexity_interface):
    """Test Perplexity with frequency_penalty parameter."""
    query = "Explain photosynthesis using unique phrasing for each part of the process."
    res = perplexity_interface(query, frequency_penalty=1.2) # API default is 1
    assert isinstance(res, SearchResult)
    assert res._value is not None and len(res._value) > 0


def test_perplexity_response_format_json(perplexity_interface):
    """Test Perplexity with response_format for JSON output using json_schema."""

    class BasketballPlayerInfo(BaseModel):
        first_name: str
        last_name: str
        year_of_birth: int
        num_seasons_in_nba: int

    query = "Tell me about Michael Jordan. Please output a JSON object containing the following fields: first_name, last_name, year_of_birth, num_seasons_in_nba."
    system_message = "Be precise and concise."

    res = perplexity_interface(
        query,
        system_message=system_message,
        response_format={
            "type": "json_schema",
            "json_schema": {"schema": BasketballPlayerInfo.model_json_schema()}
        }
    )

    assert isinstance(res, SearchResult)
    assert res._value is not None and len(res._value) > 0

    raw_content = res.raw['choices'][0]['message']['content']
    try:
        json_output = json.loads(raw_content)
        assert isinstance(json_output, dict), f"Expected JSON object, got {type(json_output)}"
        # Validate the schema structure
        assert "first_name" in json_output, "Missing 'first_name' field in response"
        assert "last_name" in json_output, "Missing 'last_name' field in response"
        assert "year_of_birth" in json_output, "Missing 'year_of_birth' field in response"
        assert "num_seasons_in_nba" in json_output, "Missing 'num_seasons_in_nba' field in response"
        # Validate data types
        assert isinstance(json_output["first_name"], str), "first_name should be a string"
        assert isinstance(json_output["last_name"], str), "last_name should be a string"
        assert isinstance(json_output["year_of_birth"], int), "year_of_birth should be an integer"
        assert isinstance(json_output["num_seasons_in_nba"], int), "num_seasons_in_nba should be an integer"
        # Validate expected values
        assert json_output["first_name"] == "Michael", "Incorrect first_name value"
        assert json_output["last_name"] == "Jordan", "Incorrect last_name value"
    except json.JSONDecodeError:
        pytest.fail(f"Response content was not valid JSON as expected. Raw content: {raw_content}")
    except Exception as e:
        pytest.fail(f"Error processing JSON response: {e}. Raw content: {raw_content}")


def test_perplexity_response_format_regex(perplexity_interface):
    """Test Perplexity with response_format using regex pattern matching."""

    query = "What is the IPv4 address of OpenDNS DNS server?"
    system_message = "Be precise and concise."

    ipv4_regex = r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"

    response_format = {
        "type": "regex",
        "regex": {"regex": ipv4_regex}
    }

    res = perplexity_interface(query, system_message=system_message, response_format=response_format)

    assert isinstance(res, SearchResult)
    assert res._value is not None, "Response value should not be None"
    assert len(res._value) > 0, "Response value should not be empty"

    ipv4_pattern = re.compile(ipv4_regex)
    assert ipv4_pattern.fullmatch(res._value.strip()), f"Response '{res._value}' does not match IPv4 pattern"


def test_perplexity_web_search_options(perplexity_interface):
    """Test Perplexity with web_search_options."""
    query = "What are some recent breakthroughs in AI research?"
    web_search_opts = {
        "search_context_size": "high", # Options: low, medium, high
        "user_location": {"country": "CA"} # Example: Filter by country
    }
    res = perplexity_interface(query, web_search_options=web_search_opts)
    assert isinstance(res, SearchResult)
    assert res._value is not None and len(res._value) > 0
