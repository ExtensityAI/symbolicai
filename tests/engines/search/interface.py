"""Shared test interface for the migrated search engines (see ENGINE_REFACTOR_RECIPE.md).

One subclass per provider module supplies provider facts and wire fixtures; the
interface asserts the uniform search-engine contract:

- API_PINNED declared in the provider's models module
- prepare() produces the provider's expected prepared_input
- build_request wire shape (url, auth header convention, body subset)
- forward through MockAPI -> typed response metadata + non-empty marked text
- the cross-provider citation contract (folded in from the deleted
  test_citation_contract.py): 1-based strictly increasing ids, [id] markers in
  text, spans in bounds, normalized urls
- malformed responses fail typed parsing
- HTTP 401 -> EngineAuthenticationError
- scrape/extract url route through MockAPI (providers that support it)
- live smoke (--engine-api=live + provider key in api_keys.log)
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import httpx
import pytest
from pydantic import ValidationError

from symai.backend.engines.search.utils import Citation, normalize_url
from symai.backend.transport import EngineAuthenticationError
from tests.engines.interface import MockAPI

DUMMY_KEY = "sk-test-not-a-real-key"
MOCK_QUERY = "Who won the UEFA Euro 2024 final and what was the score?"
KEYS_LOG = Path("api_keys.log")


def load_key(provider: str, pattern: str) -> str | None:
    raw = KEYS_LOG.read_text()
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
            f"citation url is not normalized: {citation.url}"
        )
        assert f"[{citation.id}]" in text, f"marker [{citation.id}] missing from text"


def assert_body_subset(body, subset, path="body"):
    """Recursive subset check: dicts key-wise, lists element-wise (the subset may be
    shorter than the wire list), scalars by equality."""
    if isinstance(subset, dict):
        assert isinstance(body, dict), f"{path} must be a dict, got {type(body).__name__}"
        for key, value in subset.items():
            assert key in body, f"{path}.{key} missing from wire body"
            assert_body_subset(body[key], value, f"{path}.{key}")
    elif isinstance(subset, list):
        assert isinstance(body, list), f"{path} must be a list, got {type(body).__name__}"
        assert len(body) >= len(subset), f"{path} shorter than the expected subset"
        for index, value in enumerate(subset):
            assert_body_subset(body[index], value, f"{path}[{index}]")
    else:
        assert body == subset, f"{path} is {body!r}, expected {subset!r}"


class SearchEngineTestInterface:
    """Uniform contract checks for one search provider engine. Subclass per provider."""

    # --- provider facts (override) ---
    engine_cls: ClassVar = None
    response_cls: ClassVar = None
    default_model: ClassVar[str] = ""
    wire_url: ClassVar[str] = ""
    auth_header_name: ClassVar[str] = "Authorization"
    auth_header_prefix: ClassVar[str] = "Bearer "
    api_pinned: ClassVar[str] = ""
    api_pinned_module: ClassVar[str] = ""
    keys_log_section: ClassVar[str] = ""
    keys_log_pattern: ClassVar[str] = ""
    supports_scrape: ClassVar[bool] = False
    scrape_wire_url: ClassVar[str] = ""

    # --- provider hooks (override) ---
    def mock_response_json(self) -> dict:
        """A realistic captured provider response carrying >= 2 citations."""
        raise NotImplementedError

    def response_dropping_required(self, payload: dict) -> dict:
        """Malformed variant of mock_response_json that must fail typed parsing."""
        raise NotImplementedError

    def expected_request_body_subset(self) -> dict:
        """Key wire fields the engine must send for a plain query."""
        raise NotImplementedError

    def expected_prepared_input(self, query: str):
        """What prepare() must store on argument.prop.prepared_input (OpenAI/Perplexity shape)."""
        return [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Be precise and informative.",
            },
            {"role": "user", "content": query},
        ]

    def mock_forward_kwargs(self) -> dict:
        """Extra argument kwargs for the mock forward (Gemini disables url resolution)."""
        return {}

    def scrape_mock_response_json(self) -> dict:
        """A realistic captured scrape/extract response (supports_scrape providers only)."""
        raise NotImplementedError

    def scrape_url(self) -> str:
        return "https://www.uefa.com/euro2024/"

    def make_engine(self, api_key=DUMMY_KEY):
        return self.engine_cls(api_key=api_key, model=self.default_model)

    def make_live_engine(self, api_key: str):
        return self.make_engine(api_key)

    def assert_raw_output(self, metadata: dict):
        """metadata['raw_output'] carries the typed provider response."""
        assert isinstance(metadata["raw_output"], self.response_cls)

    # --- shared helpers ---
    def make_argument(self, query=MOCK_QUERY, url=None, kwargs=None):
        return SimpleNamespace(
            prop=SimpleNamespace(query=query, url=url, prepared_input=None),
            kwargs=kwargs or {},
        )

    def forward_through_mock(self, payload=None, url=None, **extra_kwargs):
        engine = self.make_engine()
        kwargs = {**self.mock_forward_kwargs(), **extra_kwargs}
        argument = self.make_argument(url=url, kwargs=kwargs)
        with MockAPI(
            engine,
            lambda request: httpx.Response(
                200,
                json=payload if payload is not None else self.mock_response_json(),
                request=request,
            ),
        ) as api:
            engine.prepare(argument)
            output, metadata = engine.forward(argument)
        return api, output, metadata

    def require_live(self, engine_api_mode) -> str:
        if engine_api_mode != "live":
            pytest.skip("use --engine-api=live to run live search API requests")
        if not KEYS_LOG.is_file():
            pytest.skip("api_keys.log not present; live test skipped")
        api_key = load_key(self.keys_log_section, self.keys_log_pattern)
        if not api_key:
            pytest.skip(f"api_keys.log has no {self.keys_log_section} key; live test skipped")
        return api_key

    def assert_auth_header(self, headers: dict):
        # httpx lowercases header names on the wire; EngineAPIRequest.headers keeps
        # the provider's casing — compare case-insensitively.
        lowered = {key.lower(): value for key, value in headers.items()}
        assert lowered.get(self.auth_header_name.lower()) == (
            f"{self.auth_header_prefix}{DUMMY_KEY}"
        )

    # --- contract checks ---
    def test_api_pinned_matches_models_module(self):
        assert self.api_pinned, "provider models.py must declare API_PINNED"
        module = importlib.import_module(self.api_pinned_module)
        assert self.api_pinned == module.API_PINNED

    def test_prepare_produces_expected_prepared_input(self):
        engine = self.make_engine()
        argument = self.make_argument()

        engine.prepare(argument)

        assert argument.prop.prepared_input == self.expected_prepared_input(MOCK_QUERY)

    def test_build_request_wire_shape(self):
        engine = self.make_engine()
        argument = self.make_argument()
        engine.prepare(argument)

        request = engine.build_request(argument)

        assert request.method == "POST"
        assert request.url == self.wire_url
        self.assert_auth_header(request.headers)
        assert_body_subset(request.body(), self.expected_request_body_subset())

    def test_forward_mock_returns_typed_response(self):
        api, output, metadata = self.forward_through_mock()

        assert api.last_request.method == "POST"
        assert str(api.last_request.url) == self.wire_url
        self.assert_auth_header(dict(api.last_request.headers))
        assert_body_subset(api.last_body, self.expected_request_body_subset())
        assert isinstance(output[0].value, str) and output[0].value
        self.assert_raw_output(metadata)

    def test_forward_mock_satisfies_citation_contract(self):
        _api, output, _metadata = self.forward_through_mock()

        assert_citation_contract(output[0])

    def test_malformed_response_fails_typed_parsing(self):
        engine = self.make_engine()
        payload = self.response_dropping_required(self.mock_response_json())

        with MockAPI(engine, lambda request: httpx.Response(200, json=payload, request=request)):
            argument = self.make_argument(kwargs=self.mock_forward_kwargs())
            engine.prepare(argument)
            with pytest.raises((ValidationError, ValueError)):
                engine.forward(argument)

    def test_http_401_raises_authentication_error(self):
        engine = self.make_engine()
        error_body = {"error": {"code": "invalid_api_key", "message": "bad key"}}

        with MockAPI(engine, lambda request: httpx.Response(401, json=error_body, request=request)):
            argument = self.make_argument(kwargs=self.mock_forward_kwargs())
            engine.prepare(argument)
            with pytest.raises(EngineAuthenticationError):
                engine.forward(argument)

    def test_scrape_route_mock(self):
        if not self.supports_scrape:
            pytest.skip(f"{self.engine_cls.__name__} has no scrape/extract route")
        api, output, _metadata = self.forward_through_mock(
            payload=self.scrape_mock_response_json(), url=self.scrape_url()
        )

        assert str(api.last_request.url) == self.scrape_wire_url
        assert isinstance(output[0].value, str) and output[0].value

    @pytest.mark.engine_live
    def test_live_smoke(self, engine_api_mode):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(api_key)
        argument = self.make_argument()
        engine.prepare(argument)
        output, _metadata = engine.forward(argument)

        assert_citation_contract(output[0])
