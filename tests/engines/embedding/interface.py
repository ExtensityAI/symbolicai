"""Shared test interface for the migrated embedding engines.

One subclass per provider module supplies provider facts and wire fixtures; the
interface asserts the uniform embedding-engine contract:

- API_PINNED declared in the provider's models module
- prepare() stores argument.prop.entries on prepared_input
- build_request wire shape (method, url, auth header convention, body subset)
- forward through MockAPI -> typed response metadata + one vector per entry
- malformed responses fail typed parsing
- HTTP 401 -> EngineAuthenticationError
- new_dim client-side truncation with L2 re-normalization (norm ~1.0)
- usage metadata tracking (MetadataTracker "EmbeddingEngine" branch; OpenAI only)
- live smoke (--engine-api=live + provider key in api_keys.log)

The llama.cpp engine is not part of this interface: it builds a per-call
httpx.Client (no transport_client injection point for MockAPI) and bypasses
the shared transport error lattice. See test_llama_cpp_engine.py.
"""

from __future__ import annotations

import importlib
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import httpx
import pytest
from pydantic import ValidationError

from symai.backend.transport import EngineAuthenticationError
from symai.components import MetadataTracker
from tests.engines.mock_api import DUMMY_KEY, MockAPI

KEYS_LOG = Path("api_keys.log")
MOCK_DIMS = 8  # >= new_dim used in the truncation test, so truncation is observable
NEW_DIM = 4


def load_key(provider: str, pattern: str) -> str | None:
    raw = KEYS_LOG.read_text()
    section = re.search(rf"^{provider}:\n((?:\s+.*\n)+)", raw, re.MULTILINE)
    if not section:
        return None
    match = re.search(pattern, section.group(1))
    return match.group(1) if match else None


def normalized_vector(seed: float, dims: int = MOCK_DIMS) -> list[float]:
    """Deterministic L2-normalized mock vector (norm exactly 1 pre-truncation)."""
    raw = [seed * (index + 1) for index in range(dims)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


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


class EmbeddingTestInterface:
    """Uniform contract checks for one embedding provider engine. Subclass per provider."""

    # --- provider facts (override) ---
    engine_cls: ClassVar = None
    response_cls: ClassVar = None
    default_model: ClassVar[str] = ""
    expected_dims: ClassVar[int] = 0  # untruncated server-side dims for default_model
    wire_url: ClassVar[str] = ""
    auth_header_name: ClassVar[str] = "Authorization"
    auth_header_prefix: ClassVar[str] = "Bearer "
    api_pinned: ClassVar[str] = ""
    api_pinned_module: ClassVar[str] = ""
    keys_log_section: ClassVar[str] = ""
    keys_log_pattern: ClassVar[str] = ""
    supports_usage: ClassVar[bool] = False

    MOCK_ENTRIES: ClassVar[tuple[str, ...]] = ("hello", "world")

    # --- provider hooks (override) ---
    def mock_response_json(self) -> dict:
        """A realistic captured provider response with one vector per MOCK_ENTRIES item."""
        raise NotImplementedError

    def response_dropping_required(self, payload: dict) -> dict:
        """Malformed variant of mock_response_json that must fail typed parsing."""
        raise NotImplementedError

    def expected_request_body_subset(self) -> dict:
        """Key wire fields the engine must send for a plain batched embed."""
        raise NotImplementedError

    def make_engine(self, api_key=DUMMY_KEY, model: str | None = None):
        return self.engine_cls(api_key=api_key, model=model or self.default_model)

    def make_live_engine(self, api_key: str, model: str | None = None):
        return self.make_engine(api_key, model)

    # --- shared helpers ---
    def make_argument(self, entries=None, kwargs=None):
        return SimpleNamespace(
            prop=SimpleNamespace(
                entries=list(entries if entries is not None else self.MOCK_ENTRIES),
                prepared_input=None,
                processed_input=None,
            ),
            kwargs=kwargs or {},
        )

    def forward_through_mock(self, payload=None, entries=None, **kwargs):
        engine = self.make_engine()
        argument = self.make_argument(entries=entries, kwargs=kwargs)
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
            pytest.skip("use --engine-api=live to run live embedding API requests")
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

        assert argument.prop.prepared_input == list(self.MOCK_ENTRIES)

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
        vectors = output[0]
        assert len(vectors) == len(self.MOCK_ENTRIES)
        assert all(isinstance(v, list) and len(v) == MOCK_DIMS for v in vectors)
        assert all(isinstance(x, float) for v in vectors for x in v)
        assert isinstance(metadata["raw_output"], self.response_cls)

    def test_malformed_response_fails_typed_parsing(self):
        engine = self.make_engine()
        payload = self.response_dropping_required(self.mock_response_json())

        with MockAPI(engine, lambda request: httpx.Response(200, json=payload, request=request)):
            argument = self.make_argument()
            engine.prepare(argument)
            with pytest.raises((ValidationError, ValueError)):
                engine.forward(argument)

    def test_http_401_raises_authentication_error(self):
        engine = self.make_engine()
        error_body = {"error": {"code": "invalid_api_key", "message": "bad key"}}

        with MockAPI(engine, lambda request: httpx.Response(401, json=error_body, request=request)):
            argument = self.make_argument()
            engine.prepare(argument)
            with pytest.raises(EngineAuthenticationError):
                engine.forward(argument)

    def test_new_dim_truncates_and_normalizes(self):
        # new_dim truncation is client-side; the engine must L2 re-normalize the
        # truncated vector, so the post-truncate norm is ~1.0 (not the norm of the
        # first NEW_DIM components of a unit vector).
        _api, output, _metadata = self.forward_through_mock(new_dim=NEW_DIM)

        vectors = output[0]
        assert len(vectors) == len(self.MOCK_ENTRIES)
        for vector in vectors:
            assert len(vector) == NEW_DIM
            norm = math.sqrt(sum(x * x for x in vector))
            assert norm == pytest.approx(1.0)

    def test_usage_tracker_counts_embedding_tokens(self):
        if not self.supports_usage:
            pytest.skip(f"{self.engine_cls.__name__} has no usage metadata (no tracker branch)")
        engine = self.make_engine()

        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json=self.mock_response_json(), request=request),
        ):
            with MetadataTracker() as tracker:
                argument = self.make_argument()
                engine.prepare(argument)
                engine.forward(argument)
                engine.forward(argument)
            usage = tracker.usage

        # NOTE: components.py MetadataTracker's "EmbeddingEngine" branch reads
        # metadata["raw_output"].usage.prompt_tokens / .total_tokens.
        details = usage[(self.engine_cls.__name__, self.default_model)]
        mock_usage = self.mock_response_json()["usage"]
        assert details["usage"]["prompt_tokens"] == 2 * mock_usage["prompt_tokens"]
        assert details["usage"]["completion_tokens"] == 0
        assert details["usage"]["total_tokens"] == 2 * mock_usage["total_tokens"]
        assert details["usage"]["total_calls"] == 2

    @pytest.mark.engine_live
    def test_live_smoke(self, engine_api_mode):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(api_key)
        argument = self.make_argument()
        engine.prepare(argument)
        output, metadata = engine.forward(argument)

        vectors = output[0]
        assert len(vectors) == len(self.MOCK_ENTRIES)
        assert all(isinstance(v, list) and len(v) == self.expected_dims for v in vectors)
        assert all(isinstance(x, float) for v in vectors for x in v)
        assert isinstance(metadata["raw_output"], self.response_cls)

        truncated_argument = self.make_argument(entries=["hello"], kwargs={"new_dim": 128})
        engine.prepare(truncated_argument)
        truncated_output, _ = engine.forward(truncated_argument)
        assert len(truncated_output[0]) == 1
        assert len(truncated_output[0][0]) == 128
