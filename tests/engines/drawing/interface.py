"""Shared test interface for the migrated drawing engines.

One subclass per provider module supplies provider facts and wire fixtures; the
interface asserts the uniform drawing-engine contract:

- API_PINNED declared in the provider's models module
- prepare() produces the provider's expected prepared_input
- build_request wire shape (url, auth header convention, body subset)
- forward through MockAPI -> non-empty local image file(s) with the mocked bytes
- HTTP 401 -> EngineAuthenticationError
- live smoke (--engine-api=live + provider key in the environment)
"""

from __future__ import annotations

import base64
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import httpx
import pytest

from symai.backend.transport import EngineAuthenticationError
from tests.engines.mock_api import DUMMY_KEY, MockAPI

MOCK_PROMPT = "a fluffy cat with a cowboy hat"

# 1x1 transparent PNG — stand-in for provider image bytes in mock mode.
MOCK_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def assert_image_paths(value) -> list[Path]:
    """Drawing engines return a list of local image file paths; each must exist and be non-empty."""
    assert isinstance(value, list) and value, "result value must be a non-empty list of image paths"
    paths = []
    for item in value:
        path = Path(item)
        assert path.exists() and path.stat().st_size > 0, f"image file missing or empty: {path}"
        paths.append(path)
    return paths


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


class DrawingEngineTestInterface:
    """Uniform contract checks for one drawing provider engine. Subclass per provider."""

    # --- provider facts (override) ---
    engine_cls: ClassVar = None
    default_model: ClassVar[str] = ""
    wire_url: ClassVar[str] = ""
    auth_header_name: ClassVar[str] = "Authorization"
    auth_header_prefix: ClassVar[str] = "Bearer "
    api_pinned: ClassVar[str] = ""
    api_pinned_module: ClassVar[str] = ""
    # NOTE: env var holding the provider key for live runs (never hardcode keys).
    api_key_env: ClassVar[str] = ""

    # --- provider hooks (override) ---
    def mock_handler(self):
        """httpx.MockTransport handler replaying the provider's full call sequence,
        serving MOCK_PNG_BYTES as the image payload (inline or downloaded)."""
        raise NotImplementedError

    def expected_request_body_subset(self) -> dict:
        """Key wire fields the engine must send for a plain prompt."""
        raise NotImplementedError

    def expected_prepared_input(self, prompt: str):
        """What prepare() must store on argument.prop.prepared_input."""
        return prompt

    def mock_forward_kwargs(self) -> dict:
        """argument kwargs for the mock forward (e.g. operation='create')."""
        return {}

    def live_forward_kwargs(self) -> dict:
        """Cheap/fast knobs for the live smoke (small size, few steps)."""
        return {}

    def configure_mock_engine(self, engine) -> None:
        """Tweak the mock engine before forward (e.g. zero poll interval)."""

    def make_engine(self, api_key=DUMMY_KEY):
        return self.engine_cls(api_key=api_key, model=self.default_model)

    def make_live_engine(self, api_key: str):
        return self.make_engine(api_key)

    def assert_raw_output(self, metadata: dict):
        """metadata['raw_output'] carries the typed provider response (if any)."""

    # --- shared helpers ---
    def make_argument(self, prompt=MOCK_PROMPT, kwargs=None):
        return SimpleNamespace(
            prop=SimpleNamespace(processed_input=prompt, prepared_input=None),
            kwargs=kwargs or {},
        )

    def forward_through_mock(self, handler=None, **extra_kwargs):
        engine = self.make_engine()
        self.configure_mock_engine(engine)
        kwargs = {**self.mock_forward_kwargs(), **extra_kwargs}
        argument = self.make_argument(kwargs=kwargs)
        with MockAPI(engine, handler or self.mock_handler()) as api:
            engine.prepare(argument)
            output, metadata = engine.forward(argument)
        return api, output, metadata

    def require_live(self, engine_api_mode) -> str:
        if engine_api_mode != "live":
            pytest.skip("use --engine-api=live to run live drawing API requests")
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            pytest.skip(f"{self.api_key_env} not set; live test skipped")
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

        assert argument.prop.prepared_input == self.expected_prepared_input(MOCK_PROMPT)

    def test_build_request_wire_shape(self):
        engine = self.make_engine()
        argument = self.make_argument(kwargs=self.mock_forward_kwargs())
        engine.prepare(argument)

        request = engine.build_request(argument)

        assert request.method == "POST"
        assert request.url == self.wire_url
        self.assert_auth_header(request.headers)
        assert_body_subset(request.body(), self.expected_request_body_subset())

    def test_forward_mock_returns_image(self):
        api, output, metadata = self.forward_through_mock()

        assert api.requests, "expected at least one wire request"
        paths = assert_image_paths(output[0].value)
        assert paths[0].read_bytes() == MOCK_PNG_BYTES
        self.assert_raw_output(metadata)

    def test_http_401_raises_authentication_error(self):
        engine = self.make_engine()
        self.configure_mock_engine(engine)
        error_body = {"error": {"code": "invalid_api_key", "message": "bad key"}}

        with MockAPI(engine, lambda request: httpx.Response(401, json=error_body, request=request)):
            argument = self.make_argument(kwargs=self.mock_forward_kwargs())
            engine.prepare(argument)
            with pytest.raises(EngineAuthenticationError):
                engine.forward(argument)

    @pytest.mark.engine_live
    def test_live_smoke(self, engine_api_mode):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(api_key)
        argument = self.make_argument(
            kwargs={**self.mock_forward_kwargs(), **self.live_forward_kwargs()}
        )
        engine.prepare(argument)
        output, _metadata = engine.forward(argument)

        assert_image_paths(output[0].value)
