"""Shared test interface for the migrated text-to-speech engines.

One subclass per provider module supplies provider facts and wire fixtures; the
interface asserts the uniform TTS-engine contract:

- API_PINNED declared in the provider's models module
- prepare() produces the provider's expected prepared_input (voice, path, prompt)
- build_request wire shape (url, auth header convention, body subset)
- forward through MockAPI serving audio bytes -> bytes passthrough into the
  Result and onto the prepared file path, content type in metadata
- invalid typed request fields fail validation (strict request payload)
- HTTP 401 -> EngineAuthenticationError (error body is JSON)
- live smoke (--engine-api=live + provider key in the environment)
"""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace
from typing import ClassVar

import httpx
import pytest
from pydantic import ValidationError

from symai.backend.transport import EngineAuthenticationError
from tests.engines.mock_api import DUMMY_KEY, MockAPI

MOCK_VOICE = "Nova"  # mixed case on purpose: prepare() must lowercase it
MOCK_PROMPT = "Spain won the UEFA Euro 2024 final two to one against England."
# NOTE: not a real MP3 — a deterministic byte blob with the ID3 magic so the mock
# passthrough assertions exercise exactly what the engine moves around.
MOCK_AUDIO_BYTES = b"ID3\x04\x00\x00\x00\x00\x00\x21" + bytes(range(256)) * 4


def assert_body_subset(body, subset, path="body"):
    """Recursive subset check: dicts key-wise, lists element-wise, scalars by equality."""
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


class TextToSpeechTestInterface:
    """Uniform contract checks for one TTS provider engine. Subclass per provider."""

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
    mock_content_type: ClassVar[str] = "audio/mpeg"
    live_model: ClassVar[str] = ""

    # --- provider hooks (override) ---
    def expected_request_body_subset(self) -> dict:
        """Key wire fields the engine must send for a plain synthesis."""
        raise NotImplementedError

    def invalid_request_kwargs(self) -> dict:
        """Forward kwargs violating the strict typed request payload."""
        return {"speed": 9.9}

    def make_engine(self, api_key=DUMMY_KEY):
        return self.engine_cls(api_key=api_key, model=self.default_model)

    def make_live_engine(self, api_key: str):
        return self.engine_cls(api_key=api_key, model=self.live_model or self.default_model)

    # --- shared helpers ---
    def make_argument(self, path, prompt=MOCK_PROMPT, kwargs=None):
        return SimpleNamespace(
            prop=SimpleNamespace(prompt=prompt, processed_input=None, prepared_input=None),
            kwargs={"voice": MOCK_VOICE, "path": str(path), **(kwargs or {})},
        )

    def forward_through_mock(self, path, payload=None, content_type=None, **extra_kwargs):
        engine = self.make_engine()
        argument = self.make_argument(path, kwargs=extra_kwargs)
        with MockAPI(
            engine,
            lambda request: httpx.Response(
                200,
                content=payload if payload is not None else MOCK_AUDIO_BYTES,
                headers={"content-type": content_type or self.mock_content_type},
                request=request,
            ),
        ) as api:
            engine.prepare(argument)
            output, metadata = engine.forward(argument)
        return api, output, metadata

    def require_live(self, engine_api_mode) -> str:
        if engine_api_mode != "live":
            pytest.skip("use --engine-api=live to run live TTS API requests")
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

    def test_prepare_produces_expected_prepared_input(self, tmp_path):
        engine = self.make_engine()
        audio_path = tmp_path / "speech.mp3"
        argument = self.make_argument(audio_path)

        engine.prepare(argument)

        assert argument.prop.prepared_input == (MOCK_VOICE.lower(), str(audio_path), MOCK_PROMPT)

    def test_prepare_requires_voice_and_path(self):
        engine = self.make_engine()
        argument = SimpleNamespace(
            prop=SimpleNamespace(prompt=MOCK_PROMPT, processed_input=None, prepared_input=None),
            kwargs={},
        )

        with pytest.raises(AssertionError):
            engine.prepare(argument)

    def test_build_request_wire_shape(self, tmp_path):
        engine = self.make_engine()
        argument = self.make_argument(tmp_path / "speech.mp3")
        engine.prepare(argument)

        request = engine.build_request(argument)

        assert request.method == "POST"
        assert request.url == self.wire_url
        self.assert_auth_header(request.headers)
        assert_body_subset(request.body(), self.expected_request_body_subset())

    def test_forward_mock_returns_audio_bytes(self, tmp_path):
        audio_path = tmp_path / "speech.mp3"
        api, output, metadata = self.forward_through_mock(audio_path)

        assert api.last_request.method == "POST"
        assert str(api.last_request.url) == self.wire_url
        self.assert_auth_header(dict(api.last_request.headers))
        assert_body_subset(api.last_body, self.expected_request_body_subset())
        # bytes passthrough: Result value, file on disk, and metadata content type
        assert output[0].value == MOCK_AUDIO_BYTES
        assert audio_path.read_bytes() == MOCK_AUDIO_BYTES
        assert metadata["content_type"] == self.mock_content_type

    def test_forward_mock_options_land_on_wire(self, tmp_path):
        api, _output, _metadata = self.forward_through_mock(
            tmp_path / "speech.wav", response_format="wav", speed=1.25
        )

        assert api.last_body["response_format"] == "wav"
        assert api.last_body["speed"] == 1.25

    def test_invalid_request_field_fails_typed_validation(self, tmp_path):
        engine = self.make_engine()
        argument = self.make_argument(tmp_path / "speech.mp3", kwargs=self.invalid_request_kwargs())
        engine.prepare(argument)

        with pytest.raises(ValidationError):
            engine.build_request(argument)

    def test_http_401_raises_authentication_error(self, tmp_path):
        engine = self.make_engine()
        error_body = {"error": {"code": "invalid_api_key", "message": "bad key"}}

        with MockAPI(engine, lambda request: httpx.Response(401, json=error_body, request=request)):
            argument = self.make_argument(tmp_path / "speech.mp3")
            engine.prepare(argument)
            with pytest.raises(EngineAuthenticationError):
                engine.forward(argument)

    @pytest.mark.engine_live
    def test_live_smoke(self, engine_api_mode, tmp_path):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(api_key)
        audio_path = tmp_path / "speech.mp3"
        argument = self.make_argument(audio_path, prompt="Hello from the symbolicai engine test.")
        engine.prepare(argument)
        output, metadata = engine.forward(argument)

        audio = output[0].value
        assert isinstance(audio, bytes) and len(audio) > 100, "expected non-trivial audio bytes"
        assert metadata["content_type"] == self.mock_content_type
        assert audio_path.read_bytes() == audio
        # MP3 frame sync (0xFFEx) or ID3 tag magic
        assert audio[:3] == b"ID3" or (audio[0] == 0xFF and audio[1] & 0xE0 == 0xE0), (
            "default format must be MP3"
        )
