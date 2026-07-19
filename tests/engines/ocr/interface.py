"""Shared test interface for the migrated OCR engines.

One subclass per provider module supplies provider facts and wire fixtures; the
interface asserts the uniform OCR-engine contract:

- API_PINNED declared in the provider's models module
- prepare() stores the document_url/image_url on prepared_input
- build_request wire shape (method, url, auth header convention, body subset)
- forward through MockAPI -> typed response metadata + non-empty markdown
  (document_url and base64 image_url routes, assembled and per_page values,
  image base64 extraction)
- local file resolution through the files upload + signed-url round trip
- malformed responses fail typed parsing
- HTTP 401 -> EngineAuthenticationError
- usage metadata tracking (MetadataTracker "MistralOCREngine" branch)
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
from symai.components import MetadataTracker
from tests.engines.mock_api import DUMMY_KEY, MockAPI

MOCK_DOCUMENT_URL = "https://example.com/papers/sample.pdf"
MOCK_IMAGE_DATA_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
MOCK_SIGNED_URL = "https://files.mistral.ai/signed/sample.pdf?signature=abc123"


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


class OCREngineTestInterface:
    """Uniform contract checks for one OCR provider engine. Subclass per provider."""

    # --- provider facts (override) ---
    engine_cls: ClassVar = None
    response_cls: ClassVar = None
    default_model: ClassVar[str] = ""
    wire_url: ClassVar[str] = ""
    auth_header_name: ClassVar[str] = "Authorization"
    auth_header_prefix: ClassVar[str] = "Bearer "
    api_pinned: ClassVar[str] = ""
    api_pinned_module: ClassVar[str] = ""
    # NOTE: env var holding the provider key for live runs (never hardcode keys).
    api_key_env: ClassVar[str] = ""
    live_document_url: ClassVar[str] = ""

    # --- provider hooks (override) ---
    def mock_response_json(self) -> dict:
        """A realistic captured provider response with >= 1 page of markdown."""
        raise NotImplementedError

    def response_dropping_required(self, payload: dict) -> dict:
        """Malformed variant of mock_response_json that must fail typed parsing."""
        raise NotImplementedError

    def expected_request_body_subset(self) -> dict:
        """Key wire fields the engine must send for a plain document_url OCR call."""
        raise NotImplementedError

    def make_engine(self, api_key=DUMMY_KEY, model: str | None = None):
        return self.engine_cls(api_key=api_key, model=model or self.default_model)

    def make_live_engine(self, api_key: str, model: str | None = None):
        return self.make_engine(api_key, model)

    # --- shared helpers ---
    def make_argument(self, document_url=MOCK_DOCUMENT_URL, image_url=None, kwargs=None):
        return SimpleNamespace(
            prop=SimpleNamespace(
                document_url=document_url,
                image_url=image_url,
                prepared_input=None,
                processed_input=None,
            ),
            kwargs=kwargs or {},
        )

    def forward_through_mock(self, payload=None, handler=None, **argument_kwargs):
        engine = self.make_engine()
        if handler is None:
            handler = lambda request: httpx.Response(  # noqa: E731
                200,
                json=payload if payload is not None else self.mock_response_json(),
                request=request,
            )
        with MockAPI(engine, handler) as api:
            argument = self.make_argument(**argument_kwargs)
            engine.prepare(argument)
            output, metadata = engine.forward(argument)
        return api, output, metadata

    def require_live(self, engine_api_mode) -> str:
        if engine_api_mode != "live":
            pytest.skip("use --engine-api=live to run live OCR API requests")
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

        assert argument.prop.prepared_input == MOCK_DOCUMENT_URL

    def test_build_request_wire_shape(self):
        engine = self.make_engine()
        argument = self.make_argument()
        engine.prepare(argument)

        request = engine.build_request(argument)

        assert request.method == "POST"
        assert request.url == self.wire_url
        self.assert_auth_header(request.headers)
        assert_body_subset(request.body(), self.expected_request_body_subset())

    def test_forward_mock_document_url_returns_markdown(self):
        api, output, metadata = self.forward_through_mock()

        assert api.last_request.method == "POST"
        assert str(api.last_request.url) == self.wire_url
        self.assert_auth_header(dict(api.last_request.headers))
        assert_body_subset(api.last_body, self.expected_request_body_subset())

        result = output[0]
        pages = self.mock_response_json()["pages"]
        assert result.value == "\n\n".join(page["markdown"] for page in pages)
        assert isinstance(metadata["raw_output"], self.response_cls)
        assert metadata["raw_output"].usage_info.pages_processed >= 1

    def test_forward_mock_image_url_base64(self):
        api, output, _metadata = self.forward_through_mock(
            document_url=None, image_url=MOCK_IMAGE_DATA_URI
        )

        assert_body_subset(
            api.last_body,
            {"document": {"type": "image_url", "image_url": MOCK_IMAGE_DATA_URI}},
        )
        assert output[0].value

    def test_forward_mock_per_page(self):
        _api, output, _metadata = self.forward_through_mock(kwargs={"per_page": True})

        pages = self.mock_response_json()["pages"]
        assert output[0].value == [page["markdown"] for page in pages]
        assert all(isinstance(page, str) for page in output[0].value)

    def test_forward_mock_include_image_base64(self):
        api, output, _metadata = self.forward_through_mock(kwargs={"include_image_base64": True})

        assert api.last_body["include_image_base64"] is True
        expected = {}
        for page in self.mock_response_json()["pages"]:
            for image in page["images"]:
                if image.get("image_base64"):
                    expected[image["id"]] = image["image_base64"]
        assert output[0].images == expected

    def test_forward_mock_local_file_upload_resolution(self, tmp_path):
        sample = tmp_path / "sample.pdf"
        sample.write_bytes(b"%PDF-1.4 mock bytes")
        requests_seen = []

        def handler(request):
            requests_seen.append((request.method, request.url.path))
            if request.method == "POST" and request.url.path == "/v1/files":
                return httpx.Response(
                    200,
                    json={
                        "id": "file-abc123",
                        "object": "file",
                        "size_bytes": sample.stat().st_size,
                        "created_at": 1752854400,
                        "filename": "sample.pdf",
                        "purpose": "ocr",
                    },
                    request=request,
                )
            if request.method == "GET" and request.url.path == "/v1/files/file-abc123/url":
                return httpx.Response(200, json={"url": MOCK_SIGNED_URL}, request=request)
            return httpx.Response(200, json=self.mock_response_json(), request=request)

        api, output, _metadata = self.forward_through_mock(
            handler=handler, document_url=str(sample)
        )

        assert requests_seen == [
            ("POST", "/v1/files"),
            ("GET", "/v1/files/file-abc123/url"),
            ("POST", "/v1/ocr"),
        ]
        assert_body_subset(
            api.last_body,
            {"document": {"type": "document_url", "document_url": MOCK_SIGNED_URL}},
        )
        assert output[0].value

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

    def test_usage_tracker_counts_processed_pages(self):
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

        # NOTE: components.py MetadataTracker's "MistralOCREngine" branch reads
        # metadata["raw_output"].usage_info.pages_processed / .doc_size_bytes.
        details = usage[(self.engine_cls.__name__, self.default_model)]
        mock_usage = self.mock_response_json()["usage_info"]
        assert details["usage"]["total_calls"] == 2
        assert details["extras"]["pages_processed"] == 2 * mock_usage["pages_processed"]
        assert details["extras"]["doc_size_bytes"] == 2 * mock_usage["doc_size_bytes"]

    @pytest.mark.engine_live
    def test_live_smoke(self, engine_api_mode):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(api_key)
        argument = self.make_argument(document_url=self.live_document_url)
        engine.prepare(argument)
        output, metadata = engine.forward(argument)

        assert isinstance(output[0].value, str) and output[0].value.strip()
        assert isinstance(metadata["raw_output"], self.response_cls)
        assert metadata["raw_output"].usage_info.pages_processed >= 1
