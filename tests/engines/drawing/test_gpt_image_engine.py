"""OpenAI gpt-image drawing engine tests: mock wire replay + live smoke (--engine-api=live).

Ports the gpt-image classes of the legacy tests/engines/drawing/test_drawing_engine.py
(create / variation / edit) onto the shared DrawingEngineTestInterface, plus raw
multipart wire assertions for the edits/variations endpoints.
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from symai.backend.engines import drawing as drawing_package
from symai.backend.engines.drawing.openai import GPTImageEngine
from symai.backend.engines.drawing.openai.models import (
    API_PINNED,
    OPENAI_IMAGES_EDITS_URL,
    OPENAI_IMAGES_GENERATIONS_URL,
    OPENAI_IMAGES_VARIATIONS_URL,
    OpenAIImagesResponse,
)
from symai.backend.settings import SYMAI_CONFIG
from symai.functional import EngineRepository
from tests.engines.drawing.interface import MOCK_PNG_BYTES, MOCK_PROMPT, DrawingEngineTestInterface
from tests.engines.mock_api import DUMMY_KEY, MockAPI

MOCK_EDIT_PROMPT = "give the cat a medieval helmet"


def mock_images_response_json() -> dict:
    return {
        "created": 1752700000,
        "data": [{"b64_json": base64.b64encode(MOCK_PNG_BYTES).decode("ascii")}],
        "usage": {
            "total_tokens": 100,
            "input_tokens": 10,
            "output_tokens": 90,
            "input_tokens_details": {"image_tokens": 0, "text_tokens": 10},
        },
    }


class TestGPTImageEngine(DrawingEngineTestInterface):
    engine_cls = GPTImageEngine
    default_model = "gpt-image-1"
    wire_url = OPENAI_IMAGES_GENERATIONS_URL
    api_pinned = API_PINNED
    api_pinned_module = "symai.backend.engines.drawing.openai.models"
    api_key_env = "OPENAI_API_KEY"

    def mock_handler(self):
        return lambda request: httpx.Response(
            200, json=mock_images_response_json(), request=request
        )

    def expected_request_body_subset(self) -> dict:
        return {
            "model": self.default_model,
            "prompt": MOCK_PROMPT,
            "n": 1,
            "size": "1024x1024",
            "quality": "medium",
            "moderation": "auto",
            "background": "auto",
            "output_format": "png",
        }

    def mock_forward_kwargs(self) -> dict:
        return {"operation": "create", "size": "1024x1024"}

    def live_forward_kwargs(self) -> dict:
        return {"size": "1024x1024"}

    def assert_raw_output(self, metadata: dict):
        assert isinstance(metadata["raw_output"], OpenAIImagesResponse)

    # --- gpt-image-specific contract checks ---
    def test_forward_mock_sends_generations_body(self):
        api, _output, _metadata = self.forward_through_mock()

        assert api.last_request.headers["content-type"] == "application/json"

    def test_malformed_response_fails_typed_parsing(self):
        payload = mock_images_response_json()
        del payload["data"]

        engine = self.make_engine()
        with MockAPI(engine, lambda request: httpx.Response(200, json=payload, request=request)):
            argument = self.make_argument(kwargs=self.mock_forward_kwargs())
            engine.prepare(argument)
            with pytest.raises(ValidationError):
                engine.forward(argument)

    def test_int_size_normalized_to_wire_string(self):
        engine = self.make_engine()
        argument = self.make_argument(kwargs={"operation": "create", "size": 1024})
        engine.prepare(argument)

        request = engine.build_request(argument)

        assert request.body()["size"] == "1024x1024"

    def test_variation_multipart_wire(self, tmp_path):
        """Port of legacy test_gpt_image_variation: dall-e-2, int size, url response."""
        image_path = tmp_path / "cat.png"
        image_path.write_bytes(MOCK_PNG_BYTES)

        engine = self.make_engine(api_key=DUMMY_KEY)

        # NOTE: GPTImageResult downloads url responses with plain httpx.get (not the
        # engine transport); mock with b64_json to keep the variation assertion local.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=mock_images_response_json(), request=request)

        with MockAPI(engine, handler) as api:
            argument = self.make_argument(
                kwargs={
                    "operation": "variation",
                    "model": "dall-e-2",
                    "image_path": str(image_path),
                    "n": 1,
                    "size": 1024,
                    "response_format": "b64_json",
                }
            )
            engine.prepare(argument)
            output, metadata = engine.forward(argument)

        request = api.last_request
        assert str(request.url) == OPENAI_IMAGES_VARIATIONS_URL
        content_type = request.headers["content-type"]
        assert content_type.startswith("multipart/form-data")
        body = request.content
        assert b'name="image"; filename="cat.png"' in body
        assert MOCK_PNG_BYTES in body
        assert b'name="model"\r\n\r\ndall-e-2' in body
        assert b'name="n"\r\n\r\n1' in body
        assert b'name="size"\r\n\r\n1024x1024' in body
        assert b'name="response_format"\r\n\r\nb64_json' in body
        assert output[0].value
        self.assert_raw_output(metadata)

    def test_edit_multipart_wire(self, tmp_path):
        """Port of legacy test_gpt_image_edit: gpt-image-1, prompt + image + quality."""
        image_path = tmp_path / "cat.png"
        image_path.write_bytes(MOCK_PNG_BYTES)

        engine = self.make_engine(api_key=DUMMY_KEY)
        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json=mock_images_response_json(), request=request),
        ) as api:
            argument = self.make_argument(
                prompt=MOCK_EDIT_PROMPT,
                kwargs={
                    "operation": "edit",
                    "model": "gpt-image-1",
                    "image_path": str(image_path),
                    "n": 1,
                    "size": 1024,
                    "quality": "medium",
                },
            )
            engine.prepare(argument)
            output, metadata = engine.forward(argument)

        request = api.last_request
        assert str(request.url) == OPENAI_IMAGES_EDITS_URL
        assert request.headers["content-type"].startswith("multipart/form-data")
        body = request.content
        assert b'name="image"; filename="cat.png"' in body
        assert MOCK_PNG_BYTES in body
        assert f'name="prompt"\r\n\r\n{MOCK_EDIT_PROMPT}'.encode() in body
        assert b'name="quality"\r\n\r\nmedium' in body
        assert output[0].value
        self.assert_raw_output(metadata)

    def test_edit_multiple_images_and_mask(self, tmp_path):
        image_a = tmp_path / "a.png"
        image_b = tmp_path / "b.png"
        mask = tmp_path / "mask.png"
        for path in (image_a, image_b, mask):
            path.write_bytes(MOCK_PNG_BYTES)

        engine = self.make_engine()
        argument = self.make_argument(
            kwargs={
                "operation": "edit",
                "image_path": [str(image_a), str(image_b)],
                "mask_path": str(mask),
            }
        )
        engine.prepare(argument)

        request = engine.build_request(argument)

        assert [field for field, _part in request.files] == ["image", "image", "mask"]

    def test_missing_operation_raises(self):
        engine = self.make_engine()
        argument = self.make_argument()
        engine.prepare(argument)
        with pytest.raises(ValueError, match="Operation not specified"):
            engine.forward(argument)

    def test_register_from_package_registers_drawing_engine(self, monkeypatch):
        monkeypatch.setitem(SYMAI_CONFIG, "DRAWING_ENGINE_MODEL", "gpt-image-1")
        repository = EngineRepository()
        repository._engines.pop("drawing", None)

        EngineRepository.register_from_package(drawing_package, allow_engine_override=True)

        registered = EngineRepository.list()
        assert isinstance(registered.get("drawing"), GPTImageEngine)

    @pytest.mark.engine_live
    def test_live_smoke(self, engine_api_mode):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(api_key)
        argument = self.make_argument(
            kwargs={**self.mock_forward_kwargs(), **self.live_forward_kwargs()}
        )
        engine.prepare(argument)
        output, _metadata = engine.forward(argument)

        image_bytes = Path(output[0].value[0]).read_bytes()
        assert image_bytes, "live image file must be non-empty"
        # gpt-image-1 returns b64_json in the requested output_format (default png).
        is_png = image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        is_jpeg = image_bytes.startswith(b"\xff\xd8\xff")
        is_webp = image_bytes[8:12] == b"WEBP"
        assert is_png or is_jpeg or is_webp, f"unexpected image format: {image_bytes[:12]!r}"
