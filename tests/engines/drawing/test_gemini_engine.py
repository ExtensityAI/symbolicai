import base64

import httpx
import pytest
from pydantic import ValidationError

from symai.backend.engines.drawing.gemini import GeminiImageEngine
from symai.backend.engines.drawing.gemini.models import (
    API_PINNED,
    GEMINI_API_BASE,
    GeminiImageGenerateResponse,
)
from tests.engines.drawing.interface import (
    MOCK_PNG_BYTES,
    MOCK_PROMPT,
    DrawingEngineTestInterface,
    assert_body_subset,
    assert_image_paths,
)
from tests.engines.mock_api import MockAPI


class TestGeminiImageEngine(DrawingEngineTestInterface):
    engine_cls = GeminiImageEngine
    default_model = "gemini-2.5-flash-image"
    wire_url = f"{GEMINI_API_BASE}/models/{default_model}:generateContent"
    auth_header_name = "x-goog-api-key"
    auth_header_prefix = ""
    api_pinned = API_PINNED
    api_pinned_module = "symai.backend.engines.drawing.gemini.models"
    api_key_env = "GOOGLE_API_KEY"

    def mock_handler(self):
        payload = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": base64.b64encode(MOCK_PNG_BYTES).decode("ascii"),
                                }
                            },
                        ],
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        return lambda request: httpx.Response(200, json=payload, request=request)

    def expected_request_body_subset(self):
        return {
            "contents": [{"parts": [{"text": MOCK_PROMPT}]}],
            "generationConfig": {"responseModalities": ["IMAGE"]},
        }

    def mock_forward_kwargs(self):
        return {"operation": "create"}

    def assert_raw_output(self, metadata: dict):
        assert isinstance(metadata["raw_output"], GeminiImageGenerateResponse)

    def test_forward_mock_request_wire_shape(self):
        api, _output, _metadata = self.forward_through_mock()

        assert api.last_request.method == "POST"
        assert str(api.last_request.url) == self.wire_url
        self.assert_auth_header(dict(api.last_request.headers))
        assert_body_subset(api.last_body, self.expected_request_body_subset())

    def test_malformed_response_fails_typed_parsing(self):
        # candidates is required (min 1 item) — dropping it must fail typed parsing.
        engine = self.make_engine()
        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json={"promptFeedback": {}}, request=request),
        ):
            argument = self.make_argument(kwargs=self.mock_forward_kwargs())
            engine.prepare(argument)
            with pytest.raises(ValidationError):
                engine.forward(argument)

    @pytest.mark.engine_live
    @pytest.mark.parametrize("model", ["gemini-2.5-flash-image", "gemini-3-pro-image-preview"])
    def test_live_image_create_models(self, engine_api_mode, model):
        api_key = self.require_live(engine_api_mode)

        engine = self.make_live_engine(api_key)
        argument = self.make_argument(kwargs={**self.mock_forward_kwargs(), "model": model})
        engine.prepare(argument)
        output, _metadata = engine.forward(argument)

        assert_image_paths(output[0].value)
