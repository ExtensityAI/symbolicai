import base64
import json
from pathlib import Path

import httpx
import pytest

from symai.backend.engines.neurosymbolic.google.engine import GOOGLE_API_BASE, GoogleEngine
from symai.backend.engines.neurosymbolic.google.models import (
    API_PINNED,
    GOOGLE_MODEL_SPECS,
    SUPPORTED_GOOGLE_MODELS,
    GoogleResponse,
    google_strip_prefix,
)
from symai.components import MetadataTracker
from tests.engines.mock_api import MockAPI
from tests.engines.neurosymbolic.interface import NeurosymbolicEngineTestInterface


class TestGoogleEngine(NeurosymbolicEngineTestInterface):
    engine_cls = GoogleEngine
    supported_models = tuple(SUPPORTED_GOOGLE_MODELS)
    model_specs = GOOGLE_MODEL_SPECS
    default_model = "gemini:gemini-2.5-flash"
    response_cls = GoogleResponse
    wire_provider = "google"
    wire_operation = "models.generate_content"
    wire_url = f"{GOOGLE_API_BASE}/gemini-2.5-flash:generateContent"
    supports_streaming = True
    api_pinned = API_PINNED
    stream_via_url = True
    stream_options_expected = None
    supports_token_counting = True
    wire_model_present = False
    mock_usage_key = "usageMetadata"
    max_tokens_wire_key = "maxOutputTokens"
    wire_input_key = "contents"

    def wire_generation_value(self, body, key):
        return body["generationConfig"][key]

    def usage_dump(self, raw_output):
        return raw_output.usage_metadata.model_dump(by_alias=True)

    def wire_input_expected(self, argument):
        # NOTE: Gemini moves system to systemInstruction and wraps text in parts.
        engine = self.make_engine()
        _, contents = engine._build_wire_contents(argument.prop.prepared_input)
        return contents

    def spec_for(self, model):
        return self.model_specs[google_strip_prefix(model)]

    def expected_wire_model(self, model=None):
        return google_strip_prefix(model or self.default_model)

    def assert_auth_headers(self, headers):
        key = headers.get("x-goog-api-key")
        assert key == "sk-test-not-a-real-key"

    def usage_prompt_tokens(self, usage):
        return usage.get("promptTokenCount", usage.get("prompt_tokens"))

    def usage_completion_tokens(self, usage):
        return usage.get("candidatesTokenCount", usage.get("completion_tokens", 0)) + (
            usage.get("thoughtsTokenCount") or 0
        )

    def usage_total_tokens(self, usage):
        return usage.get("totalTokenCount", usage.get("total_tokens"))

    def assert_self_prompt_response_format(self, body):
        # NOTE: json_object maps to responseMimeType on Gemini, not response_format.
        assert body["generationConfig"]["responseMimeType"] == "application/json"

    def assert_self_prompt_messages(self, body):
        assert (
            "Generate a new system or developer prompt"
            in body["systemInstruction"]["parts"][0]["text"]
        )
        assert body["contents"][0]["role"] == "user"
        assert json.loads(body["contents"][0]["parts"][0]["text"]) == {
            "system": "old system",
            "user": "old user",
        }

    def inject_self_prompt_response(self, payload, content):
        payload["candidates"][0]["content"]["parts"] = [{"text": content}]
        return payload

    def vision_messages(self, image_path):
        encoded = base64.b64encode(Path(image_path).read_bytes()).decode()
        return [
            {
                "role": "user",
                "content": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": encoded}},
                    {"text": "What is in this image? Answer in one word."},
                ],
            }
        ]

    def mock_response_json(self):
        return {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "Add one and one.", "thought": True},
                            {"text": "2"},
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 2,
                "totalTokenCount": 15,
                "thoughtsTokenCount": 3,
                "cachedContentTokenCount": 4,
            },
        }

    def response_dropping_content(self, payload):
        payload["candidates"] = []
        return payload

    def response_dropping_usage(self, payload):
        del payload["usageMetadata"]
        return payload

    def mock_tool_call_json(self):
        payload = self.mock_response_json()
        payload["candidates"][0]["content"]["parts"] = [
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": {"location": "Paris"},
                }
            }
        ]
        return payload

    def weather_tool_spec(self):
        return {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }

    def tool_choice_kwarg(self):
        return {}

    def mock_sse_body(self):
        chunks = [
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Add one ", "thought": True}],
                        },
                        "index": 0,
                    }
                ]
            },
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "and one.", "thought": True}],
                        },
                        "index": 0,
                    }
                ]
            },
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "2"}]}, "index": 0}]},
            {
                "candidates": [
                    {"content": {"role": "model", "parts": []}, "finishReason": "STOP", "index": 0}
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 15,
                    "thoughtsTokenCount": 3,
                },
            },
        ]
        lines = []
        for chunk in chunks:
            lines.append(f"data: {json.dumps(chunk)}")
            lines.append("")
        return "\n".join(lines).encode("utf-8")

    def test_forward_streams_sse_and_aggregates_response(self):
        engine = self.make_engine(client_max_retries=0)

        with MockAPI(
            engine,
            lambda request: httpx.Response(
                200,
                content=self.mock_sse_body(),
                headers={"content-type": "text/event-stream"},
                request=request,
            ),
        ):
            output, metadata = engine.forward(self.make_prepared_argument(kwargs={"stream": True}))

        assert output == ["2"]
        assert metadata["thinking"] == "Add one and one."
        raw_output = metadata["raw_output"]
        assert isinstance(raw_output, GoogleResponse)
        assert raw_output.usage_metadata.total_token_count == 15

    def test_usage_tracking_bills_thinking_as_output(self):
        engine = self.make_engine(client_max_retries=0)

        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json=self.mock_response_json(), request=request),
        ):
            with MetadataTracker() as tracker:
                engine.forward(self.make_prepared_argument())
            details = tracker.usage[(self.engine_cls.__name__, self.default_model)]

        # candidates (2) + thoughts (3) — Gemini bills output including thinking
        assert details["usage"]["completion_tokens"] == 5
        assert details["usage"]["prompt_tokens"] == 10
        assert details["usage"]["total_tokens"] == 15
        assert details["prompt_breakdown"]["cached_tokens"] == 4
        assert details["completion_breakdown"]["reasoning_tokens"] == 3

    def test_build_request_strips_provider_prefix_from_wire_url(self):
        request = self.make_engine().build_request(self.make_prepared_argument())

        assert request.url.endswith("gemini-2.5-flash:generateContent")

    def test_build_request_maps_assistant_role_to_model(self):
        engine = self.make_engine()
        messages = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ]

        body = engine.build_request(self.make_prepared_argument(messages=messages)).body()

        assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]

    def test_build_request_splits_system_instruction(self):
        engine = self.make_engine()
        argument = self.make_query_argument("What is 1+1?")
        engine.prepare(argument)

        body = engine.build_request(argument).body()

        assert body["systemInstruction"]["parts"][0]["text"]
        assert all(c["role"] != "system" for c in body["contents"])

    def test_build_request_treats_empty_stop_as_unset(self):
        engine = self.make_engine()

        empty_stop = engine.build_request(self.make_prepared_argument(kwargs={"stop": ""}))
        assert "stopSequences" not in empty_stop.body().get("generationConfig", {})

        user_stop = engine.build_request(self.make_prepared_argument(kwargs={"stop": "END"}))
        assert user_stop.body()["generationConfig"]["stopSequences"] == ["END"]

    def test_build_request_thinking_config(self):
        engine = self.make_engine()

        request = engine.build_request(
            self.make_prepared_argument(kwargs={"thinking": {"thinking_budget": 2048}})
        )
        config = request.body()["generationConfig"]

        assert config["thinkingConfig"] == {"includeThoughts": True, "thinkingBudget": 2048}

    def test_build_request_rejects_non_vision_media_markers(self):
        # NOTE: video/audio/document upload used Gemini's Files API via the SDK, which
        # the raw-REST engine does not implement — it must refuse loudly, never strip.
        engine = self.make_engine()

        for marker in ("video", "audio", "document"):
            marked = [{"role": "user", "content": f"describe <<{marker}:/tmp/sample.bin:>> please"}]
            with pytest.raises(NotImplementedError, match=marker):
                engine.build_request(self.make_prepared_argument(messages=marked))

    def test_compute_required_tokens_uses_count_tokens_endpoint(self):
        engine = self.make_engine(client_max_retries=0)

        with MockAPI(
            engine,
            lambda request: httpx.Response(200, json={"totalTokens": 42}, request=request),
        ) as api:
            tokens = engine.compute_required_tokens([{"role": "user", "content": "hello"}])

        assert tokens == 42
        assert api.last_request.url.path.endswith(":countTokens")
        assert api.last_request.headers["x-goog-api-key"] == "sk-test-not-a-real-key"
