from types import SimpleNamespace

import pytest
from openai.types.responses.response_create_params import ResponseCreateParamsNonStreaming
from pydantic import TypeAdapter, ValidationError

from symai.backend.engines.neurosymbolic.engine_openai_gptX import OpenAIResponsesEngine
from symai.backend.mixin.openai import (
    OPENAI_MODEL_SPECS,
    SUPPORTED_OPENAI_MODELS,
    OpenAIResponsesCreatePayload,
)
from symai.backend.settings import SYMAI_CONFIG
from symai.core import Argument

DUMMY_KEY = "sk-test-not-a-real-key"


def make_engine(model="openai:gpt-4.1-mini"):
    return OpenAIResponsesEngine(api_key=DUMMY_KEY, model=model)


def make_prepared_argument(kwargs=None, messages=None):
    if kwargs is None:
        kwargs = {}
    if messages is None:
        messages = [{"role": "user", "content": "hello"}]
    return SimpleNamespace(
        kwargs=kwargs,
        prop=SimpleNamespace(prepared_input=messages),
    )


def test_openai_responses_request_body_uses_payload_model_shape():
    engine = make_engine()

    request = engine.build_request(make_prepared_argument(kwargs={"max_output_tokens": 16}))

    assert set(request.body()) <= set(OpenAIResponsesCreatePayload.model_fields)


def test_openai_supported_models_are_current():
    assert set(OPENAI_MODEL_SPECS) == {
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4",
        "gpt-5.4-pro",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "o3-pro",
        "o3",
        "gpt-4.1",
        "gpt-4.1-mini",
    }
    assert [f"openai:{model}" for model in OPENAI_MODEL_SPECS] == SUPPORTED_OPENAI_MODELS


def test_openai_responses_prompt_is_symai_only():
    engine = make_engine()

    assert "prompt" not in OpenAIResponsesCreatePayload.model_fields
    assert "prompt" in engine.ignored_request_kwargs()


def test_openai_responses_ignored_request_kwargs_come_from_argument_defaults():
    engine = make_engine()

    assert engine.ignored_request_kwargs() == set(Argument.default_property_values())


def test_openai_responses_build_request_forwards_typed_supported_kwargs():
    engine = make_engine()
    parameters = {"type": "object", "properties": {}}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather.",
                "parameters": parameters,
            },
        }
    ]
    kwargs = {
        "model": "gpt-4.1-mini",
        "background": False,
        "context_management": [{"type": "compaction", "compact_threshold": 2048}],
        "conversation": "conv_123",
        "include": ["message.output_text.logprobs"],
        "instructions": "Stay terse.",
        "max_output_tokens": 32,
        "max_tool_calls": 2,
        "metadata": {"purpose": "test"},
        "moderation": {"model": "omni-moderation-latest"},
        "parallel_tool_calls": False,
        "previous_response_id": "resp_prev",
        "prompt_cache_key": "cache-key",
        "prompt_cache_retention": "24h",
        "safety_identifier": "safe-user",
        "service_tier": "default",
        "store": False,
        "stream": False,
        "stream_options": {"include_obfuscation": False},
        "temperature": 0.2,
        "text": {"format": {"type": "text"}},
        "tools": tools,
        "tool_choice": "auto",
        "top_logprobs": 2,
        "top_p": 0.9,
        "truncation": "auto",
        "user": "user-1",
        "extra_headers": {"X-Test": "1"},
        "extra_query": {"debug": "1"},
        "extra_body": {"trace": "yes"},
        "timeout": 5.0,
    }

    request = engine.build_request(make_prepared_argument(kwargs=kwargs))
    body = request.body()
    sdk_kwargs = request.kwargs()

    assert body["model"] == "gpt-4.1-mini"
    assert body["input"] == [{"role": "user", "content": "hello"}]
    assert body["background"] is False
    assert body["context_management"] == [{"type": "compaction", "compact_threshold": 2048}]
    assert body["conversation"] == "conv_123"
    assert body["include"] == ["message.output_text.logprobs"]
    assert body["instructions"] == "Stay terse."
    assert body["max_output_tokens"] == 32
    assert body["max_tool_calls"] == 2
    assert body["metadata"] == {"purpose": "test"}
    assert body["moderation"] == {"model": "omni-moderation-latest"}
    assert body["parallel_tool_calls"] is False
    assert body["previous_response_id"] == "resp_prev"
    assert body["prompt_cache_key"] == "cache-key"
    assert body["prompt_cache_retention"] == "24h"
    assert body["safety_identifier"] == "safe-user"
    assert body["service_tier"] == "default"
    assert body["store"] is False
    assert body["stream"] is False
    assert body["stream_options"] == {"include_obfuscation": False}
    assert body["temperature"] == 0.2
    assert body["text"] == {"format": {"type": "text"}}
    assert body["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather.",
            "parameters": parameters,
        }
    ]
    assert body["tool_choice"] == "auto"
    assert body["top_logprobs"] == 2
    assert body["top_p"] == 0.9
    assert body["truncation"] == "auto"
    assert body["user"] == "user-1"

    assert sdk_kwargs["input"] == [{"role": "user", "content": "hello"}]
    assert sdk_kwargs["model"] == "gpt-4.1-mini"
    assert sdk_kwargs["tools"] == body["tools"]
    assert sdk_kwargs["extra_headers"] == {"X-Test": "1"}
    assert sdk_kwargs["extra_query"] == {"debug": "1"}
    assert sdk_kwargs["extra_body"] == {"trace": "yes"}
    assert sdk_kwargs["timeout"] == 5.0


def test_openai_responses_uses_max_output_tokens_directly():
    engine = make_engine()
    kwargs = {"max_output_tokens": 32}

    request = engine.build_request(make_prepared_argument(kwargs=kwargs))
    body = request.body()

    assert body["max_output_tokens"] == 32
    assert "max_tokens" not in body


def test_openai_responses_reasoning_model_filters_sampling_params():
    engine = make_engine(model="openai:o3")
    kwargs = {
        "temperature": 0.2,
        "top_p": 0.9,
        "reasoning": {"effort": "low"},
    }

    request = engine.build_request(make_prepared_argument(kwargs=kwargs))
    body = request.body()

    assert "temperature" not in body
    assert "top_p" not in body
    assert body["reasoning"] == {"effort": "low"}


def test_openai_responses_pro_reasoning_model_forces_high_reasoning_effort():
    engine = make_engine(model="openai:o3-pro")

    request = engine.build_request(make_prepared_argument(kwargs={"reasoning": {"effort": "low"}}))

    assert request.body()["reasoning"] == {"effort": "high"}


def test_openai_responses_non_reasoning_model_keeps_sampling_params():
    engine = make_engine()
    kwargs = {
        "temperature": 0.2,
        "top_p": 0.9,
    }

    request = engine.build_request(make_prepared_argument(kwargs=kwargs))
    body = request.body()

    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_output_tokens": "64"},
        {"temperature": "0.2"},
        {"top_p": 2},
        {"stream": "false"},
        {"prompt_cache_retention": "forever"},
        {"stream_options": {"include_obfuscation": "yes"}},
    ],
)
def test_openai_responses_build_request_rejects_wrong_provider_kwarg_types(kwargs):
    engine = make_engine()

    with pytest.raises(ValidationError):
        engine.build_request(make_prepared_argument(kwargs=kwargs))


def test_openai_responses_build_request_rejects_removed_model_override():
    engine = make_engine()

    with pytest.raises(ValueError, match="Unsupported model: gpt-4o-mini"):
        engine.build_request(make_prepared_argument(kwargs={"model": "gpt-4o-mini"}))


def test_openai_responses_build_request_ignores_non_provider_kwargs_for_backward_compatibility():
    engine = make_engine()
    argument = make_prepared_argument(
        kwargs={
            "prompt": "Symai prompt kwarg",
            "examples": [],
            "context": "Symai context kwarg",
            "raw_input": False,
            "return_metadata": True,
            "max_output_tokens": 16,
        }
    )

    request = engine.build_request(argument)
    body = request.body()

    assert body["max_output_tokens"] == 16
    assert "prompt" not in body
    assert "examples" not in body
    assert "context" not in body
    assert "raw_input" not in body
    assert "return_metadata" not in body


def test_openai_responses_strict_request_kwargs_allows_symai_kwargs():
    engine = make_engine()
    argument = make_prepared_argument(
        kwargs={
            "strict_request_kwargs": True,
            "prompt": "Symai prompt kwarg",
            "examples": [],
            "raw_input": False,
            "return_metadata": True,
            "max_output_tokens": 16,
        }
    )

    request = engine.build_request(argument)

    assert request.body()["max_output_tokens"] == 16


def test_openai_responses_strict_request_kwargs_rejects_unknown_kwargs():
    engine = make_engine()
    argument = make_prepared_argument(
        kwargs={"strict_request_kwargs": True, "max_output_toknes": 16}
    )

    with pytest.raises(ValueError, match="Unsupported request kwargs"):
        engine.build_request(argument)


def test_openai_responses_strict_request_kwargs_rejects_removed_max_tokens_alias():
    engine = make_engine()
    argument = make_prepared_argument(kwargs={"strict_request_kwargs": True, "max_tokens": 16})

    with pytest.raises(ValueError, match="Unsupported request kwargs"):
        engine.build_request(argument)


def test_openai_responses_request_body_matches_openai_sdk_typed_dict():
    engine = make_engine()
    request = engine.build_request(
        make_prepared_argument(kwargs={"max_output_tokens": 16, "stream": False})
    )

    validated = TypeAdapter(ResponseCreateParamsNonStreaming).validate_python(request.body())

    assert validated["model"] == "gpt-4.1-mini"
    assert validated["max_output_tokens"] == 16


def test_openai_responses_prepare_build_call_parse_with_fake_client():
    engine = make_engine()
    argument = Argument(
        args=("What is 1+1?",),
        signature_kwargs={"max_output_tokens": 16},
        decorator_kwargs={"prompt": "Answer the question.", "examples": []},
    )
    argument.prop.instance = SimpleNamespace(global_context=("", ""), _kwargs={})
    argument.prop.processed_input = "What is 1+1?"
    responses_client = FakeResponsesClient()
    engine.client = SimpleNamespace(responses=responses_client)

    engine.prepare(argument)
    output, metadata = engine.forward(argument)

    assert [message["role"] for message in argument.prop.prepared_input] == ["system", "user"]
    assert responses_client.kwargs["model"] == "gpt-4.1-mini"
    assert responses_client.kwargs["max_output_tokens"] == 16
    assert responses_client.kwargs["input"] == argument.prop.prepared_input
    assert output == ["2"]
    assert metadata["raw_output"] is responses_client.response


def test_openai_responses_prepare_builds_vision_request_shape():
    engine = make_engine()
    image_url = "https://example.com/cat.jpg"
    argument = Argument(
        args=(f"<<vision:{image_url}:>> What is in this image?",),
        signature_kwargs={"max_output_tokens": 16},
        decorator_kwargs={"prompt": "Answer the image question.", "examples": []},
    )
    argument.prop.instance = SimpleNamespace(global_context=("", ""), _kwargs={})
    argument.prop.processed_input = f"<<vision:{image_url}:>> What is in this image?"

    engine.prepare(argument)
    request = engine.build_request(argument)
    user_message = request.body()["input"][1]

    assert user_message["role"] == "user"
    assert user_message["content"] == [
        {"type": "input_image", "image_url": image_url},
        {"type": "input_text", "text": " What is in this image?"},
    ]
    assert "<<vision:" not in request.body()["input"][0]["content"]
    assert "<<vision:" not in user_message["content"][1]["text"]


def test_openai_responses_parse_function_call_without_text_output():
    engine = make_engine()
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                name="get_weather",
                arguments='{"city": "Berlin"}',
                call_id="call_1",
            )
        ],
        output_text="",
    )

    output, metadata = engine.parse_response(response)

    assert output == [""]
    assert metadata["function_call"] == {
        "name": "get_weather",
        "arguments": {"city": "Berlin"},
        "call_id": "call_1",
    }


@pytest.mark.engine_live
def test_openai_responses_live_request_interface(engine_api_mode):
    if engine_api_mode != "live":
        pytest.skip("use --engine-api=live to run live engine API requests")

    model = SYMAI_CONFIG.get("NEUROSYMBOLIC_ENGINE_MODEL", "")
    api_key = SYMAI_CONFIG.get("NEUROSYMBOLIC_ENGINE_API_KEY", "")
    if not model.startswith("openai:"):
        pytest.skip("symai.config.json NEUROSYMBOLIC_ENGINE_MODEL is not an OpenAI model")
    if not api_key:
        pytest.skip("symai.config.json NEUROSYMBOLIC_ENGINE_API_KEY is required")

    engine = OpenAIResponsesEngine(api_key=api_key, model=model)
    argument = make_prepared_argument(
        kwargs={"max_output_tokens": 32},
        messages=[{"role": "user", "content": "Reply with only: ok"}],
    )
    request = engine.build_request(argument)
    response = engine.call_request(request)
    output, metadata = engine.parse_response(response)

    assert request.operation == "responses.create"
    assert output == [] or isinstance(output[0], str)
    assert "raw_output" in metadata


class FakeResponsesClient:
    def __init__(self):
        self.kwargs = None
        self.response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(text="2")],
                )
            ],
            output_text="",
        )

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response
