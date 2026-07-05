from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from symai.backend.engines.neurosymbolic.engine_groq import GroqEngine
from symai.backend.mixin.groq import SUPPORTED_GROQ_MODELS, GroqChatCreatePayload
from symai.backend.settings import SYMAI_CONFIG
from symai.core import Argument

DUMMY_KEY = "gsk-test-not-a-real-key"


def make_engine(model="groq:openai/gpt-oss-20b"):
    return GroqEngine(api_key=DUMMY_KEY, model=model)


def make_prepared_argument(kwargs=None, messages=None):
    if kwargs is None:
        kwargs = {}
    if messages is None:
        messages = [{"role": "user", "content": "hello"}]
    return SimpleNamespace(
        kwargs=kwargs,
        prop=SimpleNamespace(prepared_input=messages),
    )


def test_groq_request_body_uses_payload_model_shape():
    engine = make_engine()

    request = engine.build_request(make_prepared_argument(kwargs={"max_completion_tokens": 16}))

    assert set(request.body()) <= set(GroqChatCreatePayload.model_fields)


def test_groq_ignored_request_kwargs_come_from_argument_defaults():
    engine = make_engine()

    assert engine.ignored_request_kwargs() == set(Argument.default_property_values())


def test_groq_build_request_forwards_typed_supported_kwargs(caplog):
    engine = make_engine()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    kwargs = {
        "model": "groq:openai/gpt-oss-120b",
        "seed": 123,
        "max_completion_tokens": 64,
        "stop": "",
        "temperature": 0.2,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "reasoning_effort": "medium",
        "service_tier": "on_demand",
        "top_p": 0.9,
        "n": 2,
        "tools": tools,
        "tool_choice": "auto",
        "response_format": {"type": "text"},
        "parallel_tool_calls": False,
        "user": "user-1",
        "extra_headers": {"X-Test": "1"},
        "extra_query": {"debug": "1"},
        "extra_body": {"trace": "yes"},
        "timeout": 5.0,
        "stream": False,
    }

    request = engine.build_request(make_prepared_argument(kwargs=kwargs))
    body = request.body()
    sdk_kwargs = request.kwargs()

    assert body["model"] == "openai/gpt-oss-120b"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["seed"] == 123
    assert body["max_completion_tokens"] == 64
    assert "stop" not in body
    assert body["temperature"] == 0.2
    assert body["frequency_penalty"] == 0
    assert body["presence_penalty"] == 0
    assert body["reasoning_effort"] == "medium"
    assert body["service_tier"] == "on_demand"
    assert body["top_p"] == 0.9
    assert body["n"] == 1
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"
    assert body["response_format"] == {"type": "text"}
    assert body["parallel_tool_calls"] is False
    assert body["user"] == "user-1"
    assert "stream" not in body
    assert "not supported by the Groq API" in caplog.text

    assert sdk_kwargs["extra_headers"] == {"X-Test": "1"}
    assert sdk_kwargs["extra_query"] == {"debug": "1"}
    assert sdk_kwargs["extra_body"] == {"trace": "yes"}
    assert sdk_kwargs["timeout"] == 5.0


def test_groq_max_tokens_alias_maps_to_max_completion_tokens():
    engine = make_engine()

    request = engine.build_request(make_prepared_argument(kwargs={"max_tokens": 32}))

    assert request.body()["max_completion_tokens"] == 32
    assert "max_tokens" not in request.body()


def test_groq_json_mode_removes_tools_and_sets_tool_choice_auto():
    engine = make_engine()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    request = engine.build_request(
        make_prepared_argument(
            kwargs={
                "response_format": {"type": "json_object"},
                "tools": tools,
                "tool_choice": "none",
            }
        )
    )
    body = request.body()

    assert "tools" not in body
    assert body["tool_choice"] == "auto"


def test_groq_build_request_rejects_removed_model_override():
    engine = make_engine()

    with pytest.raises(ValueError, match="Unsupported Groq model"):
        engine.build_request(make_prepared_argument(kwargs={"model": "groq:qwen/qwen3-32b"}))


def test_groq_build_request_rejects_unsupported_reasoning_effort():
    engine = make_engine()

    with pytest.raises(ValueError, match="Unsupported reasoning_effort"):
        engine.build_request(make_prepared_argument(kwargs={"reasoning_effort": "none"}))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_completion_tokens": "64"},
        {"max_tokens": "64"},
        {"temperature": "0.2"},
        {"top_p": 2},
        {"service_tier": "priority"},
        {"response_format": {"type": "xml"}},
    ],
)
def test_groq_build_request_rejects_wrong_provider_kwarg_types(kwargs):
    engine = make_engine()

    with pytest.raises(ValidationError):
        engine.build_request(make_prepared_argument(kwargs=kwargs))


def test_groq_strict_request_kwargs_allows_symai_kwargs():
    engine = make_engine()
    argument = make_prepared_argument(
        kwargs={
            "strict_request_kwargs": True,
            "prompt": "Symai prompt kwarg",
            "examples": [],
            "raw_input": False,
            "return_metadata": True,
            "max_completion_tokens": 16,
        }
    )

    request = engine.build_request(argument)

    assert request.body()["max_completion_tokens"] == 16


def test_groq_strict_request_kwargs_rejects_unknown_kwargs():
    engine = make_engine()
    argument = make_prepared_argument(
        kwargs={"strict_request_kwargs": True, "max_completion_toknes": 16}
    )

    with pytest.raises(ValueError, match="Unsupported request kwargs"):
        engine.build_request(argument)


def test_groq_prepare_build_call_parse_with_fake_client():
    engine = make_engine()
    argument = Argument(
        args=("What is 1+1?",),
        signature_kwargs={"max_completion_tokens": 16},
        decorator_kwargs={"prompt": "Answer the question.", "examples": []},
    )
    argument.prop.instance = SimpleNamespace(global_context=("", ""), _kwargs={})
    argument.prop.processed_input = "What is 1+1?"
    completion_client = FakeCompletionClient()
    engine.client = SimpleNamespace(chat=SimpleNamespace(completions=completion_client))

    engine.prepare(argument)
    output, metadata = engine.forward(argument)

    assert [message["role"] for message in argument.prop.prepared_input] == ["system", "user"]
    assert completion_client.kwargs["model"] == "openai/gpt-oss-20b"
    assert completion_client.kwargs["max_completion_tokens"] == 16
    assert completion_client.kwargs["messages"] == argument.prop.prepared_input
    assert output == ["2"]
    assert metadata["raw_output"] is completion_client.response


def test_groq_parse_response_extracts_thinking_content():
    engine = make_engine()
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="<think>Count.</think>2", tool_calls=None),
                finish_reason="stop",
            )
        ]
    )

    output, metadata = engine.parse_response(response)

    assert output == ["2"]
    assert metadata["thinking"] == "Count."


def test_groq_parse_function_call_without_text_output():
    engine = make_engine()
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="get_weather",
                                arguments='{"city": "Berlin"}',
                            )
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ]
    )

    output, metadata = engine.parse_response(response)

    assert output == [""]
    assert metadata["function_call"] == {
        "name": "get_weather",
        "arguments": {"city": "Berlin"},
    }


@pytest.mark.engine_live
def test_groq_live_request_interface(engine_api_mode):
    if engine_api_mode != "live":
        pytest.skip("use --engine-api=live to run live engine API requests")

    model = SYMAI_CONFIG.get("NEUROSYMBOLIC_ENGINE_MODEL", "")
    api_key = SYMAI_CONFIG.get("NEUROSYMBOLIC_ENGINE_API_KEY", "")
    if model not in SUPPORTED_GROQ_MODELS:
        pytest.skip("symai.config.json NEUROSYMBOLIC_ENGINE_MODEL is not a supported Groq model")
    if not api_key:
        pytest.skip("symai.config.json NEUROSYMBOLIC_ENGINE_API_KEY is required")

    engine = GroqEngine(api_key=api_key, model=model)
    argument = make_prepared_argument(
        kwargs={"max_completion_tokens": 32},
        messages=[{"role": "user", "content": "Reply with only: ok"}],
    )
    request = engine.build_request(argument)
    response = engine.call_request(request)
    output, metadata = engine.parse_response(response)

    assert request.operation == "chat.completions.create"
    assert output == [] or isinstance(output[0], str)
    assert "raw_output" in metadata


class FakeCompletionClient:
    def __init__(self):
        self.kwargs = None
        self.response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="2", tool_calls=None),
                    finish_reason="stop",
                )
            ]
        )

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response
