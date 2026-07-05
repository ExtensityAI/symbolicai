from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from symai.backend.engines.neurosymbolic.engine_deepseekX import (
    DeepSeekXReasoningEngine,
)
from symai.backend.mixin.deepseek import DeepSeekChatCreatePayload
from symai.backend.settings import SYMAI_CONFIG
from symai.core import Argument

DUMMY_KEY = "sk-test-not-a-real-key"


def make_engine(model="deepseek-reasoner"):
    return DeepSeekXReasoningEngine(api_key=DUMMY_KEY, model=model)


def make_prepared_argument(kwargs=None, messages=None):
    if kwargs is None:
        kwargs = {}
    if messages is None:
        messages = [{"role": "user", "content": "hello"}]
    return SimpleNamespace(
        kwargs=kwargs,
        prop=SimpleNamespace(prepared_input=messages),
    )


def test_deepseek_request_body_uses_payload_model_shape():
    engine = make_engine()

    request = engine.build_request(
        make_prepared_argument(kwargs={"thinking": {"type": "disabled"}})
    )

    assert set(request.body()) <= set(DeepSeekChatCreatePayload.model_fields)


def test_deepseek_ignored_request_kwargs_come_from_argument_defaults():
    engine = make_engine()

    assert engine.ignored_request_kwargs() == set(Argument.default_property_values())


def test_deepseek_build_request_forwards_typed_supported_kwargs():
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
    kwargs = {
        "model": "deepseek-v4-flash",
        "thinking": {"type": "disabled"},
        "reasoning_effort": "high",
        "max_tokens": 32,
        "response_format": {"type": "text"},
        "stop": ["END"],
        "stream": False,
        "stream_options": {"include_usage": True},
        "temperature": 0.2,
        "top_p": 0.9,
        "tools": tools,
        "tool_choice": "auto",
        "logprobs": True,
        "top_logprobs": 2,
        "user_id": "user-1",
        "seed": 123,
        "n": 1,
        "logit_bias": {"1": 1},
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "extra_headers": {"X-Test": "1"},
        "extra_query": {"debug": "1"},
        "extra_body": {"custom": "yes"},
        "timeout": 5.0,
    }

    request = engine.build_request(make_prepared_argument(kwargs=kwargs))
    body = request.body()
    request_kwargs = request.kwargs()

    assert body["model"] == "deepseek-v4-flash"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["thinking"] == {"type": "disabled"}
    assert body["reasoning_effort"] == "high"
    assert body["max_tokens"] == 32
    assert body["response_format"] == {"type": "text"}
    assert body["stop"] == ["END"]
    assert body["stream"] is False
    assert body["stream_options"] == {"include_usage": True}
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 2
    assert body["user_id"] == "user-1"
    assert body["seed"] == 123
    assert body["n"] == 1
    assert body["logit_bias"] == {"1": 1}
    assert body["frequency_penalty"] == 0
    assert body["presence_penalty"] == 0

    assert request_kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert request_kwargs["model"] == "deepseek-v4-flash"
    assert request_kwargs["reasoning_effort"] == "high"
    assert request_kwargs["thinking"] == {"type": "disabled"}
    assert request_kwargs["user_id"] == "user-1"
    assert request_kwargs["extra_headers"] == {"X-Test": "1"}
    assert request_kwargs["extra_query"] == {"debug": "1"}
    assert request_kwargs["extra_body"] == {"custom": "yes"}
    assert request_kwargs["timeout"] == 5.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": "32"},
        {"temperature": "0.2"},
        {"stream": "false"},
        {"thinking": {"type": "invalid"}},
        {"response_format": {"type": "xml"}},
        {"stream_options": {"include_usage": "yes"}},
    ],
)
def test_deepseek_build_request_rejects_wrong_provider_kwarg_types(kwargs):
    engine = make_engine()

    with pytest.raises(ValidationError):
        engine.build_request(make_prepared_argument(kwargs=kwargs))


def test_deepseek_build_request_ignores_non_provider_kwargs_for_backward_compatibility():
    engine = make_engine()
    argument = make_prepared_argument(
        kwargs={
            "prompt": "Symai prompt kwarg",
            "examples": [],
            "context": "Symai context kwarg",
            "raw_input": False,
            "return_metadata": True,
            "max_tokens": 16,
        }
    )

    request = engine.build_request(argument)
    body = request.body()

    assert body["max_tokens"] == 16
    assert "prompt" not in body
    assert "examples" not in body
    assert "raw_input" not in body
    assert "return_metadata" not in body


def test_deepseek_strict_request_kwargs_allows_symai_kwargs():
    engine = make_engine()
    argument = make_prepared_argument(
        kwargs={
            "strict_request_kwargs": True,
            "prompt": "Symai prompt kwarg",
            "examples": [],
            "raw_input": False,
            "return_metadata": True,
            "max_tokens": 16,
        }
    )

    request = engine.build_request(argument)

    assert request.body()["max_tokens"] == 16


def test_deepseek_strict_request_kwargs_rejects_unknown_kwargs():
    engine = make_engine()
    argument = make_prepared_argument(kwargs={"strict_request_kwargs": True, "temprature": 0.2})

    with pytest.raises(ValueError, match="Unsupported request kwargs"):
        engine.build_request(argument)


def test_deepseek_prepare_build_call_parse_with_fake_client():
    engine = make_engine()
    argument = Argument(
        args=("What is 1+1?",),
        signature_kwargs={"max_tokens": 16},
        decorator_kwargs={"prompt": "Answer the question.", "examples": []},
    )
    argument.prop.instance = SimpleNamespace(global_context=("", ""), _kwargs={})
    argument.prop.processed_input = "What is 1+1?"
    completion_client = FakeCompletionClient()
    engine.client = completion_client

    engine.prepare(argument)
    output, metadata = engine.forward(argument)

    assert [message["role"] for message in argument.prop.prepared_input] == ["system", "user"]
    assert completion_client.body["model"] == "deepseek-reasoner"
    assert completion_client.body["max_tokens"] == 16
    assert completion_client.body["messages"] == argument.prop.prepared_input
    assert output == ["2"]
    assert metadata["thinking"] == "Add one and one."


@pytest.mark.engine_live
def test_deepseek_live_request_interface(engine_api_mode):
    if engine_api_mode != "live":
        pytest.skip("use --engine-api=live to run live engine API requests")

    model = SYMAI_CONFIG.get("NEUROSYMBOLIC_ENGINE_MODEL", "")
    api_key = SYMAI_CONFIG.get("NEUROSYMBOLIC_ENGINE_API_KEY", "")
    if not model.startswith("deepseek"):
        pytest.skip("symai.config.json NEUROSYMBOLIC_ENGINE_MODEL is not a DeepSeek model")
    if not api_key:
        pytest.skip("symai.config.json NEUROSYMBOLIC_ENGINE_API_KEY is required")

    engine = DeepSeekXReasoningEngine(api_key=api_key, model=model)
    argument = make_prepared_argument(
        kwargs={"max_tokens": 128},
        messages=[{"role": "user", "content": "Reply with only: ok"}],
    )
    request = engine.build_request(argument)
    response = engine.call_request(request)
    output, metadata = engine.parse_response(response)

    assert request.operation == "chat.completions.create"
    assert isinstance(output[0], str)
    assert "thinking" in metadata


class FakeCompletionClient:
    def __init__(self):
        self.body = None
        self.options = None

    def post(self, _path, *, cast_to, body, options):
        assert cast_to is not None
        self.body = body
        self.options = options
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        reasoning_content="Add one and one.",
                        content="2",
                    )
                )
            ]
        )
