from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from symai.backend.engines.language_model.deepseek import LanguageModelEngine
from symai.clients.deepseek.chat import ChatCompletion
from symai.clients.deepseek.client import Client

DUMMY_KEY = "sk-test-not-a-real-key"


def _argument(**kwargs):
    return SimpleNamespace(
        kwargs=kwargs,
        prop=SimpleNamespace(
            prepared_input=[{"role": "user", "content": "hello"}],
        ),
    )


def _engine(http_client):
    return LanguageModelEngine(
        client=Client(api_key=DUMMY_KEY, http_client=http_client),
        model="deepseek-v4-flash",
    )


def test_engine_executes_through_standalone_deepseek_client():
    def handler(request: httpx.Request):
        return httpx.Response(
            200,
            headers={"x-request-id": "request-id"},
            json={
                "id": "response-id",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "answer",
                            "reasoning_content": "thought",
                        },
                    }
                ],
                "created": 1,
                "model": "deepseek-v4-flash",
                "object": "chat.completion",
                "usage": {
                    "completion_tokens": 2,
                    "prompt_tokens": 1,
                    "total_tokens": 3,
                },
            },
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        engine = _engine(http_client)
        output, metadata = engine.forward(_argument(max_tokens=16))

    assert output == ["answer"]
    assert metadata["thinking"] == "thought"
    assert isinstance(metadata["raw_output"], ChatCompletion)
    assert metadata["response"].metadata.request_id == "request-id"


@pytest.mark.parametrize("unsupported", [{"stream": True}, {"tools": []}])
def test_engine_rejects_features_outside_provider_client_contract(unsupported):
    with httpx.Client() as http_client:
        engine = _engine(http_client)
        with pytest.raises((ValidationError, ValueError)):
            engine.build_request(_argument(**unsupported))
