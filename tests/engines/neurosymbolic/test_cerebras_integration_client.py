from types import SimpleNamespace

import httpx
import pytest

from symai.backend.engines.neurosymbolic.engine_cerebras import CerebrasEngine
from symai.backend.integrations.cerebras.chat import ChatResponse

DUMMY_KEY = "sk-test-not-a-real-key"


def _argument(**kwargs):
    return SimpleNamespace(
        kwargs=kwargs,
        prop=SimpleNamespace(
            prepared_input=[{"role": "user", "content": "hello"}],
        ),
    )


def test_engine_executes_through_standalone_cerebras_client():
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
                            "reasoning": "thought",
                        },
                    }
                ],
                "model": "gpt-oss-120b",
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
        engine = CerebrasEngine(
            api_key=DUMMY_KEY,
            model="cerebras:gpt-oss-120b",
            http_client=http_client,
        )
        output, metadata = engine.forward(_argument(max_completion_tokens=16))

    assert output == ["answer"]
    assert metadata["thinking"] == "thought"
    assert isinstance(metadata["raw_output"], ChatResponse)
    assert metadata["response"].metadata.request_id == "request-id"


@pytest.mark.parametrize("unsupported", [{"stream": True}, {"tools": []}])
def test_engine_rejects_features_outside_provider_client_contract(unsupported):
    with httpx.Client() as http_client:
        engine = CerebrasEngine(
            api_key=DUMMY_KEY,
            model="cerebras:gpt-oss-120b",
            http_client=http_client,
        )
        with pytest.raises(ValueError, match="does not support"):
            engine.build_request(_argument(**unsupported))
