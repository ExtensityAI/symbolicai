from types import SimpleNamespace

import httpx

from symai.backend.engines.embedding.engine_openai import EmbeddingEngine
from symai.backend.integrations.openai.embeddings import EmbeddingResponse

DUMMY_KEY = "sk-test-not-a-real-key"


def test_embedding_engine_executes_through_standalone_openai_client():
    def handler(request: httpx.Request):
        return httpx.Response(
            200,
            headers={"x-request-id": "request-id"},
            json={
                "object": "list",
                "data": [{"object": "embedding", "embedding": [3.0, 4.0, 0.0], "index": 0}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
            request=request,
        )

    argument = SimpleNamespace(
        args=(),
        kwargs={"new_dim": 2},
        prop=SimpleNamespace(prepared_input=["hello"]),
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        engine = EmbeddingEngine(
            api_key=DUMMY_KEY,
            model="text-embedding-3-small",
            http_client=http_client,
        )
        output, metadata = engine.forward(argument)

    assert output == [[[0.6, 0.8]]]
    assert isinstance(metadata["raw_output"], EmbeddingResponse)
    assert metadata["response"].metadata.request_id == "request-id"
