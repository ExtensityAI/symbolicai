import inspect

import httpx
import pytest

from symai.backend.engines.embedding import openai as openai_embedding
from symai.backend.engines.language_model import cerebras, deepseek, openai
from symai.backend.engines.provider import create_provider_http_client
from symai.clients.cerebras.client import Client as CerebrasClient
from symai.clients.deepseek.client import Client as DeepSeekClient
from symai.clients.openai.client import Client as OpenAIClient


@pytest.mark.parametrize(
    ("engine_type", "client_type", "model", "provider"),
    [
        (openai.LanguageModelEngine, OpenAIClient, "gpt-5.4", "openai"),
        (cerebras.LanguageModelEngine, CerebrasClient, "gpt-oss-120b", "cerebras"),
        (deepseek.LanguageModelEngine, DeepSeekClient, "deepseek-v4-flash", "deepseek"),
    ],
)
def test_language_model_engines_require_typed_clients(
    engine_type,
    client_type,
    model,
    provider,
):
    assert tuple(inspect.signature(engine_type).parameters) == ("client", "model")

    with httpx.Client() as http_client:
        client = client_type(api_key="test-key", http_client=http_client)
        engine = engine_type(client=client, model=model)

    assert engine.client is client
    assert engine.model == model
    assert engine.provider == provider
    assert engine.capability == "language_model"
    assert engine.id() == "neurosymbolic"


def test_embedding_engine_requires_typed_client():
    assert tuple(inspect.signature(openai_embedding.EmbeddingEngine).parameters) == (
        "client",
        "model",
    )

    with httpx.Client() as http_client:
        client = OpenAIClient(api_key="test-key", http_client=http_client)
        engine = openai_embedding.EmbeddingEngine(
            client=client,
            model="text-embedding-3-small",
        )

    assert engine.client is client
    assert engine.model == "text-embedding-3-small"
    assert engine.provider == "openai"
    assert engine.capability == "embedding"
    assert engine.id() == "embedding"


@pytest.mark.parametrize(
    ("engine_type", "client_type", "model"),
    [
        (openai.LanguageModelEngine, OpenAIClient, "unknown-model"),
        (cerebras.LanguageModelEngine, CerebrasClient, "unknown-model"),
        (deepseek.LanguageModelEngine, DeepSeekClient, "unknown-model"),
        (openai_embedding.EmbeddingEngine, OpenAIClient, "unknown-model"),
    ],
)
def test_provider_engines_reject_unknown_models(engine_type, client_type, model):
    with httpx.Client() as http_client:
        client = client_type(api_key="test-key", http_client=http_client)
        with pytest.raises(ValueError, match="Unsupported model"):
            engine_type(client=client, model=model)


@pytest.mark.parametrize(
    ("capability", "model", "expected_timeout"),
    [
        ("embedding", "text-embedding-3-small", 600.0),
        ("language_model", "openai:gpt-5.4", 600.0),
        ("language_model", "cerebras:gpt-oss-120b", None),
        ("language_model", "deepseek-v4-flash", None),
    ],
)
def test_provider_composition_preserves_request_timeout_defaults(
    capability,
    model,
    expected_timeout,
):
    with create_provider_http_client(capability=capability, model=model) as client:
        assert client.timeout.read == expected_timeout
