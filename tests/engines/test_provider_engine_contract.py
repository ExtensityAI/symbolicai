import ast
import inspect
from pathlib import Path

import httpx
import pytest

from symai.backend.base import ENGINE_UNREGISTERED
from symai.backend.engines.embedding import openai as openai_embedding
from symai.backend.engines.language_model import cerebras, deepseek, openai
from symai.backend.provider_runtime import ProviderRuntimeOptions, create_provider_http_client
from symai.clients.cerebras.client import Client as CerebrasClient
from symai.clients.deepseek.client import Client as DeepSeekClient
from symai.clients.openai.client import Client as OpenAIClient


def test_provider_adapters_do_not_import_application_components():
    for module in (openai, cerebras, deepseek, openai_embedding):
        tree = ast.parse(Path(module.__file__).read_text())
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imports.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )

        assert not any(
            imported == "symai.components" or imported.startswith("symai.components.")
            for imported in imports
        ), module.__name__


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
    assert engine.id() == ENGINE_UNREGISTERED


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
    assert engine.id() == ENGINE_UNREGISTERED


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


def test_provider_composition_uses_uniform_finite_timeout_defaults():
    with create_provider_http_client(ProviderRuntimeOptions()) as client:
        assert client.timeout.read == 600.0
        assert client.timeout.connect == 10.0
