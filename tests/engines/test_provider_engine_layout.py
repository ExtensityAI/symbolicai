from types import SimpleNamespace

import httpx
import pytest

from symai import EngineRepository
from symai.backend.engines.provider import create_provider_engine
from symai.backend.settings import SYMAI_CONFIG
from symai.components import DynamicEngine


@pytest.fixture(autouse=True)
def close_provider_engines():
    EngineRepository.close()
    yield
    EngineRepository.close()


def _argument():
    return SimpleNamespace(
        kwargs={},
        prop=SimpleNamespace(prepared_input=[{"role": "user", "content": "hello"}]),
    )


@pytest.mark.parametrize(
    ("model", "module"),
    [
        ("cerebras:gpt-oss-120b", "symai.backend.engines.language_model.cerebras"),
        ("deepseek-v4-flash", "symai.backend.engines.language_model.deepseek"),
        ("openai:gpt-5.4", "symai.backend.engines.language_model.openai"),
    ],
)
def test_language_model_provider_engines_use_capability_first_modules(
    monkeypatch,
    model,
    module,
):
    repository = EngineRepository()
    monkeypatch.setitem(SYMAI_CONFIG, "NEUROSYMBOLIC_ENGINE_MODEL", model)
    monkeypatch.setitem(SYMAI_CONFIG, "NEUROSYMBOLIC_ENGINE_API_KEY", "test-key")

    engine = repository.get("neurosymbolic")

    assert type(engine).__module__ == module
    assert type(engine).__name__ == "LanguageModelEngine"


def test_openai_embedding_engine_is_composed_from_provider_client(monkeypatch):
    repository = EngineRepository()
    monkeypatch.setitem(SYMAI_CONFIG, "EMBEDDING_ENGINE_MODEL", "text-embedding-3-small")
    monkeypatch.setitem(SYMAI_CONFIG, "EMBEDDING_ENGINE_API_KEY", "test-key")

    engine = repository.get("embedding")

    assert type(engine).__module__ == "symai.backend.engines.embedding.openai"
    assert type(engine).__name__ == "EmbeddingEngine"


def test_dynamic_provider_engine_owns_http_client():
    dynamic_engine = DynamicEngine("deepseek-v4-flash", "test-key")
    with dynamic_engine as engine:
        assert type(engine).__module__ == "symai.backend.engines.language_model.deepseek"

    argument = _argument()
    with pytest.raises(RuntimeError, match="client has been closed"):
        engine.forward(argument)


def test_repository_releases_provider_client_when_engine_is_overridden(monkeypatch):
    repository = EngineRepository()
    monkeypatch.setitem(
        SYMAI_CONFIG,
        "NEUROSYMBOLIC_ENGINE_MODEL",
        "deepseek-v4-flash",
    )
    monkeypatch.setitem(SYMAI_CONFIG, "NEUROSYMBOLIC_ENGINE_API_KEY", "test-key")
    engine = repository.get("neurosymbolic")

    with httpx.Client() as replacement_http_client:
        replacement = create_provider_engine(
            capability="language_model",
            model="deepseek-v4-flash",
            api_key="replacement-key",
            http_client=replacement_http_client,
        )
        try:
            repository.register(
                "neurosymbolic",
                replacement,
                allow_engine_override=True,
            )
            with pytest.raises(RuntimeError, match="client has been closed"):
                engine.forward(_argument())
            assert repository.get("neurosymbolic") is replacement
        finally:
            repository._engines.pop("neurosymbolic", None)


def test_repository_same_engine_registration_preserves_owned_client(monkeypatch):
    repository = EngineRepository()
    monkeypatch.setitem(
        SYMAI_CONFIG,
        "NEUROSYMBOLIC_ENGINE_MODEL",
        "deepseek-v4-flash",
    )
    monkeypatch.setitem(SYMAI_CONFIG, "NEUROSYMBOLIC_ENGINE_API_KEY", "test-key")
    engine = repository.get("neurosymbolic")
    http_client = engine.client._http_client
    repository.register("neurosymbolic", engine, allow_engine_override=True)
    assert repository.get("neurosymbolic") is engine
    assert http_client.is_closed is False
