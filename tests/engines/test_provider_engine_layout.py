import pytest

from symai import EngineRepository
from symai.backend.settings import SYMAI_CONFIG


@pytest.mark.parametrize(
    ("model", "module"),
    [
        ("cerebras:gpt-oss-120b", "symai.backend.integrations.cerebras.engines.neurosymbolic"),
        ("deepseek-v4-flash", "symai.backend.integrations.deepseek.engines.neurosymbolic"),
        ("openai:gpt-5.4", "symai.backend.integrations.openai.engines.neurosymbolic"),
    ],
)
def test_neurosymbolic_provider_engines_are_discovered_from_integrations(
    monkeypatch,
    model,
    module,
):
    repository = EngineRepository()
    monkeypatch.delitem(repository._engines, "neurosymbolic", raising=False)
    monkeypatch.setitem(SYMAI_CONFIG, "NEUROSYMBOLIC_ENGINE_MODEL", model)
    monkeypatch.setitem(SYMAI_CONFIG, "NEUROSYMBOLIC_ENGINE_API_KEY", "test-key")

    engine = repository.get("neurosymbolic")

    assert type(engine).__module__ == module


def test_openai_embedding_engine_is_discovered_from_integrations(monkeypatch):
    repository = EngineRepository()
    monkeypatch.delitem(repository._engines, "embedding", raising=False)
    monkeypatch.setitem(SYMAI_CONFIG, "EMBEDDING_ENGINE_MODEL", "text-embedding-3-small")
    monkeypatch.setitem(SYMAI_CONFIG, "EMBEDDING_ENGINE_API_KEY", "test-key")

    engine = repository.get("embedding")

    assert type(engine).__module__ == "symai.backend.integrations.openai.engines.embedding"
