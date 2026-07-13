from types import SimpleNamespace

import pytest

from symai import EngineRepository
from symai.backend import provider_runtime
from symai.backend.provider_runtime import create_provider_engine_handle
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
    ("model", "provider"),
    [
        ("cerebras:gpt-oss-120b", "cerebras"),
        ("deepseek:deepseek-v4-flash", "deepseek"),
        ("openai:gpt-5.4", "openai"),
    ],
)
def test_repository_composes_configured_language_model_provider(
    monkeypatch,
    model,
    provider,
):
    monkeypatch.setitem(SYMAI_CONFIG, "NEUROSYMBOLIC_ENGINE_MODEL", model)
    monkeypatch.setitem(SYMAI_CONFIG, "NEUROSYMBOLIC_ENGINE_API_KEY", "test-key")

    engine = EngineRepository.get("neurosymbolic")

    assert engine.provider == provider
    assert engine.capability == "language_model"
    assert engine.model == model.partition(":")[2]


def test_repository_composes_configured_embedding_provider(monkeypatch):
    monkeypatch.setitem(
        SYMAI_CONFIG,
        "EMBEDDING_ENGINE_MODEL",
        "openai:text-embedding-3-small",
    )
    monkeypatch.setitem(SYMAI_CONFIG, "EMBEDDING_ENGINE_API_KEY", "test-key")

    engine = EngineRepository.get("embedding")

    assert engine.provider == "openai"
    assert engine.capability == "embedding"
    assert engine.model == "text-embedding-3-small"


def test_dynamic_provider_engine_releases_its_lease_on_exit():
    dynamic_engine = DynamicEngine("deepseek:deepseek-v4-flash", "test-key")

    with dynamic_engine as engine:
        assert engine.provider == "deepseek"

    with pytest.raises(RuntimeError, match="client has been closed"):
        engine.forward(_argument())


def test_dynamic_provider_engine_uses_explicit_transport_options(monkeypatch):
    captured = {}
    create_engine_handle = provider_runtime.create_provider_engine_handle

    def capture_options(**kwargs):
        captured["options"] = kwargs["options"]
        return create_engine_handle(**kwargs)

    monkeypatch.setattr(provider_runtime, "create_provider_engine_handle", capture_options)
    dynamic_engine = DynamicEngine(
        "deepseek:deepseek-v4-flash",
        "test-key",
        request_timeout=42.0,
        connect_timeout=3.0,
        connect_retries=2,
    )

    with dynamic_engine:
        pass

    assert captured["options"] == provider_runtime.ProviderRuntimeOptions(
        request_timeout=42.0,
        connect_timeout=3.0,
        connect_retries=2,
    )


def test_repository_replacement_closes_previous_provider_lease(monkeypatch):
    monkeypatch.setitem(
        SYMAI_CONFIG,
        "NEUROSYMBOLIC_ENGINE_MODEL",
        "deepseek:deepseek-v4-flash",
    )
    monkeypatch.setitem(SYMAI_CONFIG, "NEUROSYMBOLIC_ENGINE_API_KEY", "test-key")
    engine = EngineRepository.get("neurosymbolic")
    replacement = create_provider_engine_handle(
        capability="language_model",
        model="deepseek:deepseek-v4-flash",
        api_key="replacement-key",
    )
    assert replacement is not None

    EngineRepository.register(
        "neurosymbolic",
        replacement,
        allow_engine_override=True,
    )

    with pytest.raises(RuntimeError, match="client has been closed"):
        engine.forward(_argument())
    assert EngineRepository.get("neurosymbolic") is replacement.engine
