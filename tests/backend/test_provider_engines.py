import httpx
import pytest

from symai.backend import provider_engines
from symai.backend.engines.embedding.openai import EmbeddingEngine
from symai.backend.engines.language_model import cerebras, deepseek, openai


@pytest.mark.parametrize(
    ("capability", "model", "engine_type"),
    [
        (
            provider_engines.Capability.LANGUAGE_MODEL,
            "openai:gpt-5.4",
            openai.LanguageModelEngine,
        ),
        (
            provider_engines.Capability.LANGUAGE_MODEL,
            "cerebras:gpt-oss-120b",
            cerebras.LanguageModelEngine,
        ),
        (
            provider_engines.Capability.LANGUAGE_MODEL,
            "deepseek:deepseek-v4-flash",
            deepseek.LanguageModelEngine,
        ),
        (
            provider_engines.Capability.EMBEDDING,
            "openai:text-embedding-3-small",
            EmbeddingEngine,
        ),
    ],
)
def test_create_provider_engine_handle_uses_registered_factory(
    capability, model, engine_type
):
    handle = provider_engines.create_provider_engine_handle(
        capability=capability,
        model=model,
        api_key="test-key",
    )

    assert handle is not None
    assert isinstance(handle.engine, engine_type)
    assert handle.engine.model == model.partition(":")[2]
    handle.close()


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (":gpt-5.4", "provider must not be empty"),
        ("openai:", "model must not be empty"),
        ("opneai:gpt-5.4", "Unknown provider: opneai"),
        ("gpt-5.4", "must include a provider prefix"),
    ],
)
def test_invalid_provider_selection_fails_before_transport(monkeypatch, model, message):
    allocated = False

    def create_http_client(_options):
        nonlocal allocated
        allocated = True
        return httpx.Client()

    monkeypatch.setattr(
        provider_engines, "create_provider_http_client", create_http_client
    )

    with pytest.raises(ValueError, match=message):
        provider_engines.create_provider_engine_handle(
            capability=provider_engines.Capability.LANGUAGE_MODEL,
            model=model,
            api_key="test-key",
        )

    assert allocated is False


def test_unqualified_legacy_model_returns_none():
    assert (
        provider_engines.create_provider_engine_handle(
            capability=provider_engines.Capability.LANGUAGE_MODEL,
            model="gemini-3.1-pro",
            api_key="test-key",
        )
        is None
    )


def test_unsupported_provider_capability_fails_before_transport(monkeypatch):
    monkeypatch.setattr(
        provider_engines,
        "create_provider_http_client",
        lambda _options: pytest.fail("transport must not be allocated"),
    )

    with pytest.raises(ValueError, match="does not support capability"):
        provider_engines.create_provider_engine_handle(
            capability=provider_engines.Capability.EMBEDDING,
            model="deepseek:deepseek-v4-flash",
            api_key="test-key",
        )


@pytest.mark.parametrize(
    ("model", "api_key", "message"),
    [
        ("deepseek:future-model", "test-key", "Unsupported deepseek language_model model"),
        ("deepseek:deepseek-v4-flash", "", "api_key must not be empty"),
    ],
)
def test_invalid_local_configuration_fails_before_transport(
    monkeypatch, model, api_key, message
):
    monkeypatch.setattr(
        provider_engines,
        "create_provider_http_client",
        lambda _options: pytest.fail("transport must not be allocated"),
    )

    with pytest.raises(ValueError, match=message):
        provider_engines.create_provider_engine_handle(
            capability=provider_engines.Capability.LANGUAGE_MODEL,
            model=model,
            api_key=api_key,
        )


def test_factory_registry_is_typed_complete_and_cataloged():
    assert set(provider_engines._FACTORIES) == {
        (
            provider_engines.Provider.OPENAI,
            provider_engines.Capability.LANGUAGE_MODEL,
        ),
        (provider_engines.Provider.OPENAI, provider_engines.Capability.EMBEDDING),
        (
            provider_engines.Provider.CEREBRAS,
            provider_engines.Capability.LANGUAGE_MODEL,
        ),
        (
            provider_engines.Provider.DEEPSEEK,
            provider_engines.Capability.LANGUAGE_MODEL,
        ),
    }
    assert all(factory.models for factory in provider_engines._FACTORIES.values())


def test_engine_construction_failure_closes_allocated_transport(monkeypatch):
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200))
    )
    monkeypatch.setattr(
        provider_engines,
        "create_provider_http_client",
        lambda _options: http_client,
    )

    def fail_construction(**_kwargs):
        raise RuntimeError("construction failed")

    monkeypatch.setattr(
        provider_engines.deepseek_engine,
        "LanguageModelEngine",
        fail_construction,
    )

    with pytest.raises(RuntimeError, match="construction failed"):
        provider_engines.create_provider_engine_handle(
            capability=provider_engines.Capability.LANGUAGE_MODEL,
            model="deepseek:deepseek-v4-flash",
            api_key="test-key",
        )

    assert http_client.is_closed is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_timeout", 0),
        ("request_timeout", float("inf")),
        ("request_timeout", float("nan")),
        ("connect_timeout", 0),
        ("connect_timeout", float("inf")),
        ("connect_timeout", float("nan")),
        ("connect_retries", -1),
    ],
)
def test_provider_transport_options_reject_invalid_transport_bounds(field, value):
    with pytest.raises(ValueError, match=field):
        provider_engines.ProviderTransportOptions(**{field: value})


def test_create_provider_engine_handle_owns_transport_with_explicit_options(monkeypatch):
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200))
    )
    captured = {}

    def create_http_client(options):
        captured["options"] = options
        return http_client

    monkeypatch.setattr(
        provider_engines, "create_provider_http_client", create_http_client
    )
    options = provider_engines.ProviderTransportOptions(
        request_timeout=30,
        connect_timeout=2,
        connect_retries=1,
    )

    handle = provider_engines.create_provider_engine_handle(
        capability=provider_engines.Capability.LANGUAGE_MODEL,
        model="deepseek:deepseek-v4-flash",
        api_key="test-key",
        options=options,
    )

    assert handle is not None
    assert captured["options"] == options
    assert http_client.is_closed is False

    handle.close()

    assert http_client.is_closed is True
