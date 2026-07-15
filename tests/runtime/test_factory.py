from collections.abc import Callable
from typing import cast

import httpx
import pytest
from pydantic import SecretStr

import symai.runtime.factory as factory_module
from symai.backend.engines.embedding.openai import EmbeddingEngine
from symai.backend.engines.language_model import cerebras, deepseek, openai
from symai.runtime.errors import UnsupportedCapabilityError, UnsupportedModelError
from symai.runtime.models import (
    Provider,
    ProviderEngineConfig,
    RuntimeConfig,
    TransportConfig,
)
from symai.runtime.runtime import Runtime


def _engine_config(
    provider: Provider,
    model: str,
    *,
    timeout: float = 600.0,
) -> ProviderEngineConfig:
    return ProviderEngineConfig(
        provider=provider,
        model=model,
        api_key=SecretStr("test-key"),
        transport=TransportConfig(request_timeout=timeout),
    )


@pytest.mark.parametrize(
    ("field", "provider", "model", "engine_type"),
    [
        ("language_model", Provider.OPENAI, "gpt-5.4", openai.LanguageModelEngine),
        (
            "language_model",
            Provider.CEREBRAS,
            "gpt-oss-120b",
            cerebras.LanguageModelEngine,
        ),
        (
            "language_model",
            Provider.DEEPSEEK,
            "deepseek-v4-flash",
            deepseek.LanguageModelEngine,
        ),
        ("embedding", Provider.OPENAI, "text-embedding-3-small", EmbeddingEngine),
    ],
)
def test_create_runtime_supports_every_registered_provider_capability(
    field: factory_module._Capability,
    provider: Provider,
    model: str,
    engine_type: type[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[factory_module._ProviderEngine] = []
    registered = factory_module._FACTORIES[(provider, field)]

    def capture_engine(
        model: str,
        api_key: SecretStr,
        client: httpx.Client,
    ) -> factory_module._ProviderEngine:
        assert isinstance(api_key, SecretStr)
        engine = registered.create(model, api_key, client)
        constructed.append(engine)
        return engine

    factories = dict(factory_module._FACTORIES)
    factories[(provider, field)] = factory_module._EngineFactory(
        models=registered.models,
        create=capture_engine,
    )
    monkeypatch.setattr(factory_module, "_FACTORIES", factories)

    runtime = factory_module.create_runtime(
        RuntimeConfig(**{field: _engine_config(provider, model)})
    )
    try:
        assert isinstance(runtime, Runtime)
        assert len(constructed) == 1
        assert isinstance(constructed[0], engine_type)
    finally:
        runtime.close()


def test_factory_registry_is_the_complete_four_entry_provider_matrix() -> None:
    assert set(factory_module._FACTORIES) == {
        (Provider.OPENAI, "language_model"),
        (Provider.OPENAI, "embedding"),
        (Provider.CEREBRAS, "language_model"),
        (Provider.DEEPSEEK, "language_model"),
    }
    assert all(registered.models for registered in factory_module._FACTORIES.values())


@pytest.mark.parametrize(
    ("config", "error", "message"),
    [
        (
            RuntimeConfig(embedding=_engine_config(Provider.DEEPSEEK, "deepseek-v4-flash")),
            UnsupportedCapabilityError,
            "does not support embedding",
        ),
        (
            RuntimeConfig(language_model=_engine_config(Provider.DEEPSEEK, "future-model")),
            UnsupportedModelError,
            "Unsupported deepseek language_model model",
        ),
    ],
)
def test_invalid_provider_selection_fails_before_transport_allocation(
    config: RuntimeConfig,
    error: type[Exception],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        factory_module,
        "_create_http_client",
        lambda _transport: pytest.fail("transport must not be allocated"),
    )

    with pytest.raises(error, match=message):
        factory_module.create_runtime(config)


def test_every_configuration_is_resolved_before_any_transport_is_allocated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RuntimeConfig(
        language_model=_engine_config(Provider.OPENAI, "gpt-5.4"),
        embedding=_engine_config(Provider.DEEPSEEK, "deepseek-v4-flash"),
    )
    monkeypatch.setattr(
        factory_module,
        "_create_http_client",
        lambda _transport: pytest.fail("transport must not be allocated"),
    )

    with pytest.raises(UnsupportedCapabilityError, match="does not support embedding"):
        factory_module.create_runtime(config)


def test_each_engine_uses_its_own_transport_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[TransportConfig] = []
    clients: list[httpx.Client] = []

    def create_http_client(transport: TransportConfig) -> httpx.Client:
        observed.append(transport)
        client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
        clients.append(client)
        return client

    monkeypatch.setattr(factory_module, "_create_http_client", create_http_client)
    language = _engine_config(Provider.OPENAI, "gpt-5.4", timeout=31.0)
    embedding = _engine_config(Provider.OPENAI, "text-embedding-3-small", timeout=47.0)

    runtime = factory_module.create_runtime(
        RuntimeConfig(language_model=language, embedding=embedding)
    )
    try:
        assert observed == [language.transport, embedding.transport]
        assert all(not client.is_closed for client in clients)
    finally:
        runtime.close()

    assert all(client.is_closed for client in clients)


def test_http_client_uses_finite_timeout_and_bounded_connect_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Transport:
        def __init__(self, *, retries: int) -> None:
            captured["retries"] = retries

    class Client:
        def __init__(self, *, timeout: httpx.Timeout, transport: object) -> None:
            captured["timeout"] = timeout
            captured["transport"] = transport
            captured["client"] = self

    monkeypatch.setattr(factory_module.httpx, "HTTPTransport", Transport)
    monkeypatch.setattr(factory_module.httpx, "Client", Client)

    result = factory_module._create_http_client(
        TransportConfig(request_timeout=30.0, connect_timeout=2.0, connect_retries=3)
    )

    assert result is captured["client"]
    timeout = cast("httpx.Timeout", captured["timeout"])
    assert timeout.read == 30.0
    assert timeout.connect == 2.0
    assert captured["retries"] == 3


def test_http_client_construction_failure_closes_allocated_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_attempts = 0
    failure = RuntimeError("client construction failed")

    class Transport:
        def __init__(self, *, retries: int) -> None:
            assert retries == 0

        def close(self) -> None:
            nonlocal close_attempts
            close_attempts += 1

    def fail_client(**_kwargs: object) -> httpx.Client:
        raise failure

    monkeypatch.setattr(factory_module.httpx, "HTTPTransport", Transport)
    monkeypatch.setattr(factory_module.httpx, "Client", fail_client)

    with pytest.raises(RuntimeError, match="client construction failed") as caught:
        factory_module._create_http_client(TransportConfig())

    assert caught.value is failure
    assert close_attempts == 1


class _RecordingClient:
    def __init__(
        self,
        name: str,
        attempts: list[str],
        failure: BaseException | None = None,
    ) -> None:
        self.name = name
        self.attempts = attempts
        self.failure = failure
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        self.attempts.append(self.name)
        if self.failure is not None:
            raise self.failure


def _construction_factories(
    language_create: Callable[[str, str, httpx.Client], factory_module._ProviderEngine],
    embedding_create: Callable[[str, str, httpx.Client], factory_module._ProviderEngine],
) -> dict[tuple[Provider, factory_module._Capability], factory_module._EngineFactory]:
    return {
        (Provider.OPENAI, "language_model"): factory_module._EngineFactory(
            models={"language": object()},
            create=language_create,
        ),
        (Provider.OPENAI, "embedding"): factory_module._EngineFactory(
            models={"embedding": object()},
            create=embedding_create,
        ),
    }


def _two_engine_config() -> RuntimeConfig:
    return RuntimeConfig(
        language_model=_engine_config(Provider.OPENAI, "language", timeout=1.0),
        embedding=_engine_config(Provider.OPENAI, "embedding", timeout=2.0),
    )


def test_later_engine_construction_failure_closes_current_and_earlier_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []
    clients = {
        1.0: _RecordingClient("language", attempts),
        2.0: _RecordingClient("embedding", attempts),
    }
    failure = RuntimeError("embedding construction failed")

    def create_http_client(transport: TransportConfig) -> httpx.Client:
        return cast("httpx.Client", clients[transport.request_timeout])

    def fail_embedding(
        _model: str,
        _key: str,
        _client: httpx.Client,
    ) -> factory_module._ProviderEngine:
        raise failure

    monkeypatch.setattr(factory_module, "_create_http_client", create_http_client)
    monkeypatch.setattr(
        factory_module,
        "_FACTORIES",
        _construction_factories(
            lambda *_: cast("factory_module._ProviderEngine", object()),
            fail_embedding,
        ),
    )

    with pytest.raises(RuntimeError, match="embedding construction failed") as caught:
        factory_module.create_runtime(_two_engine_config())

    assert caught.value is failure
    assert attempts == ["embedding", "language"]
    assert [client.close_count for client in clients.values()] == [1, 1]


def test_all_cleanup_failures_become_notes_on_original_construction_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []
    clients = {
        1.0: _RecordingClient("language", attempts, RuntimeError("language close failed")),
        2.0: _RecordingClient("embedding", attempts, KeyboardInterrupt("embedding close failed")),
    }
    failure = LookupError("runtime construction failed")

    def create_http_client(transport: TransportConfig) -> httpx.Client:
        return cast("httpx.Client", clients[transport.request_timeout])

    def fail_runtime(**_kwargs: object) -> Runtime:
        raise failure

    monkeypatch.setattr(factory_module, "_create_http_client", create_http_client)
    monkeypatch.setattr(
        factory_module,
        "_FACTORIES",
        _construction_factories(
            lambda *_: cast("factory_module._ProviderEngine", object()),
            lambda *_: cast("factory_module._ProviderEngine", object()),
        ),
    )
    monkeypatch.setattr(factory_module, "Runtime", fail_runtime)

    with pytest.raises(LookupError, match="runtime construction failed") as caught:
        factory_module.create_runtime(_two_engine_config())

    assert caught.value is failure
    assert attempts == ["embedding", "language"]
    assert len(caught.value.__notes__) == 2
    assert "embedding close failed" in caught.value.__notes__[0]
    assert "language close failed" in caught.value.__notes__[1]


def test_success_transfers_each_resource_to_runtime_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []
    clients = {
        1.0: _RecordingClient("language", attempts),
        2.0: _RecordingClient("embedding", attempts),
    }

    def create_http_client(transport: TransportConfig) -> httpx.Client:
        return cast("httpx.Client", clients[transport.request_timeout])

    monkeypatch.setattr(factory_module, "_create_http_client", create_http_client)
    monkeypatch.setattr(
        factory_module,
        "_FACTORIES",
        _construction_factories(
            lambda *_: cast("factory_module._ProviderEngine", object()),
            lambda *_: cast("factory_module._ProviderEngine", object()),
        ),
    )

    runtime = factory_module.create_runtime(_two_engine_config())
    assert attempts == []

    runtime.close()
    runtime.close()

    assert attempts == ["language", "embedding"]
    assert [client.close_count for client in clients.values()] == [1, 1]
