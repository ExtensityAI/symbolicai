from collections.abc import Callable
from typing import cast

import httpx
import pytest
from pydantic import SecretStr

import symai.runtime.factory as factory_module
from symai.providers.openai import EmbeddingEngine
from symai.providers import cerebras, deepseek, openai
from symai.runtime.errors import UnsupportedCapabilityError, UnsupportedModelError
from symai.runtime.models import (
    LanguageModelRequest,
    LanguageModelResponse,
    NamedEngineConfig,
    Provider,
    RuntimeConfig,
    TextContent,
    TransportConfig,
    UserMessage,
)
from symai.runtime.runtime import Runtime


def _engine_config(
    name: str,
    provider: Provider,
    model: str,
    *,
    api_key: str = "test-key",
    timeout: float = 600.0,
) -> NamedEngineConfig:
    return NamedEngineConfig(
        name=name,
        provider=provider,
        model=model,
        api_key=SecretStr(api_key),
        transport=TransportConfig(request_timeout=timeout),
    )


@pytest.mark.parametrize(
    ("capability", "provider", "model", "engine_type"),
    [
        (
            "language_model",
            Provider.OPENAI,
            "gpt-5.4",
            openai.ResponsesEngine,
        ),
        (
            "language_model",
            Provider.CEREBRAS,
            "gpt-oss-120b",
            cerebras.ChatCompletionsEngine,
        ),
        (
            "language_model",
            Provider.DEEPSEEK,
            "deepseek-v4-flash",
            deepseek.ChatCompletionsEngine,
        ),
        (
            "embedding",
            Provider.OPENAI,
            "text-embedding-3-small",
            EmbeddingEngine,
        ),
    ],
)
def test_create_runtime_supports_every_registered_provider_capability(
    capability: factory_module._Capability,
    provider: Provider,
    model: str,
    engine_type: type[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[factory_module._ProviderEngine] = []
    registered = factory_module._FACTORIES[(provider, capability)]

    def capture_engine(
        model: str,
        api_key: SecretStr,
        transport: TransportConfig,
    ) -> factory_module._ProviderEngine:
        assert isinstance(api_key, SecretStr)
        engine = registered.create(model, api_key, transport)
        constructed.append(engine)
        return engine

    factories = dict(factory_module._FACTORIES)
    factories[(provider, capability)] = factory_module._EngineFactory(
        models=registered.models,
        create=capture_engine,
    )
    monkeypatch.setattr(factory_module, "_FACTORIES", factories)

    configured = _engine_config("primary", provider, model)
    if capability == "language_model":
        config = RuntimeConfig(language_models=(configured,))
    else:
        config = RuntimeConfig(embeddings=(configured,))
    runtime = factory_module.create_runtime(config)
    try:
        assert isinstance(runtime, Runtime)
        assert len(constructed) == 1
        assert isinstance(constructed[0], engine_type)
    finally:
        runtime.close()


def test_factory_composes_runtime_through_direct_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    runtime_type = Runtime

    def capture_runtime(**kwargs: object) -> Runtime:
        captured.update(kwargs)
        return runtime_type(**kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(factory_module, "Runtime", capture_runtime)
    runtime = factory_module.create_runtime(
        RuntimeConfig(
            language_models=(_engine_config("primary", Provider.OPENAI, "gpt-5.4"),),
            default_language_model="primary",
        )
    )
    try:
        assert set(cast("dict[str, object]", captured["language_models"])) == {"primary"}
        assert captured["default_language_model"] == "primary"
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
            RuntimeConfig(
                embeddings=(
                    _engine_config(
                        "vector",
                        Provider.DEEPSEEK,
                        "deepseek-v4-flash",
                    ),
                ),
            ),
            UnsupportedCapabilityError,
            "does not support embedding",
        ),
        (
            RuntimeConfig(
                language_models=(
                    _engine_config(
                        "chat",
                        Provider.DEEPSEEK,
                        "future-model",
                    ),
                ),
            ),
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
        "_create_engine",
        lambda _resolved: pytest.fail("engine resources must not be allocated"),
    )

    with pytest.raises(error, match=message):
        factory_module.create_runtime(config)


def test_every_configuration_is_resolved_before_any_transport_is_allocated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RuntimeConfig(
        language_models=(_engine_config("chat", Provider.OPENAI, "gpt-5.4"),),
        embeddings=(
            _engine_config(
                "vector",
                Provider.DEEPSEEK,
                "deepseek-v4-flash",
            ),
        ),
    )
    monkeypatch.setattr(
        factory_module,
        "_create_engine",
        lambda _resolved: pytest.fail("engine resources must not be allocated"),
    )

    with pytest.raises(UnsupportedCapabilityError, match="does not support embedding"):
        factory_module.create_runtime(config)


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


class _RecordingLanguageEngine:
    def __init__(
        self,
        name: str,
        calls: list[str],
        client: _RecordingClient,
    ) -> None:
        self.name = name
        self.calls = calls
        self.client = client

    def execute(self, _request: LanguageModelRequest) -> LanguageModelResponse:
        self.calls.append(self.name)
        return cast("LanguageModelResponse", object())

    def close(self) -> None:
        self.client.close()


def test_same_provider_model_instances_keep_distinct_keys_clients_and_transports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []
    clients: list[_RecordingClient] = []
    observed: list[tuple[str, str, TransportConfig, _RecordingClient]] = []
    engine_calls: list[str] = []
    engines: list[_RecordingLanguageEngine] = []

    def create_engine(
        model: str,
        api_key: SecretStr,
        transport: TransportConfig,
    ) -> factory_module._ProviderEngine:
        client = _RecordingClient(str(transport.request_timeout), attempts)
        clients.append(client)
        observed.append(
            (
                model,
                api_key.get_secret_value(),
                transport,
                client,
            )
        )
        engine = _RecordingLanguageEngine(
            api_key.get_secret_value(),
            engine_calls,
            client,
        )
        engines.append(engine)
        return cast("factory_module._ProviderEngine", engine)

    monkeypatch.setattr(
        factory_module,
        "_FACTORIES",
        {
            (Provider.OPENAI, "language_model"): factory_module._EngineFactory(
                models={"same-model": object()},
                create=create_engine,
            ),
        },
    )
    first = _engine_config(
        "tenant-a",
        Provider.OPENAI,
        "same-model",
        api_key="first-key",
        timeout=1.0,
    )
    second = _engine_config(
        "tenant-b",
        Provider.OPENAI,
        "same-model",
        api_key="second-key",
        timeout=2.0,
    )

    runtime = factory_module.create_runtime(
        RuntimeConfig(
            language_models=(first, second),
            default_language_model="tenant-b",
        )
    )
    assert [
        (model, key, transport.request_timeout, client)
        for model, key, transport, client in observed
    ] == [
        ("same-model", "first-key", 1.0, clients[0]),
        ("same-model", "second-key", 2.0, clients[1]),
    ]
    assert clients[0] is not clients[1]
    assert attempts == []
    assert not hasattr(runtime, "handles")
    assert not hasattr(runtime, "clients")
    request = LanguageModelRequest(messages=(UserMessage(content=(TextContent(text="hello"),)),))

    with runtime:
        runtime.execute(request)

    assert engine_calls == ["second-key"]
    assert len(engines) == 2

    runtime.close()

    assert attempts == ["2.0", "1.0"]
    assert [client.close_count for client in clients] == [1, 1]


def test_http_client_uses_finite_timeout_and_bounded_connect_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Transport:
        def __init__(self, *, retries: int) -> None:
            captured["retries"] = retries

        def close(self) -> None:
            pass

    class HTTPClient:
        def __init__(self, *, timeout: httpx.Timeout, transport: object) -> None:
            captured["timeout"] = timeout
            captured["transport"] = transport

        def close(self) -> None:
            pass

    monkeypatch.setattr(factory_module.httpx, "HTTPTransport", Transport)
    monkeypatch.setattr(factory_module.httpx, "Client", HTTPClient)
    config = TransportConfig(
        request_timeout=30.0,
        connect_timeout=2.0,
        connect_retries=3,
    )

    client = openai.Client(
        api_key=SecretStr("test-key"),
        timeout=factory_module._timeout(config),
        connect_retries=config.connect_retries,
    )
    client.close()

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
        openai.Client(api_key=SecretStr("test-key"))

    assert caught.value is failure
    assert close_attempts == 1


def _construction_factory(
    create: Callable[
        [str, SecretStr, TransportConfig],
        factory_module._ProviderEngine,
    ],
) -> factory_module._EngineFactory:
    return factory_module._EngineFactory(
        models={"language": object()},
        create=create,
    )


def _three_engine_config() -> RuntimeConfig:
    return RuntimeConfig(
        language_models=(
            _engine_config("first", Provider.OPENAI, "language", timeout=1.0),
            _engine_config("second", Provider.OPENAI, "language", timeout=2.0),
            _engine_config("third", Provider.OPENAI, "language", timeout=3.0),
        ),
        default_language_model="first",
    )


def test_later_engine_construction_failure_closes_current_and_earlier_resources_in_reverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []
    clients = {
        1.0: _RecordingClient("first", attempts),
        2.0: _RecordingClient("second", attempts),
        3.0: _RecordingClient("third", attempts),
    }
    failure = RuntimeError("third construction failed")

    def create_engine(
        _model: str,
        _key: SecretStr,
        transport: TransportConfig,
    ) -> factory_module._ProviderEngine:
        client = clients[transport.request_timeout]
        if transport.request_timeout == 3.0:
            client.close()
            raise failure
        return cast(
            "factory_module._ProviderEngine",
            _RecordingLanguageEngine(client.name, [], client),
        )
    monkeypatch.setattr(
        factory_module,
        "_FACTORIES",
        {
            (Provider.OPENAI, "language_model"): _construction_factory(create_engine),
        },
    )

    with pytest.raises(RuntimeError, match="third construction failed") as caught:
        factory_module.create_runtime(_three_engine_config())

    assert caught.value is failure
    assert attempts == ["third", "second", "first"]
    assert [client.close_count for client in clients.values()] == [1, 1, 1]


def test_all_cleanup_failures_become_notes_on_original_construction_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []
    clients = {
        1.0: _RecordingClient("first", attempts, RuntimeError("first close failed")),
        2.0: _RecordingClient("second", attempts, KeyboardInterrupt("second close failed")),
        3.0: _RecordingClient("third", attempts),
    }
    failure = LookupError("runtime construction failed")

    def create_engine(
        _model: str,
        _key: SecretStr,
        transport: TransportConfig,
    ) -> factory_module._ProviderEngine:
        client = clients[transport.request_timeout]
        return cast(
            "factory_module._ProviderEngine",
            _RecordingLanguageEngine(client.name, [], client),
        )

    def fail_runtime(**_kwargs: object) -> Runtime:
        raise failure

    monkeypatch.setattr(
        factory_module,
        "_FACTORIES",
        {
            (Provider.OPENAI, "language_model"): _construction_factory(create_engine),
        },
    )
    monkeypatch.setattr(factory_module, "Runtime", fail_runtime)

    with pytest.raises(LookupError, match="runtime construction failed") as caught:
        factory_module.create_runtime(_three_engine_config())

    assert caught.value is failure
    assert attempts == ["third", "second", "first"]
    assert len(caught.value.__notes__) == 2
    assert "second close failed" in caught.value.__notes__[0]
    assert "first close failed" in caught.value.__notes__[1]
