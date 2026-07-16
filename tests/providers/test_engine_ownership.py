import json
from collections.abc import Callable
from inspect import signature

import httpx
import pytest
from pydantic import SecretStr

from symai.providers import cerebras, deepseek, openai
from symai.providers._client import errors as _client_errors
from symai.providers._client import transport as _client_transport
from symai.providers._engine import mapping
from symai.runtime import errors as runtime_errors
from symai.runtime.runtime import Runtime


class CountingTransport(httpx.MockTransport):
    def __init__(self) -> None:
        super().__init__(self._handle)
        self.close_count = 0

    @staticmethod
    def _handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    def close(self) -> None:
        self.close_count += 1
        super().close()


ProviderClient = openai.Client | cerebras.Client | deepseek.Client
ProviderEngine = (
    openai.ResponsesEngine
    | openai.EmbeddingEngine
    | cerebras.ChatCompletionsEngine
    | deepseek.ChatCompletionsEngine
)
ClientFactory = Callable[[CountingTransport], ProviderClient]
EngineFactory = Callable[[ProviderClient, str], ProviderEngine]


def _openai_client(transport: CountingTransport) -> openai.Client:
    return openai.Client(api_key=SecretStr("test-key"), transport=transport)


def _cerebras_client(transport: CountingTransport) -> cerebras.Client:
    return cerebras.Client(api_key=SecretStr("test-key"), transport=transport)


def _deepseek_client(transport: CountingTransport) -> deepseek.Client:
    return deepseek.Client(api_key=SecretStr("test-key"), transport=transport)


def _openai_language(client: ProviderClient, model: str) -> openai.ResponsesEngine:
    return openai.ResponsesEngine(client=client, model=model)  # pyright: ignore[reportArgumentType]


def _openai_embedding(client: ProviderClient, model: str) -> openai.EmbeddingEngine:
    return openai.EmbeddingEngine(client=client, model=model)  # pyright: ignore[reportArgumentType]


def _cerebras_language(client: ProviderClient, model: str) -> cerebras.ChatCompletionsEngine:
    return cerebras.ChatCompletionsEngine(client=client, model=model)  # pyright: ignore[reportArgumentType]


def _deepseek_language(client: ProviderClient, model: str) -> deepseek.ChatCompletionsEngine:
    return deepseek.ChatCompletionsEngine(client=client, model=model)  # pyright: ignore[reportArgumentType]


ENGINE_CASES: tuple[tuple[ClientFactory, EngineFactory, str], ...] = (
    (_openai_client, _openai_language, "gpt-5.4"),
    (_openai_client, _openai_embedding, "text-embedding-3-small"),
    (_cerebras_client, _cerebras_language, "gpt-oss-120b"),
    (_deepseek_client, _deepseek_language, "deepseek-v4-flash"),
)


@pytest.mark.parametrize("client_type", [openai.Client, cerebras.Client, deepseek.Client])
def test_provider_clients_construct_their_owned_http_client(
    client_type: type[ProviderClient],
) -> None:
    parameters = signature(client_type).parameters

    assert "transport" in parameters
    assert "http_client" not in parameters


@pytest.mark.parametrize("client_type", [openai.Client, cerebras.Client, deepseek.Client])
def test_client_construction_preserves_primary_failure_when_transport_close_fails(
    client_type: type[ProviderClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction_failure = RuntimeError("client construction failed")
    cleanup_failure = KeyboardInterrupt("transport close failed")

    class FailingCloseTransport(httpx.MockTransport):
        def close(self) -> None:
            raise cleanup_failure

    def owned_transport(**_kwargs: object) -> httpx.BaseTransport:
        return FailingCloseTransport(lambda _request: httpx.Response(200))

    def fail_client(**_kwargs: object) -> httpx.Client:
        raise construction_failure

    monkeypatch.setattr(httpx, "HTTPTransport", owned_transport)
    monkeypatch.setattr(httpx, "Client", fail_client)

    # No injected transport: the client owns the one it built, so it must close it.
    with pytest.raises(RuntimeError, match="client construction failed") as caught:
        client_type(api_key=SecretStr("test-key"))

    assert caught.value is construction_failure
    assert len(caught.value.__notes__) == 1
    assert "transport close failed" in caught.value.__notes__[0]


@pytest.mark.parametrize("client_type", [openai.Client, cerebras.Client, deepseek.Client])
def test_failed_construction_does_not_close_a_caller_owned_transport(
    client_type: type[ProviderClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TrackingTransport(httpx.MockTransport):
        def __init__(self) -> None:
            super().__init__(lambda _request: httpx.Response(200))
            self.closed = False

        def close(self) -> None:
            self.closed = True

    transport = TrackingTransport()

    def fail_client(**_kwargs: object) -> httpx.Client:
        msg = "client construction failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(httpx, "Client", fail_client)

    # An injected transport belongs to the caller and may outlive this client.
    with pytest.raises(RuntimeError, match="client construction failed"):
        client_type(api_key=SecretStr("test-key"), transport=transport)

    assert transport.closed is False


@pytest.mark.parametrize(("client_factory", "engine_factory", "model"), ENGINE_CASES)
def test_engine_close_closes_its_provider_client_exactly_once(
    client_factory: ClientFactory,
    engine_factory: EngineFactory,
    model: str,
) -> None:
    transport = CountingTransport()
    client = client_factory(transport)
    engine = engine_factory(client, model)

    engine.close()
    engine.close()

    assert transport.close_count == 1


@pytest.mark.parametrize(("client_factory", "engine_factory", "model"), ENGINE_CASES)
def test_engine_constructor_failure_closes_the_accepted_client(
    client_factory: ClientFactory,
    engine_factory: EngineFactory,
    model: str,
) -> None:
    transport = CountingTransport()
    client = client_factory(transport)

    with pytest.raises(Exception, match="Unsupported"):
        engine_factory(client, f"unsupported-{model}")

    assert transport.close_count == 1


def test_runtime_accepts_same_provider_and_model_with_distinct_owned_clients() -> None:
    first_transport = CountingTransport()
    second_transport = CountingTransport()
    first = openai.ResponsesEngine(
        client=_openai_client(first_transport),
        model="gpt-5.4",
    )
    second = openai.ResponsesEngine(
        client=_openai_client(second_transport),
        model="gpt-5.4",
    )
    runtime = Runtime(language_models={"tenant-a": first, "tenant-b": second})

    runtime.close()
    runtime.close()

    assert first_transport.close_count == 1
    assert second_transport.close_count == 1


def test_runtime_reaches_provider_client_only_through_engine_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CountingTransport()
    client = _openai_client(transport)
    engine = openai.ResponsesEngine(client=client, model="gpt-5.4")
    engine_close = engine.close
    client_close = client.close
    inside_engine_close = False

    def guarded_client_close() -> None:
        assert inside_engine_close is True
        client_close()

    def observed_engine_close() -> None:
        nonlocal inside_engine_close
        inside_engine_close = True
        try:
            engine_close()
        finally:
            inside_engine_close = False

    monkeypatch.setattr(client, "close", guarded_client_close)
    monkeypatch.setattr(engine, "close", observed_engine_close)
    runtime = Runtime(language_models={"chat": engine})

    runtime.close()

    assert transport.close_count == 1


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (400, runtime_errors.InvalidRequestError),
        (403, runtime_errors.PermissionDeniedError),
        (404, runtime_errors.InvalidRequestError),
        (500, runtime_errors.ProviderError),
        (503, runtime_errors.ProviderError),
    ],
)
def test_non_success_statuses_map_to_distinct_runtime_errors(
    status_code: int,
    expected: type[Exception],
) -> None:
    metadata = _client_transport.ResponseMetadata(
        status_code=status_code, request_id="req-1", retry_after=None
    )
    error = _client_errors.APIError(metadata, json.dumps({"error": {"code": "c", "param": "p"}}))

    with pytest.raises(expected) as raised:
        mapping.raise_mapped_client_error(
            error,
            provider="openai",
            model="gpt-5.4",
            messages=mapping.ClientErrorMessages(
                authentication="auth",
                rate_limit="rate",
                response="response",
                transport="transport",
                api="api error {status_code}",
            ),
        )

    assert raised.value.metadata is not None
    assert raised.value.metadata.status_code == status_code
    assert raised.value.metadata.error_code == "c"
    assert raised.value.metadata.param == "p"


@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [(400, False), (403, False), (408, True), (429, True), (500, True), (503, True)],
)
def test_retryability_marks_capacity_and_provider_health_only(
    status_code: int,
    retryable: bool,
) -> None:
    metadata = _client_transport.ResponseMetadata(
        status_code=status_code, request_id=None, retry_after=None
    )
    error = _client_errors.APIError(metadata, "{}")

    with pytest.raises(runtime_errors.ExecutionError) as raised:
        mapping.raise_mapped_client_error(
            error,
            provider="openai",
            model="gpt-5.4",
            messages=mapping.ClientErrorMessages(
                authentication="auth",
                rate_limit="rate",
                response="response",
                transport="transport",
                api="api error {status_code}",
            ),
        )

    assert raised.value.metadata is not None
    assert raised.value.metadata.retryable is retryable
