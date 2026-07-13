import httpx
import pytest

from symai.backend.engines.embedding.openai import EmbeddingEngine
from symai.backend.engines.language_model.cerebras import (
    LanguageModelEngine as CerebrasLanguageModelEngine,
)
from symai.backend.engines.language_model.deepseek import (
    LanguageModelEngine as DeepSeekLanguageModelEngine,
)
from symai.backend.engines.language_model.openai import (
    LanguageModelEngine as OpenAILanguageModelEngine,
)
from symai.backend.provider_runtime import (
    ProviderRuntimeOptions,
    create_provider_engine_handle,
)


@pytest.mark.parametrize(
    ("capability", "model", "engine_type"),
    [
        ("language_model", "openai:gpt-5.4", OpenAILanguageModelEngine),
        ("language_model", "cerebras:gpt-oss-120b", CerebrasLanguageModelEngine),
        ("language_model", "deepseek:deepseek-v4-flash", DeepSeekLanguageModelEngine),
        ("embedding", "openai:text-embedding-3-small", EmbeddingEngine),
    ],
)
def test_create_provider_engine_handle_composes_known_provider_model(
    capability, model, engine_type
):
    lease = create_provider_engine_handle(
        capability=capability,
        model=model,
        api_key="test-key",
    )

    assert lease is not None
    assert isinstance(lease.engine, engine_type)
    assert lease.engine.model == model.partition(":")[2]
    lease.close()


@pytest.mark.parametrize(
    ("capability", "model", "message"),
    [
        ("language_model", "deepseek:future-model", "Unsupported DeepSeek chat model"),
        ("embedding", "cerebras:gpt-oss-120b", "does not provide embedding"),
        ("language_model", "gpt-5.4", "must include a provider prefix"),
    ],
)
def test_create_provider_engine_handle_rejects_invalid_managed_provider_config(
    capability, model, message
):
    with pytest.raises(ValueError, match=message):
        create_provider_engine_handle(
            capability=capability,
            model=model,
            api_key="test-key",
        )


def test_create_provider_engine_handle_leaves_unmanaged_provider_to_legacy_registry():
    assert (
        create_provider_engine_handle(
            capability="language_model",
            model="gemini-3.1-pro",
            api_key="test-key",
        )
        is None
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_timeout", 0),
        ("request_timeout", float("inf")),
        ("request_timeout", float("nan")),
        ("connect_timeout", 0),
        ("connect_timeout", float("inf")),
        ("connect_retries", -1),
    ],
)
def test_provider_runtime_options_reject_invalid_transport_bounds(field, value):
    with pytest.raises(ValueError, match=field):
        ProviderRuntimeOptions(**{field: value})


def test_create_provider_engine_handle_owns_transport_with_explicit_options(monkeypatch):
    http_client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    captured = {}

    def create_http_client(options):
        captured["options"] = options
        return http_client

    monkeypatch.setattr(
        "symai.backend.provider_runtime.create_provider_http_client", create_http_client
    )
    options = ProviderRuntimeOptions(request_timeout=30, connect_timeout=2, connect_retries=1)

    lease = create_provider_engine_handle(
        capability="language_model",
        model="deepseek:deepseek-v4-flash",
        api_key="test-key",
        options=options,
    )

    assert lease is not None
    assert captured["options"] == options
    assert http_client.is_closed is False

    lease.close()

    assert http_client.is_closed is True
