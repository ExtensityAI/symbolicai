from collections.abc import Callable, Mapping

import pytest
from pydantic import SecretStr

from symai.loading import load_runtime
from symai.providers.cerebras.loading import load_chat_completions as load_cerebras
from symai.providers.deepseek.loading import load_chat_completions as load_deepseek
from symai.providers.openai.client import client as openai_client
from symai.providers.openai.loading import load_embedding, load_responses
from symai.runtime.config import EngineConfig, RuntimeConfig
from symai.runtime.errors import UnsupportedModelError


@pytest.mark.parametrize(
    ("load", "settings", "engine_name"),
    (
        (
            load_responses,
            {"api_key": SecretStr("key"), "model": "gpt-5.4"},
            "ResponsesEngine",
        ),
        (
            load_embedding,
            {"api_key": SecretStr("key"), "model": "text-embedding-3-small"},
            "EmbeddingEngine",
        ),
        (
            load_cerebras,
            {"api_key": SecretStr("key"), "model": "gpt-oss-120b"},
            "ChatCompletionsEngine",
        ),
        (
            load_deepseek,
            {"api_key": SecretStr("key"), "model": "deepseek-v4-flash"},
            "ChatCompletionsEngine",
        ),
    ),
)
def test_provider_loaders_resolve_to_a_factory_that_constructs_the_engine(
    load: Callable[[Mapping[str, object]], object],
    settings: Mapping[str, object],
    engine_name: str,
) -> None:
    factory = load(settings)

    assert callable(factory)

    engine = factory()  # type: ignore[operator]
    try:
        assert type(engine).__name__ == engine_name
    finally:
        engine.close()


def test_resolving_a_provider_loader_allocates_no_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocations = 0

    def count_allocation(*_args: object, **_kwargs: object) -> None:
        nonlocal allocations
        allocations += 1

    from symai.providers.openai.client import client as openai_client

    monkeypatch.setattr(openai_client.Client, "__init__", count_allocation)
    factory = load_responses({"api_key": SecretStr("key"), "model": "gpt-5.4"})

    assert allocations == 0

    factory()

    assert allocations == 1


@pytest.mark.parametrize(
    ("load", "foreign_model"),
    (
        (load_responses, "gpt-oss-120b"),
        (load_cerebras, "deepseek-v4-flash"),
        (load_deepseek, "gpt-5.4"),
        (load_embedding, "deepseek-v4-flash"),
    ),
)
def test_provider_loaders_reject_cross_provider_settings_before_client_allocation(
    load: Callable[[Mapping[str, object]], object],
    foreign_model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocations = 0

    def reject_allocation(*_args: object, **_kwargs: object) -> None:
        nonlocal allocations
        allocations += 1
        raise AssertionError("client allocation must follow provider validation")

    module_name = load.__module__.rsplit(".", 1)[0]
    client_module = __import__(f"{module_name}.client.client", fromlist=["Client"])
    monkeypatch.setattr(client_module.Client, "__init__", reject_allocation)

    with pytest.raises(UnsupportedModelError):
        load({"api_key": SecretStr("key"), "model": foreign_model})

    assert allocations == 0


def test_no_transport_is_allocated_when_a_later_engine_is_misconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIXPLAN §2: construction resolves all configurations before allocating transport.

    A settings error in the last engine must not leave the first engine's HTTP client
    allocated and then torn down.
    """
    allocations = 0
    real_init = openai_client.Client.__init__

    def count_allocation(self: object, **kwargs: object) -> None:
        nonlocal allocations
        allocations += 1
        real_init(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(openai_client.Client, "__init__", count_allocation)

    config = RuntimeConfig(
        language_models={
            "good": EngineConfig(
                implementation="openai:responses",
                settings={"api_key": SecretStr("key"), "model": "gpt-5.4"},
            ),
            "bad": EngineConfig(
                implementation="openai:responses",
                settings={"api_key": SecretStr("key"), "model": "NOT-A-REAL-MODEL"},
            ),
        },
    )

    with pytest.raises(UnsupportedModelError):
        load_runtime(config)

    assert allocations == 0
