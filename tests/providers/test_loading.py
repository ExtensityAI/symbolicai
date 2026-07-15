from collections.abc import Callable, Mapping

import pytest
from pydantic import SecretStr, ValidationError

from symai.providers.cerebras.loading import load_chat_completions as load_cerebras
from symai.providers.cerebras.settings import ChatCompletionsSettings as CerebrasSettings
from symai.providers.deepseek.loading import load_chat_completions as load_deepseek
from symai.providers.deepseek.settings import ChatCompletionsSettings as DeepSeekSettings
from symai.providers.openai.loading import load_embedding, load_responses
from symai.providers.openai.settings import EmbeddingSettings, ResponsesSettings
from symai.runtime.errors import UnsupportedModelError


@pytest.mark.parametrize(
    ("settings_type", "payload"),
    (
        (ResponsesSettings, {"api_key": SecretStr("key"), "model": "gpt-5.4"}),
        (
            EmbeddingSettings,
            {"api_key": SecretStr("key"), "model": "text-embedding-3-small"},
        ),
        (
            CerebrasSettings,
            {"api_key": SecretStr("key"), "model": "gpt-oss-120b"},
        ),
        (
            DeepSeekSettings,
            {"api_key": SecretStr("key"), "model": "deepseek-v4-flash"},
        ),
    ),
)
def test_provider_settings_are_distinct_strict_models(
    settings_type: type[object],
    payload: Mapping[str, object],
) -> None:
    settings = settings_type.model_validate(payload)  # type: ignore[attr-defined]

    assert settings.__class__ is settings_type
    with pytest.raises(ValidationError):
        settings_type.model_validate({**payload, "model_path": "/models/local.gguf"})  # type: ignore[attr-defined]


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
def test_provider_loaders_parse_settings_and_construct_engines(
    load: Callable[[Mapping[str, object]], object],
    settings: Mapping[str, object],
    engine_name: str,
) -> None:
    engine = load(settings)
    try:
        assert type(engine).__name__ == engine_name
    finally:
        engine.close()  # type: ignore[attr-defined]


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
    client_module = __import__(f"{module_name}.client", fromlist=["Client"])
    monkeypatch.setattr(client_module.Client, "__init__", reject_allocation)

    with pytest.raises(UnsupportedModelError):
        load({"api_key": SecretStr("key"), "model": foreign_model})

    assert allocations == 0
