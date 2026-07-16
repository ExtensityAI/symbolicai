from types import MappingProxyType

import pytest
from pydantic import TypeAdapter, ValidationError

from symai.runtime.config import EngineConfig, ImplementationId, RuntimeConfig


@pytest.mark.parametrize(
    "value",
    (
        "",
        "openai",
        ":responses",
        "openai:",
        "openai:responses:extra",
        " openai:responses",
        "openai:responses ",
        "open ai:responses",
        "openai:res ponses",
        "openai:\tresponses",
    ),
)
def test_implementation_id_rejects_noncanonical_values(value: str) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(ImplementationId).validate_python(value)


def test_implementation_id_normalizes_case_and_accepts_external_ids() -> None:
    adapter = TypeAdapter(ImplementationId)

    assert adapter.validate_python("ACME_Local:GGUF_V2") == "acme_local:gguf_v2"


def test_runtime_config_copies_and_freezes_envelope_mappings_only() -> None:
    opaque_value = {"context": [4096]}
    settings = {"model_path": "/models/example.gguf", "options": opaque_value}
    language_models = {"local": EngineConfig(implementation="local:gguf", settings=settings)}

    config = RuntimeConfig(
        language_models=language_models,
    )
    settings["model_path"] = "/models/changed.gguf"
    language_models.clear()
    opaque_value["context"].append(8192)

    assert isinstance(config.language_models, MappingProxyType)
    assert isinstance(config.language_models["local"].settings, MappingProxyType)
    assert config.language_models["local"].settings["model_path"] == "/models/example.gguf"
    assert config.language_models["local"].settings["options"] is opaque_value
    assert opaque_value == {"context": [4096, 8192]}
    with pytest.raises(TypeError):
        config.language_models["other"] = config.language_models["local"]  # type: ignore[index]
    with pytest.raises(TypeError):
        config.language_models["local"].settings["model_path"] = "other"  # type: ignore[index]


@pytest.mark.parametrize(
    "field,value",
    (
        ("api_key", "secret"),
        ("base_url", "https://example.invalid"),
        ("model_path", "/models/example.gguf"),
        ("start_subprocess", True),
        ("model", "example-model"),
    ),
)
def test_provider_settings_are_not_core_engine_spec_fields(field: str, value: object) -> None:
    payload = {
        "implementation": "external:local",
        "settings": {field: value},
        field: value,
    }

    with pytest.raises(ValidationError):
        EngineConfig.model_validate(payload)

    spec = EngineConfig(
        implementation="external:local",
        settings={field: value},
    )
    assert spec.settings[field] == value


@pytest.mark.parametrize(
    "field,value",
    (
        ("api_key", "secret"),
        ("base_url", "https://example.invalid"),
        ("model_path", "/models/example.gguf"),
        ("start_subprocess", True),
        ("model", "example-model"),
    ),
)
def test_provider_settings_are_not_runtime_config_fields(field: str, value: object) -> None:
    payload = {
        "language_models": {
            "local": {
                "implementation": "external:local",
                "settings": {field: value},
            }
        },
        field: value,
    }

    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    (
        {"language_models": {"": {"implementation": "x:y", "settings": {}}}},
        {
            "language_models": {"lm": {"implementation": "x:y", "settings": {}}},
            "default_language_model": "lm",
        },
        {
            "embeddings": {"vector": {"implementation": "x:y", "settings": {}}},
            "default_embedding": "vector",
        },
    ),
)
def test_runtime_config_rejects_invalid_aliases_and_removed_defaults(payload: object) -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate(payload)
