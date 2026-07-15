from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Self

from pydantic import BeforeValidator, Field, TypeAdapter, field_validator, model_validator

from symai.runtime.models import FrozenModel, ProviderId

_PROVIDER_ID_ADAPTER = TypeAdapter(ProviderId)


def _normalize_implementation_id(value: object) -> str:
    """Normalize case to lowercase while rejecting all whitespace and noncanonical shape."""
    if not isinstance(value, str):
        msg = "Implementation ID must be a string"
        raise ValueError(msg)
    if value.count(":") != 1:
        msg = "Implementation ID must contain exactly one colon"
        raise ValueError(msg)

    provider, name = value.split(":")
    try:
        normalized_provider = _PROVIDER_ID_ADAPTER.validate_python(provider)
        normalized_name = _PROVIDER_ID_ADAPTER.validate_python(name)
    except ValueError as error:
        msg = "Implementation ID components must be nonempty and contain no whitespace"
        raise ValueError(msg) from error

    return f"{normalized_provider}:{normalized_name}"


ImplementationId = Annotated[str, BeforeValidator(_normalize_implementation_id)]


class EngineSpec(FrozenModel):
    implementation: ImplementationId
    settings: Mapping[str, object]

    @field_validator("settings", mode="after")
    @classmethod
    def freeze_settings(cls, settings: Mapping[str, object]) -> Mapping[str, object]:
        return MappingProxyType(dict(settings))


class RuntimeConfig(FrozenModel):
    language_models: Mapping[str, EngineSpec] = Field(default_factory=dict)
    embeddings: Mapping[str, EngineSpec] = Field(default_factory=dict)
    default_language_model: str | None = None
    default_embedding: str | None = None

    @field_validator("language_models", "embeddings", mode="after")
    @classmethod
    def freeze_engines(
        cls,
        engines: Mapping[str, EngineSpec],
    ) -> Mapping[str, EngineSpec]:
        return MappingProxyType(dict(engines))

    @model_validator(mode="after")
    def validate_aliases_and_defaults(self) -> Self:
        if not self.language_models and not self.embeddings:
            msg = "Runtime configuration requires at least one engine"
            raise ValueError(msg)

        self._validate_aliases("language model", self.language_models)
        self._validate_aliases("embedding", self.embeddings)
        self._validate_default(
            "language model",
            self.default_language_model,
            self.language_models,
        )
        self._validate_default("embedding", self.default_embedding, self.embeddings)
        return self

    @staticmethod
    def _validate_aliases(operation: str, engines: Mapping[str, EngineSpec]) -> None:
        for alias in engines:
            if not alias:
                msg = f"{operation.capitalize()} engine alias must not be empty"
                raise ValueError(msg)
            if alias != alias.strip():
                msg = f"{operation.capitalize()} engine alias must not contain outer whitespace"
                raise ValueError(msg)

    @staticmethod
    def _validate_default(
        operation: str,
        default: str | None,
        engines: Mapping[str, EngineSpec],
    ) -> None:
        if default is None:
            return
        if not default or default != default.strip():
            msg = f"Default {operation} engine alias is invalid: {default!r}"
            raise ValueError(msg)
        if default in engines:
            return

        msg = f"Default {operation} engine alias is not configured: {default!r}"
        raise ValueError(msg)
