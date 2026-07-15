from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from threading import Lock, get_ident
from types import MappingProxyType, TracebackType
from typing import Literal, overload

from symai.runtime.engines import EmbeddingEngine, LanguageModelEngine
from symai.runtime.errors import (
    AmbiguousEngineError,
    EngineCapability,
    EngineCapabilityError,
    RuntimeClosedError,
    RuntimeOwnershipError,
    UnknownEngineError,
    UnsupportedCapabilityError,
)
from symai.runtime.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    LanguageModelRequest,
    LanguageModelResponse,
)

type _OwnedEngine = LanguageModelEngine | EmbeddingEngine


class _RuntimeState(StrEnum):
    CREATED = "created"
    ACTIVE = "active"
    CLOSED = "closed"


class Runtime:
    """Single-threaded lifecycle owner for explicitly composed engines."""

    __slots__ = (
        "_acceptance_order",
        "_default_embedding",
        "_default_language_model",
        "_embeddings",
        "_language_models",
        "_lifecycle_lock",
        "_owner_thread_id",
        "_state",
    )

    def __init__(
        self,
        *,
        language_models: Mapping[str, LanguageModelEngine] | None = None,
        embeddings: Mapping[str, EmbeddingEngine] | None = None,
        default_language_model: str | None = None,
        default_embedding: str | None = None,
    ) -> None:
        language_snapshot = dict(language_models) if language_models is not None else {}
        embedding_snapshot = dict(embeddings) if embeddings is not None else {}

        self._validate_aliases("language-model", language_snapshot)
        self._validate_aliases("embedding", embedding_snapshot)
        self._validate_default(
            "language-model",
            default_language_model,
            language_snapshot,
        )
        self._validate_default("embedding", default_embedding, embedding_snapshot)
        self._validate_engine_identities(language_snapshot, embedding_snapshot)
        if not language_snapshot and not embedding_snapshot:
            msg = "Runtime requires at least one engine"
            raise ValueError(msg)

        acceptance_order: tuple[_OwnedEngine, ...] = (
            *language_snapshot.values(),
            *embedding_snapshot.values(),
        )
        self._acceptance_order = acceptance_order
        self._language_models = MappingProxyType(language_snapshot)
        self._embeddings = MappingProxyType(embedding_snapshot)
        self._default_language_model = default_language_model
        self._default_embedding = default_embedding
        self._lifecycle_lock = Lock()
        self._owner_thread_id: int | None = None
        self._state = _RuntimeState.CREATED

    @staticmethod
    def _validate_aliases(
        operation: str,
        engines: Mapping[object, object],
    ) -> None:
        for alias in engines:
            if not isinstance(alias, str):
                msg = f"{operation.capitalize()} engine alias must be a string"
                raise TypeError(msg)
            if not alias:
                msg = f"{operation.capitalize()} engine alias must not be empty"
                raise ValueError(msg)

    @staticmethod
    def _validate_default(
        operation: str,
        default: object,
        engines: Mapping[str, object],
    ) -> None:
        if default is None:
            return
        if not isinstance(default, str):
            msg = f"Default {operation} engine alias must be a string"
            raise TypeError(msg)
        if default in engines:
            return

        msg = f"Default {operation} engine alias is not configured: {default!r}"
        raise ValueError(msg)

    @staticmethod
    def _validate_engine_identities(
        language_models: Mapping[str, LanguageModelEngine],
        embeddings: Mapping[str, EmbeddingEngine],
    ) -> None:
        identities: set[int] = set()
        for engine in (*language_models.values(), *embeddings.values()):
            identity = id(engine)
            if identity in identities:
                msg = "The same engine object cannot be accepted more than once"
                raise ValueError(msg)
            identities.add(identity)

    def __enter__(self) -> Runtime:
        with self._lifecycle_lock:
            self._require_owner_thread("enter")
            if self._state is not _RuntimeState.CREATED:
                msg = "Runtime contexts have a single lifecycle and cannot be re-entered"
                raise RuntimeClosedError(msg)

            self._owner_thread_id = get_ident()
            self._state = _RuntimeState.ACTIVE
            return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> Literal[False]:
        with self._lifecycle_lock:
            self._require_owner_thread("exit")

        try:
            self.close()
        except BaseExceptionGroup as cleanup_failures:
            if exc_value is None:
                raise
            for failure in cleanup_failures.exceptions:
                exc_value.add_note(f"Runtime cleanup failed: {failure!r}")

        return False

    @overload
    def execute(
        self,
        request: LanguageModelRequest,
        /,
        *,
        engine: str | None = None,
    ) -> LanguageModelResponse: ...

    @overload
    def execute(
        self,
        request: EmbeddingRequest,
        /,
        *,
        engine: str | None = None,
    ) -> EmbeddingResponse: ...

    def execute(
        self,
        request: LanguageModelRequest | EmbeddingRequest,
        /,
        *,
        engine: str | None = None,
    ) -> LanguageModelResponse | EmbeddingResponse:
        with self._lifecycle_lock:
            self._require_owner_thread("execute")
            if self._state is not _RuntimeState.ACTIVE:
                msg = "Runtime only accepts execution while its context is active"
                raise RuntimeClosedError(msg)

            if isinstance(request, LanguageModelRequest):
                selected = self._resolve_engine(
                    "language_model",
                    self._language_models,
                    self._embeddings,
                    self._default_language_model,
                    engine,
                )
            elif isinstance(request, EmbeddingRequest):
                selected = self._resolve_engine(
                    "embedding",
                    self._embeddings,
                    self._language_models,
                    self._default_embedding,
                    engine,
                )
            else:
                msg = f"Unsupported runtime request type: {type(request).__name__}"
                raise TypeError(msg)

        return selected.execute(request)

    @staticmethod
    def _resolve_engine[EngineT](
        capability: EngineCapability,
        engines: Mapping[str, EngineT],
        other_engines: Mapping[str, object],
        default: str | None,
        engine_name: str | None,
    ) -> EngineT:
        if engine_name is not None:
            selected = engines.get(engine_name)
            if selected is not None:
                return selected
            if engine_name in other_engines:
                other_capability: EngineCapability = (
                    "embedding" if capability == "language_model" else "language_model"
                )
                raise EngineCapabilityError(
                    engine_name,
                    requested_capability=capability,
                    engine_capability=other_capability,
                )
            raise UnknownEngineError(engine_name)

        if default is not None:
            return engines[default]
        if len(engines) == 1:
            return next(iter(engines.values()))
        if engines:
            raise AmbiguousEngineError(capability, tuple(engines))

        capability_label = capability.replace("_", "-")
        msg = f"Runtime has no {capability_label} capability"
        raise UnsupportedCapabilityError(msg)

    def close(self) -> None:
        with self._lifecycle_lock:
            self._require_owner_thread("close")
            if self._state is _RuntimeState.CLOSED:
                return

            self._state = _RuntimeState.CLOSED
            engines = tuple(reversed(self._acceptance_order))
            self._acceptance_order = ()
            self._language_models = MappingProxyType({})
            self._embeddings = MappingProxyType({})

        failures: list[BaseException] = []
        for engine in engines:
            try:
                engine.close()
            except BaseException as error:
                failures.append(error)

        if failures:
            msg = "Runtime cleanup failed"
            raise BaseExceptionGroup(msg, failures)

    def _require_owner_thread(
        self,
        operation: Literal["enter", "execute", "close", "exit"],
    ) -> None:
        owner_thread_id = self._owner_thread_id
        if owner_thread_id is None or owner_thread_id == get_ident():
            return

        raise RuntimeOwnershipError(operation)
