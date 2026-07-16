from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import Lock, get_ident
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Literal, cast, overload

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

if TYPE_CHECKING:
    from collections.abc import Mapping

type _OwnedEngine = LanguageModelEngine | EmbeddingEngine


class _RuntimeState(StrEnum):
    CREATED = "created"
    ACTIVE = "active"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class LanguageModel:
    _runtime: Runtime
    _name: str

    def execute(self, request: LanguageModelRequest, /) -> LanguageModelResponse:
        return self._runtime.execute(request, engine=self._name)


@dataclass(frozen=True, slots=True)
class EmbeddingModel:
    _runtime: Runtime
    _name: str

    def execute(self, request: EmbeddingRequest, /) -> EmbeddingResponse:
        return self._runtime.execute(request, engine=self._name)


class Runtime:
    """Single-threaded lifecycle owner for explicitly composed engines."""

    __slots__ = (
        "_acceptance_order",
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
    ) -> None:
        language_snapshot = dict(language_models) if language_models is not None else {}
        embedding_snapshot = dict(embeddings) if embeddings is not None else {}

        self._validate_aliases("language-model", language_snapshot)
        self._validate_aliases("embedding", embedding_snapshot)
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
        self._lifecycle_lock = Lock()
        self._owner_thread_id: int | None = None
        self._state = _RuntimeState.CREATED

    @staticmethod
    def _validate_aliases(
        operation: str,
        engines: Mapping[str, object],
    ) -> None:
        for alias in engines:
            if not isinstance(alias, str):
                msg = f"{operation.capitalize()} engine alias must be a string"
                raise TypeError(msg)
            if not alias:
                msg = f"{operation.capitalize()} engine alias must not be empty"
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

    def language_model(self, name: str | None = None, /) -> LanguageModel:
        engine_name = self._resolve_engine_name(
            "language_model",
            self._language_models,
            self._embeddings,
            name,
        )
        return LanguageModel(self, engine_name)

    def embedding(self, name: str | None = None, /) -> EmbeddingModel:
        engine_name = self._resolve_engine_name(
            "embedding",
            self._embeddings,
            self._language_models,
            name,
        )
        return EmbeddingModel(self, engine_name)

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
                engine_name = self._resolve_engine_name(
                    "language_model",
                    self._language_models,
                    self._embeddings,
                    engine,
                )
                selected = self._language_models[engine_name]
            elif isinstance(request, EmbeddingRequest):
                engine_name = self._resolve_engine_name(
                    "embedding",
                    self._embeddings,
                    self._language_models,
                    engine,
                )
                selected = self._embeddings[engine_name]
            else:
                msg = f"Unsupported runtime request type: {type(request).__name__}"
                raise TypeError(msg)

        if isinstance(request, LanguageModelRequest):
            return cast("LanguageModelEngine", selected).execute(request)
        return cast("EmbeddingEngine", selected).execute(request)

    @staticmethod
    def _resolve_engine_name(
        capability: EngineCapability,
        engines: Mapping[str, object],
        other_engines: Mapping[str, object],
        engine_name: str | None,
    ) -> str:
        if engine_name is not None:
            if engine_name in engines:
                return engine_name
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

        if len(engines) == 1:
            return next(iter(engines))
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
