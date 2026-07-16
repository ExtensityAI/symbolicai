from collections.abc import Callable, Mapping

import pytest
from pydantic import BaseModel, ConfigDict

from symai.loading import load_runtime as load_builtin_runtime
from symai.runtime.config import EngineConfig, RuntimeConfig
from symai.runtime.loading import load_runtime
from symai.runtime.observability import ExecutionRecord
from symai.runtime.models import (
    AssistantOutputMessage,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    FinishReason,
    LanguageModelOutput,
    LanguageModelRequest,
    LanguageModelResponse,
    ResponseMetadata,
    TextContent,
    UserMessage,
)


class LocalSettings(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    model_path: str
    context_size: int


class LocalEngine:
    provider = "external"

    def __init__(self, settings: LocalSettings, events: list[str]) -> None:
        self.settings = settings
        self._events = events

    def execute(self, request: LanguageModelRequest, /) -> LanguageModelResponse:
        self._events.append("execute")
        return LanguageModelResponse(
            outputs=(
                LanguageModelOutput(
                    index=0,
                    message=AssistantOutputMessage(
                        content=(TextContent(text=f"loaded:{self.settings.model_path}"),)
                    ),
                    finish_reason=FinishReason.STOP,
                ),
            ),
            metadata=ResponseMetadata(
                provider=self.provider,
                requested_model=self.settings.model_path,
                status_code=200,
            ),
        )

    def close(self) -> None:
        self._events.append("close")


class RecordingEngine(LocalEngine):
    def __init__(
        self,
        name: str,
        events: list[str],
        close_error: BaseException | None = None,
    ) -> None:
        super().__init__(
            LocalSettings(model_path=name, context_size=1),
            events,
        )
        self._name = name
        self._close_error = close_error

    def close(self) -> None:
        self._events.append(f"close-{self._name}")
        if self._close_error is not None:
            raise self._close_error


def _request() -> LanguageModelRequest:
    return LanguageModelRequest(messages=(UserMessage(content=(TextContent(text="hello"),)),))


def test_credential_free_external_loader_constructs_and_executes() -> None:
    events: list[str] = []
    records: list[ExecutionRecord] = []

    def load_local(settings: Mapping[str, object]) -> Callable[[], LocalEngine]:
        parsed = LocalSettings.model_validate(dict(settings))
        events.append("load")
        return lambda: LocalEngine(parsed, events)

    config = RuntimeConfig(
        language_models={
            "offline": EngineConfig(
                implementation="local:gguf",
                settings={"model_path": "/models/tiny.gguf", "context_size": 4096},
            )
        },
    )

    runtime = load_runtime(
        config,
        language_model_loaders=(("local:gguf", load_local),),
        embedding_loaders=(),
        observers=(records.append,),
    )
    with runtime:
        response = runtime.execute(_request())

    assert response.outputs[0].text == "loaded:/models/tiny.gguf"
    assert events == ["load", "execute", "close"]
    assert [record.engine for record in records] == ["offline"]


def test_builtin_loader_forwards_execution_observers() -> None:
    events: list[str] = []
    records: list[ExecutionRecord] = []

    def load_local(settings: Mapping[str, object]) -> Callable[[], LocalEngine]:
        parsed = LocalSettings.model_validate(dict(settings))
        events.append("load")
        return lambda: LocalEngine(parsed, events)

    config = RuntimeConfig(
        language_models={
            "offline": EngineConfig(
                implementation="local:observed",
                settings={"model_path": "/models/tiny.gguf", "context_size": 4096},
            )
        },
    )

    runtime = load_builtin_runtime(
        config,
        language_model_loaders=(("local:observed", load_local),),
        observers=(records.append,),
    )
    with runtime:
        runtime.execute(_request())

    assert [record.engine for record in records] == ["offline"]


def test_all_implementation_references_are_checked_before_allocation() -> None:
    calls = 0

    def load_local(settings: Mapping[str, object]) -> LocalEngine:
        nonlocal calls
        calls += 1
        return LocalEngine(LocalSettings.model_validate(dict(settings)), [])

    config = RuntimeConfig(
        language_models={
            "known": EngineConfig(
                implementation="local:known",
                settings={"model_path": "known", "context_size": 1},
            ),
            "missing": EngineConfig(
                implementation="local:missing",
                settings={"model_path": "missing", "context_size": 1},
            ),
        }
    )

    with pytest.raises(ValueError, match="local:missing"):
        load_runtime(
            config,
            language_model_loaders=(("local:known", load_local),),
            embedding_loaders=(),
        )

    assert calls == 0


def test_duplicate_loader_ids_are_rejected_before_allocation() -> None:
    calls = 0

    def load_local(settings: Mapping[str, object]) -> LocalEngine:
        nonlocal calls
        calls += 1
        return LocalEngine(LocalSettings.model_validate(dict(settings)), [])

    config = RuntimeConfig(
        language_models={
            "local": EngineConfig(
                implementation="local:duplicate",
                settings={"model_path": "local", "context_size": 1},
            )
        }
    )

    with pytest.raises(ValueError, match="Duplicate implementation ID"):
        load_runtime(
            config,
            language_model_loaders=(
                ("local:duplicate", load_local),
                ("LOCAL:DUPLICATE", load_local),
            ),
            embedding_loaders=(),
        )

    assert calls == 0


def test_same_implementation_id_can_load_once_per_operation() -> None:
    events: list[str] = []

    class LanguageEngine(LocalEngine):
        def close(self) -> None:
            events.append("close-language")

    class VectorEngine:
        def execute(self, request: EmbeddingRequest, /) -> EmbeddingResponse:
            events.append("execute-embedding")
            return EmbeddingResponse(
                vectors=(EmbeddingVector(index=0, values=(1.0,)),),
                metadata=ResponseMetadata(
                    provider="external",
                    requested_model="shared",
                    status_code=200,
                ),
            )

        def close(self) -> None:
            events.append("close-embedding")

    def load_language(settings: Mapping[str, object]) -> Callable[[], LanguageEngine]:
        events.append("load-language")
        parsed = LocalSettings.model_validate(dict(settings))
        return lambda: LanguageEngine(parsed, events)

    def load_embedding(_settings: Mapping[str, object]) -> Callable[[], VectorEngine]:
        events.append("load-embedding")
        return VectorEngine

    config = RuntimeConfig(
        language_models={
            "chat": EngineConfig(
                implementation="shared:engine",
                settings={"model_path": "shared", "context_size": 1},
            )
        },
        embeddings={
            "vector": EngineConfig(
                implementation="shared:engine",
                settings={},
            )
        },
    )

    runtime = load_runtime(
        config,
        language_model_loaders=(("shared:engine", load_language),),
        embedding_loaders=(("shared:engine", load_embedding),),
    )
    with runtime:
        language_response = runtime.execute(_request())
        embedding_response = runtime.execute(EmbeddingRequest(inputs=("hello",)))

    assert language_response.outputs[0].text == "loaded:shared"
    assert embedding_response.vectors[0].values == (1.0,)
    assert events == [
        "load-language",
        "load-embedding",
        "execute",
        "execute-embedding",
        "close-embedding",
        "close-language",
    ]


def test_duplicate_builtin_and_external_ids_are_rejected_before_allocation() -> None:
    calls = 0

    def duplicate(settings: Mapping[str, object]) -> LocalEngine:
        nonlocal calls
        calls += 1
        return LocalEngine(LocalSettings.model_validate(dict(settings)), [])

    config = RuntimeConfig(
        language_models={
            "primary": EngineConfig(
                implementation="openai:responses",
                settings={"api_key": "not-used", "model": "gpt-5.4"},
            )
        }
    )

    with pytest.raises(ValueError, match="Duplicate implementation ID"):
        load_builtin_runtime(
            config,
            language_model_loaders=(("OPENAI:RESPONSES", duplicate),),
        )

    assert calls == 0


def test_loading_rolls_back_exhaustively_and_preserves_primary_failure() -> None:
    events: list[str] = []
    close_a = RuntimeError("close-a failed")
    close_b = RuntimeError("close-b failed")
    primary = RuntimeError("load-c failed")

    # The failure is in the construct phase: a resolution failure allocates nothing, so
    # rollback is only reachable once engines exist.
    def loader(name: str, close_error: BaseException | None = None):
        def load(_settings: Mapping[str, object]) -> Callable[[], RecordingEngine]:
            def construct() -> RecordingEngine:
                events.append(f"load-{name}")
                return RecordingEngine(name, events, close_error)

            return construct

        return load

    def fail(_settings: Mapping[str, object]) -> Callable[[], RecordingEngine]:
        def construct() -> RecordingEngine:
            events.append("load-c")
            raise primary

        return construct

    config = RuntimeConfig(
        language_models={
            "a": EngineConfig(implementation="test:a", settings={}),
            "b": EngineConfig(implementation="test:b", settings={}),
            "c": EngineConfig(implementation="test:c", settings={}),
        }
    )

    with pytest.raises(RuntimeError) as raised:
        load_runtime(
            config,
            language_model_loaders=(
                ("test:a", loader("a", close_a)),
                ("test:b", loader("b", close_b)),
                ("test:c", fail),
            ),
            embedding_loaders=(),
        )

    assert raised.value is primary
    assert events == ["load-a", "load-b", "load-c", "close-b", "close-a"]
    cleanup_group = raised.value.__cause__
    assert isinstance(cleanup_group, BaseExceptionGroup)
    assert cleanup_group.exceptions == (close_b, close_a)


def test_a_resolution_failure_constructs_no_engine_at_all() -> None:
    """FIXPLAN §2: construction resolves all configurations before allocating transport."""
    events: list[str] = []
    primary = RuntimeError("settings for c are invalid")

    def loader(name: str):
        def load(_settings: Mapping[str, object]) -> Callable[[], RecordingEngine]:
            events.append(f"resolve-{name}")

            def construct() -> RecordingEngine:
                events.append(f"load-{name}")
                return RecordingEngine(name, events, None)

            return construct

        return load

    def fail_resolution(_settings: Mapping[str, object]) -> Callable[[], RecordingEngine]:
        events.append("resolve-c")
        raise primary

    config = RuntimeConfig(
        language_models={
            "a": EngineConfig(implementation="test:a", settings={}),
            "b": EngineConfig(implementation="test:b", settings={}),
            "c": EngineConfig(implementation="test:c", settings={}),
        }
    )

    with pytest.raises(RuntimeError) as raised:
        load_runtime(
            config,
            language_model_loaders=(
                ("test:a", loader("a")),
                ("test:b", loader("b")),
                ("test:c", fail_resolution),
            ),
            embedding_loaders=(),
        )

    assert raised.value is primary
    # No engine was constructed, so nothing had to be closed.
    assert events == ["resolve-a", "resolve-b", "resolve-c"]


def test_a_loader_returning_a_non_factory_is_rejected_before_construction() -> None:
    config = RuntimeConfig(
        language_models={"a": EngineConfig(implementation="test:a", settings={})}
    )

    with pytest.raises(TypeError, match="engine factory"):
        load_runtime(
            config,
            language_model_loaders=(("test:a", lambda _settings: "not-a-factory"),),  # pyright: ignore[reportArgumentType]
            embedding_loaders=(),
        )
