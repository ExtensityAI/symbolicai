from collections.abc import Callable, Mapping
from contextvars import ContextVar, Token
from importlib import import_module
from threading import Thread
from typing import cast

import pytest

from symai.runtime.errors import (
    AmbiguousEngineError,
    EngineCapabilityError,
    NoActiveRuntimeError,
    RuntimeClosedError,
    RuntimeOwnershipError,
    UnknownEngineError,
)
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
from symai.runtime.runtime import Runtime, current_runtime

LANGUAGE_REQUEST = LanguageModelRequest(
    messages=(UserMessage(content=(TextContent(text="question"),)),),
)
EMBEDDING_REQUEST = EmbeddingRequest(inputs=("first", "second"))
METADATA = ResponseMetadata(
    provider="openai",
    requested_model="test-model",
    status_code=200,
)
LANGUAGE_RESPONSE = LanguageModelResponse(
    outputs=(
        LanguageModelOutput(
            index=0,
            message=AssistantOutputMessage(content=(TextContent(text="answer"),)),
            finish_reason=FinishReason.STOP,
        ),
    ),
    metadata=METADATA,
)
EMBEDDING_RESPONSE = EmbeddingResponse(
    vectors=(EmbeddingVector(index=0, values=(1.0, 2.0)),),
    metadata=METADATA,
)


class LanguageEngine:
    def __init__(
        self,
        label: str = "language",
        *,
        closed: list[str] | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.label = label
        self.requests: list[LanguageModelRequest] = []
        self.close_count = 0
        self._closed = closed
        self._close_error = close_error

    def execute(self, request: LanguageModelRequest, /) -> LanguageModelResponse:
        self.requests.append(request)
        return LANGUAGE_RESPONSE

    def close(self) -> None:
        self.close_count += 1
        if self._closed is not None:
            self._closed.append(self.label)
        if self._close_error is not None:
            raise self._close_error


class EmbeddingEngine:
    def __init__(
        self,
        label: str = "embedding",
        *,
        closed: list[str] | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.label = label
        self.requests: list[EmbeddingRequest] = []
        self.close_count = 0
        self._closed = closed
        self._close_error = close_error

    def execute(self, request: EmbeddingRequest, /) -> EmbeddingResponse:
        self.requests.append(request)
        return EMBEDDING_RESPONSE

    def close(self) -> None:
        self.close_count += 1
        if self._closed is not None:
            self._closed.append(self.label)
        if self._close_error is not None:
            raise self._close_error


def capture_thread_error(operation: Callable[[], object]) -> BaseException | None:
    errors: list[BaseException] = []

    def target() -> None:
        try:
            operation()
        except BaseException as error:
            errors.append(error)

    thread = Thread(target=target)
    thread.start()
    thread.join(timeout=2)
    assert thread.is_alive() is False
    assert len(errors) <= 1
    return errors[0] if errors else None


def test_operation_protocols_expose_only_execute_and_close() -> None:
    module = import_module("symai.runtime.engines")

    for protocol_name in ("LanguageModelEngine", "EmbeddingEngine"):
        protocol = getattr(module, protocol_name)
        public_members = {name for name in vars(protocol) if not name.startswith("_")}
        assert public_members == {"close", "execute"}


def test_empty_runtime_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one engine"):
        Runtime()


@pytest.mark.parametrize("operation", ["language_model", "embedding"])
@pytest.mark.parametrize("alias", ["", 1])
def test_invalid_alias_is_rejected_before_ownership_transfer(
    operation: str,
    alias: object,
) -> None:
    engine = LanguageEngine() if operation == "language_model" else EmbeddingEngine()
    mapping = cast("Mapping[str, object]", {alias: engine})

    with pytest.raises((TypeError, ValueError), match="alias"):
        if operation == "language_model":
            Runtime(language_models=cast("Mapping[str, LanguageEngine]", mapping))
        else:
            Runtime(embeddings=cast("Mapping[str, EmbeddingEngine]", mapping))

    assert engine.close_count == 0


def test_same_alias_is_allowed_once_in_each_operation_map() -> None:
    language = LanguageEngine()
    embedding = EmbeddingEngine()
    runtime = Runtime(
        language_models={"primary": language},
        embeddings={"primary": embedding},
    )

    with runtime:
        assert runtime.execute(LANGUAGE_REQUEST, engine="primary") is LANGUAGE_RESPONSE
        assert runtime.execute(EMBEDDING_REQUEST, engine="primary") is EMBEDDING_RESPONSE

    assert language.requests == [LANGUAGE_REQUEST]
    assert embedding.requests == [EMBEDDING_REQUEST]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "language_models": {"chat": LanguageEngine()},
                "default_language_model": "missing",
            },
            "(?i)default language-model",
        ),
        (
            {
                "embeddings": {"vectors": EmbeddingEngine()},
                "default_embedding": "missing",
            },
            "(?i)default embedding",
        ),
    ],
)
def test_invalid_default_is_rejected_before_ownership_transfer(
    kwargs: dict[str, object],
    message: str,
) -> None:
    engine = next(iter(cast("dict[str, object]", next(iter(kwargs.values()))).values()))

    with pytest.raises(ValueError, match=message):
        Runtime(**kwargs)  # pyright: ignore[reportArgumentType]

    assert cast("LanguageEngine | EmbeddingEngine", engine).close_count == 0


@pytest.mark.parametrize("across_operations", [False, True])
def test_duplicate_engine_identity_is_rejected_before_transfer(
    across_operations: bool,
) -> None:
    engine = LanguageEngine()

    with pytest.raises(ValueError, match="same engine object"):
        if across_operations:
            Runtime(
                language_models={"chat": engine},
                embeddings={"vectors": cast("EmbeddingEngine", engine)},
            )
        else:
            Runtime(language_models={"first": engine, "second": engine})

    assert engine.close_count == 0


def test_runtime_snapshots_input_mappings() -> None:
    original = LanguageEngine("original")
    replacement = LanguageEngine("replacement")
    language_models = {"chat": original}
    runtime = Runtime(language_models=language_models)
    language_models.clear()
    language_models["other"] = replacement

    with runtime:
        assert runtime.execute(LANGUAGE_REQUEST, engine="chat") is LANGUAGE_RESPONSE
        with pytest.raises(UnknownEngineError):
            runtime.execute(LANGUAGE_REQUEST, engine="other")

    assert original.requests == [LANGUAGE_REQUEST]
    assert replacement.requests == []
    assert replacement.close_count == 0


def test_explicit_selection_overrides_configured_default() -> None:
    first = LanguageEngine("first")
    second = LanguageEngine("second")
    runtime = Runtime(
        language_models={"first": first, "second": second},
        default_language_model="first",
    )

    with runtime:
        assert runtime.execute(LANGUAGE_REQUEST, engine="second") is LANGUAGE_RESPONSE

    assert first.requests == []
    assert second.requests == [LANGUAGE_REQUEST]


def test_unique_engine_is_selected_implicitly() -> None:
    engine = LanguageEngine()
    runtime = Runtime(language_models={"chat": engine})

    with runtime:
        assert runtime.execute(LANGUAGE_REQUEST) is LANGUAGE_RESPONSE

    assert engine.requests == [LANGUAGE_REQUEST]


def test_configured_defaults_are_operation_local() -> None:
    language = {"first": LanguageEngine("first"), "second": LanguageEngine("second")}
    embeddings = {"first": EmbeddingEngine("first"), "second": EmbeddingEngine("second")}
    runtime = Runtime(
        language_models=language,
        embeddings=embeddings,
        default_language_model="second",
        default_embedding="first",
    )

    with runtime:
        assert runtime.execute(LANGUAGE_REQUEST) is LANGUAGE_RESPONSE
        assert runtime.execute(EMBEDDING_REQUEST) is EMBEDDING_RESPONSE

    assert language["second"].requests == [LANGUAGE_REQUEST]
    assert embeddings["first"].requests == [EMBEDDING_REQUEST]


def test_implicit_selection_rejects_ambiguity() -> None:
    runtime = Runtime(
        language_models={"zeta": LanguageEngine(), "alpha": LanguageEngine()},
    )

    with runtime, pytest.raises(AmbiguousEngineError) as caught:
        runtime.execute(LANGUAGE_REQUEST)

    assert caught.value.engine_names == ("alpha", "zeta")


def test_explicit_selection_rejects_unknown_alias() -> None:
    runtime = Runtime(language_models={"chat": LanguageEngine()})

    with runtime, pytest.raises(UnknownEngineError) as caught:
        runtime.execute(LANGUAGE_REQUEST, engine="missing")

    assert caught.value.engine_name == "missing"


def test_explicit_selection_rejects_alias_from_wrong_operation() -> None:
    runtime = Runtime(
        language_models={"chat": LanguageEngine()},
        embeddings={"vectors": EmbeddingEngine()},
    )

    with runtime, pytest.raises(EngineCapabilityError) as caught:
        runtime.execute(LANGUAGE_REQUEST, engine="vectors")

    assert caught.value.requested_capability == "language_model"
    assert caught.value.engine_capability == "embedding"


def test_execute_before_enter_is_rejected_without_engine_touch() -> None:
    engine = LanguageEngine()
    runtime = Runtime(language_models={"chat": engine})

    with pytest.raises(RuntimeClosedError):
        runtime.execute(LANGUAGE_REQUEST)

    assert engine.requests == []
    runtime.close()


def test_execute_after_close_is_rejected() -> None:
    runtime = Runtime(language_models={"chat": LanguageEngine()})
    with runtime:
        pass

    with pytest.raises(RuntimeClosedError):
        runtime.execute(LANGUAGE_REQUEST)


def test_runtime_cannot_be_reentered() -> None:
    runtime = Runtime(language_models={"chat": LanguageEngine()})
    with runtime:
        pass

    with pytest.raises(RuntimeClosedError, match="re-entered"):
        runtime.__enter__()


def test_current_runtime_keeps_temporary_ambient_compatibility() -> None:
    outer = Runtime(language_models={"outer": LanguageEngine()})
    inner = Runtime(language_models={"inner": LanguageEngine()})

    with pytest.raises(NoActiveRuntimeError):
        current_runtime()

    with outer:
        assert current_runtime() is outer
        with inner:
            assert current_runtime() is inner
        assert current_runtime() is outer

    with pytest.raises(NoActiveRuntimeError):
        current_runtime()


def test_foreign_thread_execute_is_rejected_before_engine_touch() -> None:
    engine = LanguageEngine()
    runtime = Runtime(language_models={"chat": engine})

    with runtime:
        error = capture_thread_error(lambda: runtime.execute(LANGUAGE_REQUEST))

    assert isinstance(error, RuntimeOwnershipError)
    assert error.operation == "execute"
    assert engine.requests == []


def test_foreign_thread_close_is_rejected_before_engine_touch() -> None:
    engine = LanguageEngine()
    runtime = Runtime(language_models={"chat": engine})

    with runtime:
        error = capture_thread_error(runtime.close)
        assert isinstance(error, RuntimeOwnershipError)
        assert error.operation == "close"
        assert engine.close_count == 0

    assert engine.close_count == 1


def test_owner_thread_close_is_idempotent() -> None:
    engine = LanguageEngine()
    runtime = Runtime(language_models={"chat": engine})

    with runtime:
        runtime.close()
        runtime.close()

    runtime.close()
    assert engine.close_count == 1


def test_close_before_entry_is_idempotent_and_prevents_entry() -> None:
    engine = LanguageEngine()
    runtime = Runtime(language_models={"chat": engine})

    runtime.close()
    runtime.close()

    assert engine.close_count == 1
    with pytest.raises(RuntimeClosedError):
        runtime.__enter__()


def test_runtime_closes_each_engine_once_in_reverse_acceptance_order() -> None:
    closed: list[str] = []
    engines = (
        LanguageEngine("language-first", closed=closed),
        LanguageEngine("language-second", closed=closed),
        EmbeddingEngine("embedding-first", closed=closed),
        EmbeddingEngine("embedding-second", closed=closed),
    )
    runtime = Runtime(
        language_models={"first": engines[0], "second": engines[1]},
        embeddings={"first": engines[2], "second": engines[3]},
    )

    runtime.close()
    runtime.close()

    assert closed == [
        "embedding-second",
        "embedding-first",
        "language-second",
        "language-first",
    ]
    assert [engine.close_count for engine in engines] == [1, 1, 1, 1]


def test_runtime_attempts_all_closes_and_groups_failures() -> None:
    closed: list[str] = []
    first_failure = RuntimeError("first close failed")
    second_failure = KeyboardInterrupt("second close failed")
    first = LanguageEngine("first", closed=closed, close_error=first_failure)
    middle = LanguageEngine("middle", closed=closed)
    second = EmbeddingEngine("second", closed=closed, close_error=second_failure)
    runtime = Runtime(
        language_models={"first": first, "middle": middle},
        embeddings={"second": second},
    )

    with pytest.raises(BaseExceptionGroup, match="Runtime cleanup failed") as caught:
        runtime.close()

    assert closed == ["second", "middle", "first"]
    assert caught.value.exceptions == (second_failure, first_failure)
    runtime.close()
    assert [first.close_count, middle.close_count, second.close_count] == [1, 1, 1]


def test_body_exception_remains_primary_when_cleanup_fails() -> None:
    class BodyFailure(Exception):
        pass

    runtime = Runtime(
        language_models={
            "first": LanguageEngine(
                "first",
                close_error=RuntimeError("first cleanup failed"),
            ),
            "second": LanguageEngine(
                "second",
                close_error=RuntimeError("second cleanup failed"),
            ),
        },
    )
    body_failure = BodyFailure("body failed")

    with pytest.raises(BodyFailure) as caught, runtime:
        raise body_failure

    assert caught.value is body_failure
    assert len(caught.value.__notes__) == 2
    assert "second cleanup failed" in caught.value.__notes__[0]
    assert "first cleanup failed" in caught.value.__notes__[1]


def test_runtime_can_be_owned_entirely_by_its_entering_thread() -> None:
    runtime = Runtime(language_models={"chat": LanguageEngine()})

    def use_runtime() -> None:
        with runtime:
            assert runtime.execute(LANGUAGE_REQUEST) is LANGUAGE_RESPONSE

    error = capture_thread_error(use_runtime)

    assert error is None
