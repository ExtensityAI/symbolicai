from collections.abc import Callable, Sequence
from contextvars import ContextVar, Token
from threading import Event, Thread
from typing import assert_type

import pytest

import symai.runtime.runtime as runtime_module
from symai.backend.engine_handle import EngineHandle
from symai.runtime.errors import (
    AmbiguousEngineError,
    EngineCapabilityError,
    NoActiveRuntimeError,
    RuntimeClosedError,
    RuntimeOwnershipError,
    UnknownEngineError,
    UnsupportedCapabilityError,
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
    Provider,
    ResponseMetadata,
    TextContent,
    UserMessage,
)
from symai.runtime.runtime import Runtime, current_runtime

LANGUAGE_REQUEST = LanguageModelRequest(
    messages=(UserMessage(content=(TextContent(text="question"),)),)
)
EMBEDDING_REQUEST = EmbeddingRequest(inputs=("first", "second"))
METADATA = ResponseMetadata(
    provider=Provider.OPENAI,
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
    def __init__(self, label: str = "language") -> None:
        self.label = label
        self.requests: list[LanguageModelRequest] = []

    def execute(self, request: LanguageModelRequest) -> LanguageModelResponse:
        self.requests.append(request)
        return LANGUAGE_RESPONSE


class EmbeddingEngine:
    def __init__(self, label: str = "embedding") -> None:
        self.label = label
        self.requests: list[EmbeddingRequest] = []

    def execute(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        return EMBEDDING_RESPONSE


LanguageEntry = tuple[str, LanguageEngine, Callable[[], None] | None]
EmbeddingEntry = tuple[str, EmbeddingEngine, Callable[[], None] | None]


class BlockingCurrentRuntime:
    def __init__(self) -> None:
        self._context = ContextVar[Runtime | None]("blocking_runtime", default=None)
        self.first_set_started = Event()
        self.second_set_started = Event()
        self.release_set = Event()
        self._set_calls = 0

    def get(self) -> Runtime | None:
        return self._context.get()

    def set(self, runtime: Runtime) -> Token[Runtime | None]:
        self._set_calls += 1
        if self._set_calls == 1:
            self.first_set_started.set()
        else:
            self.second_set_started.set()
        assert self.release_set.wait(timeout=2)
        return self._context.set(runtime)

    def reset(self, token: Token[Runtime | None]) -> None:
        self._context.reset(token)


def make_runtime(
    *,
    language: Sequence[LanguageEntry] = (),
    embeddings: Sequence[EmbeddingEntry] = (),
    default_language_model: str | None = None,
    default_embedding: str | None = None,
) -> Runtime:
    handles: list[EngineHandle[object]] = [
        EngineHandle(
            name=name,
            capability="language_model",
            engine=engine,
            cleanup=cleanup,
        )
        for name, engine, cleanup in language
    ]
    handles.extend(
        EngineHandle(
            name=name,
            capability="embedding",
            engine=engine,
            cleanup=cleanup,
        )
        for name, engine, cleanup in embeddings
    )
    return Runtime._from_engine_handles(
        handles,
        default_language_model=default_language_model,
        default_embedding=default_embedding,
    )


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


def test_current_runtime_requires_explicit_context() -> None:
    with pytest.raises(NoActiveRuntimeError):
        current_runtime()


def test_nested_runtimes_restore_legacy_ambient_context_before_cleanup() -> None:
    observed_during_inner_cleanup: list[Runtime] = []
    outer = make_runtime(language=(("outer", LanguageEngine(), None),))
    inner = make_runtime(
        language=(
            (
                "inner",
                LanguageEngine(),
                lambda: observed_during_inner_cleanup.append(current_runtime()),
            ),
        )
    )

    with outer:
        assert current_runtime() is outer
        with inner:
            assert current_runtime() is inner

        assert observed_during_inner_cleanup == [outer]
        assert current_runtime() is outer

    with pytest.raises(NoActiveRuntimeError):
        current_runtime()


def test_entry_claim_is_atomic_against_foreign_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocking_context = BlockingCurrentRuntime()
    monkeypatch.setattr(runtime_module, "_CURRENT_RUNTIME", blocking_context)
    cleanup_called = Event()
    close_started = Event()
    close_finished = Event()
    body_entered = Event()
    execution_finished = Event()
    release_body = Event()
    engine = LanguageEngine()
    runtime = make_runtime(language=(("chat", engine, cleanup_called.set),))
    entry_errors: list[BaseException] = []
    close_errors: list[BaseException] = []

    def enter_runtime() -> None:
        try:
            with runtime:
                body_entered.set()
                try:
                    runtime.execute(LANGUAGE_REQUEST)
                    execution_finished.set()
                finally:
                    assert release_body.wait(timeout=2)
        except BaseException as error:
            entry_errors.append(error)

    entering = Thread(target=enter_runtime)
    entering.start()
    assert blocking_context.first_set_started.wait(timeout=2)

    def close_runtime() -> None:
        close_started.set()
        try:
            runtime.close()
        except BaseException as error:
            close_errors.append(error)
        finally:
            close_finished.set()

    closing = Thread(target=close_runtime)
    closing.start()
    assert close_started.wait(timeout=2)
    close_finished_while_entry_paused = close_finished.wait(timeout=0.05)
    cleanup_while_entry_paused = cleanup_called.is_set()

    blocking_context.release_set.set()
    assert body_entered.wait(timeout=2)
    assert execution_finished.wait(timeout=2)
    assert close_finished.wait(timeout=2)

    assert close_finished_while_entry_paused is False
    assert cleanup_while_entry_paused is False
    assert len(close_errors) == 1
    assert isinstance(close_errors[0], RuntimeOwnershipError)
    assert close_errors[0].operation == "close"
    assert engine.requests == [LANGUAGE_REQUEST]
    assert cleanup_called.is_set() is False

    release_body.set()
    entering.join(timeout=2)
    closing.join(timeout=2)

    assert entering.is_alive() is False
    assert closing.is_alive() is False
    assert entry_errors == []
    assert cleanup_called.is_set()


def test_competing_entry_waits_for_atomic_owner_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocking_context = BlockingCurrentRuntime()
    monkeypatch.setattr(runtime_module, "_CURRENT_RUNTIME", blocking_context)
    cleanup_called = Event()
    first_body_entered = Event()
    first_execution_finished = Event()
    release_first_body = Event()
    second_started = Event()
    second_finished = Event()
    second_body_entered = Event()
    engine = LanguageEngine()
    runtime = make_runtime(language=(("chat", engine, cleanup_called.set),))
    first_errors: list[BaseException] = []
    second_errors: list[BaseException] = []

    def first_entry() -> None:
        try:
            with runtime:
                first_body_entered.set()
                runtime.execute(LANGUAGE_REQUEST)
                first_execution_finished.set()
                assert release_first_body.wait(timeout=2)
        except BaseException as error:
            first_errors.append(error)

    first = Thread(target=first_entry)
    first.start()
    assert blocking_context.first_set_started.wait(timeout=2)

    def second_entry() -> None:
        second_started.set()
        try:
            with runtime:
                second_body_entered.set()
        except BaseException as error:
            second_errors.append(error)
        finally:
            second_finished.set()

    second = Thread(target=second_entry)
    second.start()
    assert second_started.wait(timeout=2)
    second_set_while_claim_paused = blocking_context.second_set_started.wait(timeout=0.05)
    second_finished_while_claim_paused = second_finished.is_set()

    blocking_context.release_set.set()
    assert first_body_entered.wait(timeout=2)
    assert first_execution_finished.wait(timeout=2)
    assert second_finished.wait(timeout=2)

    assert second_set_while_claim_paused is False
    assert second_finished_while_claim_paused is False
    assert second_body_entered.is_set() is False
    assert len(second_errors) == 1
    assert isinstance(second_errors[0], RuntimeOwnershipError)
    assert second_errors[0].operation == "enter"
    assert engine.requests == [LANGUAGE_REQUEST]
    assert cleanup_called.is_set() is False

    release_first_body.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert first.is_alive() is False
    assert second.is_alive() is False
    assert first_errors == []
    assert cleanup_called.is_set()


def test_explicit_name_selects_matching_engine_over_default() -> None:
    primary = LanguageEngine("primary")
    secondary = LanguageEngine("secondary")
    runtime = make_runtime(
        language=(
            ("primary", primary, None),
            ("secondary", secondary, None),
        ),
        default_language_model="primary",
    )

    with runtime:
        assert runtime.execute(LANGUAGE_REQUEST, engine="secondary") is LANGUAGE_RESPONSE

    assert primary.requests == []
    assert secondary.requests == [LANGUAGE_REQUEST]


def test_configured_capability_default_resolves_ambiguous_omission() -> None:
    primary = LanguageEngine("primary")
    secondary = LanguageEngine("secondary")
    runtime = make_runtime(
        language=(
            ("primary", primary, None),
            ("secondary", secondary, None),
        ),
        default_language_model="secondary",
    )

    with runtime:
        assert runtime.execute(LANGUAGE_REQUEST) is LANGUAGE_RESPONSE

    assert primary.requests == []
    assert secondary.requests == [LANGUAGE_REQUEST]


def test_sole_matching_engine_is_selected_without_a_default() -> None:
    language = LanguageEngine()
    embedding = EmbeddingEngine()
    runtime = make_runtime(
        language=(("chat", language, None),),
        embeddings=(("vectors", embedding, None),),
    )

    with runtime:
        assert runtime.execute(LANGUAGE_REQUEST) is LANGUAGE_RESPONSE
        assert runtime.execute(EMBEDDING_REQUEST) is EMBEDDING_RESPONSE

    assert language.requests == [LANGUAGE_REQUEST]
    assert embedding.requests == [EMBEDDING_REQUEST]


def test_ambiguous_omission_lists_only_sorted_safe_names() -> None:
    secret = "credential-must-not-leak"
    alpha = LanguageEngine(secret)
    zeta = LanguageEngine(secret)
    runtime = make_runtime(
        language=(("zeta", zeta, None), ("alpha", alpha, None)),
    )

    with runtime, pytest.raises(AmbiguousEngineError) as caught:
        runtime.execute(LANGUAGE_REQUEST)

    assert caught.value.capability == "language_model"
    assert caught.value.engine_names == ("alpha", "zeta")
    assert secret not in str(caught.value)
    assert secret not in repr(vars(caught.value))
    assert alpha.requests == []
    assert zeta.requests == []


def test_unknown_explicit_name_is_distinct_and_touches_no_engine() -> None:
    configured = LanguageEngine("credential-must-not-leak")
    runtime = make_runtime(language=(("configured", configured, None),))

    with runtime, pytest.raises(UnknownEngineError) as caught:
        runtime.execute(LANGUAGE_REQUEST, engine="missing")

    assert caught.value.engine_name == "missing"
    assert vars(caught.value) == {"engine_name": "missing"}
    assert configured.requests == []
    assert "credential-must-not-leak" not in str(caught.value)


def test_explicit_name_with_wrong_capability_is_distinct() -> None:
    embedding = EmbeddingEngine("credential-must-not-leak")
    runtime = make_runtime(embeddings=(("vectors", embedding, None),))

    with runtime, pytest.raises(EngineCapabilityError) as caught:
        runtime.execute(LANGUAGE_REQUEST, engine="vectors")

    assert caught.value.engine_name == "vectors"
    assert caught.value.requested_capability == "language_model"
    assert caught.value.engine_capability == "embedding"
    assert embedding.requests == []
    assert "credential-must-not-leak" not in str(caught.value)


@pytest.mark.parametrize("runtime_request", [LANGUAGE_REQUEST, EMBEDDING_REQUEST])
def test_no_matching_engine_preserves_unsupported_capability(
    runtime_request: LanguageModelRequest | EmbeddingRequest,
) -> None:
    runtime = make_runtime(
        embeddings=(("vectors", EmbeddingEngine(), None),)
        if isinstance(runtime_request, LanguageModelRequest)
        else (),
        language=(("chat", LanguageEngine(), None),)
        if isinstance(runtime_request, EmbeddingRequest)
        else (),
    )

    with runtime, pytest.raises(UnsupportedCapabilityError):
        runtime.execute(runtime_request)


def test_execute_overloads_preserve_precise_response_types_with_engine_keyword() -> None:
    runtime = make_runtime(
        language=(("chat", LanguageEngine(), None),),
        embeddings=(("vectors", EmbeddingEngine(), None),),
    )

    with runtime:
        language_response = assert_type(
            runtime.execute(LANGUAGE_REQUEST, engine="chat"),
            LanguageModelResponse,
        )
        embedding_response = assert_type(
            runtime.execute(EMBEDDING_REQUEST, engine="vectors"),
            EmbeddingResponse,
        )

    assert language_response is LANGUAGE_RESPONSE
    assert embedding_response is EMBEDDING_RESPONSE


def test_capability_maps_and_defaults_are_immutable() -> None:
    runtime = make_runtime(
        language=(("chat", LanguageEngine(), None),),
        embeddings=(("vectors", EmbeddingEngine(), None),),
        default_language_model="chat",
    )

    with pytest.raises(TypeError):
        runtime._language_models["other"] = (  # pyright: ignore[reportIndexIssue]
            runtime._language_models["chat"]
        )
    with pytest.raises(TypeError):
        runtime._embeddings["other"] = (  # pyright: ignore[reportIndexIssue]
            runtime._embeddings["vectors"]
        )
    with pytest.raises(TypeError):
        runtime._defaults[  # pyright: ignore[reportIndexIssue]
            "language_model"
        ] = "other"

    runtime.close()


def test_execution_requires_successful_entry() -> None:
    engine = LanguageEngine()
    runtime = make_runtime(language=(("chat", engine, None),))

    with pytest.raises(RuntimeClosedError):
        runtime.execute(LANGUAGE_REQUEST)

    assert engine.requests == []
    runtime.close()


def test_execution_and_reentry_are_rejected_after_exit() -> None:
    runtime = make_runtime(language=(("chat", LanguageEngine(), None),))

    with runtime:
        assert runtime.execute(LANGUAGE_REQUEST) is LANGUAGE_RESPONSE

    with pytest.raises(RuntimeClosedError):
        runtime.execute(LANGUAGE_REQUEST)
    with pytest.raises(RuntimeClosedError), runtime:
        pass


def test_same_runtime_reentry_is_rejected_without_disrupting_owner_context() -> None:
    runtime = make_runtime(language=(("chat", LanguageEngine(), None),))

    with runtime:
        with pytest.raises(RuntimeClosedError), runtime:
            pass

        assert current_runtime() is runtime
        assert runtime.execute(LANGUAGE_REQUEST) is LANGUAGE_RESPONSE


def test_close_before_entry_is_idempotent_and_closes_in_reverse_order() -> None:
    closed: list[str] = []
    runtime = make_runtime(
        language=(
            ("first", LanguageEngine(), lambda: closed.append("first")),
            ("second", LanguageEngine(), lambda: closed.append("second")),
        ),
        embeddings=(("third", EmbeddingEngine(), lambda: closed.append("third")),),
    )

    runtime.close()
    runtime.close()

    assert closed == ["third", "second", "first"]
    with pytest.raises(RuntimeClosedError), runtime:
        pass


def test_owner_close_while_active_detaches_once_and_exit_remains_idempotent() -> None:
    closed: list[str] = []
    runtime = make_runtime(language=(("chat", LanguageEngine(), lambda: closed.append("chat")),))

    runtime.__enter__()
    runtime.close()
    runtime.close()
    runtime.__exit__(None, None, None)

    assert closed == ["chat"]
    with pytest.raises(NoActiveRuntimeError):
        current_runtime()


def test_close_attempts_every_cleanup_in_reverse_and_groups_failures() -> None:
    attempts: list[str] = []

    def fail(name: str, error: BaseException) -> Callable[[], None]:
        def cleanup() -> None:
            attempts.append(name)
            raise error

        return cleanup

    runtime = make_runtime(
        language=(
            ("first", LanguageEngine(), fail("first", RuntimeError("first failed"))),
            ("second", LanguageEngine(), fail("second", ValueError("second failed"))),
        ),
        embeddings=(
            (
                "third",
                EmbeddingEngine(),
                fail("third", KeyboardInterrupt("third failed")),
            ),
        ),
    )

    with pytest.raises(BaseExceptionGroup) as caught:
        runtime.close()

    assert attempts == ["third", "second", "first"]
    assert [str(error) for error in caught.value.exceptions] == [
        "third failed",
        "second failed",
        "first failed",
    ]
    runtime.close()


def test_body_exception_stays_primary_and_receives_all_cleanup_failure_notes() -> None:
    class BodyFailure(Exception):
        pass

    def fail(message: str) -> Callable[[], None]:
        def cleanup() -> None:
            raise RuntimeError(message)

        return cleanup

    runtime = make_runtime(
        language=(
            ("first", LanguageEngine(), fail("first cleanup failed")),
            ("second", LanguageEngine(), fail("second cleanup failed")),
        )
    )

    with pytest.raises(BodyFailure, match="body failed") as caught, runtime:
        msg = "body failed"
        raise BodyFailure(msg)

    assert len(caught.value.__notes__) == 2
    assert "second cleanup failed" in caught.value.__notes__[0]
    assert "first cleanup failed" in caught.value.__notes__[1]
    runtime.close()


def test_foreign_thread_execute_is_rejected_before_engine_touch() -> None:
    engine = LanguageEngine()
    runtime = make_runtime(language=(("chat", engine, None),))

    with runtime:
        error = capture_thread_error(lambda: runtime.execute(LANGUAGE_REQUEST))

        assert isinstance(error, RuntimeOwnershipError)
        assert error.operation == "execute"
        assert engine.requests == []
        assert runtime.execute(LANGUAGE_REQUEST) is LANGUAGE_RESPONSE


def test_foreign_thread_active_close_is_rejected_before_cleanup_or_detach() -> None:
    cleanup_called = Event()
    engine = LanguageEngine()
    runtime = make_runtime(language=(("chat", engine, cleanup_called.set),))

    with runtime:
        error = capture_thread_error(runtime.close)

        assert isinstance(error, RuntimeOwnershipError)
        assert error.operation == "close"
        assert cleanup_called.is_set() is False
        assert runtime.execute(LANGUAGE_REQUEST) is LANGUAGE_RESPONSE

    assert cleanup_called.is_set()


def test_foreign_thread_exit_is_rejected_before_context_or_cleanup_touch() -> None:
    cleanup_called = Event()
    runtime = make_runtime(language=(("chat", LanguageEngine(), cleanup_called.set),))

    with runtime:
        error = capture_thread_error(lambda: runtime.__exit__(None, None, None))

        assert isinstance(error, RuntimeOwnershipError)
        assert error.operation == "exit"
        assert cleanup_called.is_set() is False
        assert current_runtime() is runtime
        assert runtime.execute(LANGUAGE_REQUEST) is LANGUAGE_RESPONSE

    assert cleanup_called.is_set()


def test_close_before_entry_is_not_bound_to_construction_thread() -> None:
    cleanup_called = Event()
    runtime = make_runtime(language=(("chat", LanguageEngine(), cleanup_called.set),))

    error = capture_thread_error(runtime.close)

    assert error is None
    assert cleanup_called.is_set()


def test_independent_runtimes_execute_on_independent_owner_threads() -> None:
    first_engine = LanguageEngine("first")
    second_engine = LanguageEngine("second")
    first = make_runtime(language=(("first", first_engine, None),))
    second = make_runtime(language=(("second", second_engine, None),))
    start = Event()
    results: list[LanguageModelResponse] = []
    errors: list[BaseException] = []

    def execute(runtime: Runtime) -> None:
        try:
            assert start.wait(timeout=2)
            with runtime:
                results.append(runtime.execute(LANGUAGE_REQUEST))
        except BaseException as error:
            errors.append(error)

    threads = [Thread(target=execute, args=(first,)), Thread(target=execute, args=(second,))]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join(timeout=2)

    assert all(thread.is_alive() is False for thread in threads)
    assert errors == []
    assert results == [LANGUAGE_RESPONSE, LANGUAGE_RESPONSE]
    assert first_engine.requests == [LANGUAGE_REQUEST]
    assert second_engine.requests == [LANGUAGE_REQUEST]


def test_direct_runtime_construction_is_rejected() -> None:
    with pytest.raises(TypeError, match="create_runtime"):
        Runtime()
