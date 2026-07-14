from collections.abc import Callable
from contextvars import ContextVar, Token
from threading import Event, Lock, Thread
from typing import assert_type

import pytest

import symai.runtime.runtime as runtime_module
from symai.backend.engine_handle import EngineHandle
from symai.runtime.errors import (
    NoActiveRuntimeError,
    RuntimeClosedError,
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
METADATA = ResponseMetadata(provider=Provider.OPENAI, model="test-model", status_code=200)
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
        execute: Callable[[LanguageModelRequest], LanguageModelResponse] | None = None,
    ) -> None:
        self.requests: list[LanguageModelRequest] = []
        self._execute = execute

    def execute(self, request: LanguageModelRequest) -> LanguageModelResponse:
        self.requests.append(request)
        if self._execute is not None:
            return self._execute(request)
        return LANGUAGE_RESPONSE


class EmbeddingEngine:
    def __init__(self) -> None:
        self.requests: list[EmbeddingRequest] = []

    def execute(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        return EMBEDDING_RESPONSE


def make_runtime(
    *,
    language_engine: LanguageEngine | None = None,
    language_cleanup: Callable[[], None] | None = None,
    embedding_engine: EmbeddingEngine | None = None,
    embedding_cleanup: Callable[[], None] | None = None,
) -> Runtime:
    language_handle = (
        EngineHandle(language_engine, language_cleanup) if language_engine is not None else None
    )
    embedding_handle = (
        EngineHandle(embedding_engine, embedding_cleanup) if embedding_engine is not None else None
    )
    return Runtime(language_model=language_handle, embedding=embedding_handle)


def test_current_runtime_requires_explicit_context() -> None:
    with pytest.raises(NoActiveRuntimeError):
        current_runtime()


def test_entry_installs_context_before_active_state_is_visible_to_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_started = Event()
    release_set = Event()
    close_started = Event()
    cleanup_started = Event()
    body_entered = Event()
    release_body = Event()

    class BlockingCurrentRuntime:
        def __init__(self) -> None:
            self._context = ContextVar[Runtime | None]("blocking_runtime", default=None)

        def get(self) -> Runtime | None:
            return self._context.get()

        def set(self, runtime: Runtime) -> Token[Runtime | None]:
            set_started.set()
            assert release_set.wait(timeout=2)
            return self._context.set(runtime)

        def reset(self, token: Token[Runtime | None]) -> None:
            self._context.reset(token)

    monkeypatch.setattr(runtime_module, "_CURRENT_RUNTIME", BlockingCurrentRuntime())
    runtime = make_runtime(
        language_engine=LanguageEngine(),
        language_cleanup=cleanup_started.set,
    )

    def enter_runtime() -> None:
        with runtime:
            body_entered.set()
            assert release_body.wait(timeout=2)

    entering = Thread(target=enter_runtime)
    entering.start()
    assert set_started.wait(timeout=2)

    def close_runtime() -> None:
        close_started.set()
        runtime.close()

    closing = Thread(target=close_runtime)
    closing.start()
    assert close_started.wait(timeout=2)
    assert cleanup_started.wait(timeout=0.05) is False

    release_set.set()
    assert body_entered.wait(timeout=2)
    closing.join(timeout=2)
    release_body.set()
    entering.join(timeout=2)

    assert closing.is_alive() is False
    assert entering.is_alive() is False
    assert cleanup_started.is_set()


def test_different_nested_runtimes_restore_exact_previous_context_before_cleanup() -> None:
    observed_during_inner_cleanup: list[Runtime] = []
    outer = make_runtime(language_engine=LanguageEngine())
    inner = make_runtime(
        language_engine=LanguageEngine(),
        language_cleanup=lambda: observed_during_inner_cleanup.append(current_runtime()),
    )

    with outer:
        assert current_runtime() is outer
        with inner:
            assert current_runtime() is inner

        assert observed_during_inner_cleanup == [outer]
        assert current_runtime() is outer

    with pytest.raises(NoActiveRuntimeError):
        current_runtime()


def test_context_is_restored_before_outer_runtime_cleanup() -> None:
    no_runtime_during_cleanup = False

    def cleanup() -> None:
        nonlocal no_runtime_during_cleanup
        with pytest.raises(NoActiveRuntimeError):
            current_runtime()
        no_runtime_during_cleanup = True

    runtime = make_runtime(language_engine=LanguageEngine(), language_cleanup=cleanup)

    with runtime:
        assert current_runtime() is runtime

    assert no_runtime_during_cleanup is True


def test_same_runtime_reentry_is_rejected_without_disrupting_outer_context() -> None:
    engine = LanguageEngine()
    runtime = make_runtime(language_engine=engine)

    with runtime:
        with pytest.raises(RuntimeClosedError), runtime:
            pass

        assert current_runtime() is runtime
        assert runtime.execute(LANGUAGE_REQUEST) is LANGUAGE_RESPONSE


def test_execution_requires_active_runtime_context() -> None:
    runtime = make_runtime(language_engine=LanguageEngine())

    with pytest.raises(RuntimeClosedError):
        runtime.execute(LANGUAGE_REQUEST)


def test_execution_and_entry_are_rejected_after_context_close() -> None:
    runtime = make_runtime(language_engine=LanguageEngine())

    with runtime:
        assert runtime.execute(LANGUAGE_REQUEST) is LANGUAGE_RESPONSE

    with pytest.raises(RuntimeClosedError):
        runtime.execute(LANGUAGE_REQUEST)
    with pytest.raises(RuntimeClosedError), runtime:
        pass


def test_close_from_created_cleans_handles_once_and_rejects_later_entry() -> None:
    closed: list[str] = []
    runtime = make_runtime(
        language_engine=LanguageEngine(),
        language_cleanup=lambda: closed.append("language"),
        embedding_engine=EmbeddingEngine(),
        embedding_cleanup=lambda: closed.append("embedding"),
    )

    runtime.close()
    runtime.close()

    assert closed == ["language", "embedding"]
    with pytest.raises(RuntimeClosedError), runtime:
        pass


def test_execute_dispatches_concrete_request_types_with_precise_responses() -> None:
    language_engine = LanguageEngine()
    embedding_engine = EmbeddingEngine()
    runtime = make_runtime(
        language_engine=language_engine,
        embedding_engine=embedding_engine,
    )

    with runtime:
        language_response = assert_type(runtime.execute(LANGUAGE_REQUEST), LanguageModelResponse)
        embedding_response = assert_type(runtime.execute(EMBEDDING_REQUEST), EmbeddingResponse)

    assert language_response is LANGUAGE_RESPONSE
    assert embedding_response is EMBEDDING_RESPONSE
    assert language_engine.requests == [LANGUAGE_REQUEST]
    assert embedding_engine.requests == [EMBEDDING_REQUEST]


@pytest.mark.parametrize("runtime_request", [LANGUAGE_REQUEST, EMBEDDING_REQUEST])
def test_execute_rejects_missing_capability(
    runtime_request: LanguageModelRequest | EmbeddingRequest,
) -> None:
    runtime = Runtime()

    with runtime, pytest.raises(UnsupportedCapabilityError):
        runtime.execute(runtime_request)


def test_close_waits_for_in_flight_execution() -> None:
    execution_started = Event()
    release_execution = Event()
    close_started = Event()
    close_finished = Event()

    def execute(_request: LanguageModelRequest) -> LanguageModelResponse:
        execution_started.set()
        assert release_execution.wait(timeout=2)
        return LANGUAGE_RESPONSE

    runtime = make_runtime(language_engine=LanguageEngine(execute))
    runtime.__enter__()
    execution_result: list[LanguageModelResponse] = []

    execution_thread = Thread(
        target=lambda: execution_result.append(runtime.execute(LANGUAGE_REQUEST))
    )

    def close_runtime() -> None:
        close_started.set()
        runtime.close()
        close_finished.set()

    close_thread = Thread(target=close_runtime)
    execution_thread.start()
    assert execution_started.wait(timeout=2)
    close_thread.start()
    assert close_started.wait(timeout=2)
    assert close_finished.wait(timeout=0.05) is False

    release_execution.set()
    execution_thread.join(timeout=2)
    close_thread.join(timeout=2)
    runtime.__exit__(None, None, None)

    assert execution_thread.is_alive() is False
    assert close_thread.is_alive() is False
    assert execution_result == [LANGUAGE_RESPONSE]
    assert close_finished.is_set()


def test_new_work_is_rejected_while_cleanup_runs_outside_runtime_lock() -> None:
    cleanup_started = Event()
    release_cleanup = Event()
    close_finished = Event()

    def cleanup() -> None:
        cleanup_started.set()
        assert release_cleanup.wait(timeout=2)

    runtime = make_runtime(
        language_engine=LanguageEngine(),
        language_cleanup=cleanup,
    )
    runtime.__enter__()

    def close_runtime() -> None:
        runtime.close()
        close_finished.set()

    close_thread = Thread(target=close_runtime)
    close_thread.start()
    assert cleanup_started.wait(timeout=2)

    rejected = Event()

    def execute_during_cleanup() -> None:
        with pytest.raises(RuntimeClosedError):
            runtime.execute(LANGUAGE_REQUEST)
        rejected.set()

    execution_thread = Thread(target=execute_during_cleanup)
    execution_thread.start()

    assert rejected.wait(timeout=0.5)
    assert close_finished.is_set() is False

    release_cleanup.set()
    execution_thread.join(timeout=2)
    close_thread.join(timeout=2)
    runtime.__exit__(None, None, None)

    assert execution_thread.is_alive() is False
    assert close_thread.is_alive() is False


def test_concurrent_close_runs_cleanup_exactly_once() -> None:
    cleanup_started = Event()
    release_cleanup = Event()
    cleanup_attempts = 0
    attempts_lock = Lock()

    def cleanup() -> None:
        nonlocal cleanup_attempts
        with attempts_lock:
            cleanup_attempts += 1
        cleanup_started.set()
        assert release_cleanup.wait(timeout=2)

    runtime = make_runtime(
        language_engine=LanguageEngine(),
        language_cleanup=cleanup,
    )
    owner = Thread(target=runtime.close)
    owner.start()
    assert cleanup_started.wait(timeout=2)

    waiter_count = 6
    waiter_finished = [Event() for _ in range(waiter_count)]
    waiters = [
        Thread(target=lambda done=done: (runtime.close(), done.set())) for done in waiter_finished
    ]
    for waiter in waiters:
        waiter.start()

    assert all(done.wait(timeout=0.05) is False for done in waiter_finished)

    release_cleanup.set()
    owner.join(timeout=2)
    for waiter in waiters:
        waiter.join(timeout=2)

    assert owner.is_alive() is False
    assert all(not waiter.is_alive() for waiter in waiters)
    assert all(done.is_set() for done in waiter_finished)
    assert cleanup_attempts == 1


def test_concurrent_closer_waits_for_failed_owner_cleanup_to_finish() -> None:
    cleanup_started = Event()
    release_cleanup = Event()
    owner_failures: list[BaseException] = []
    waiter_failures: list[BaseException] = []
    waiter_finished = Event()

    def cleanup() -> None:
        cleanup_started.set()
        assert release_cleanup.wait(timeout=2)
        msg = "owner cleanup failed"
        raise RuntimeError(msg)

    runtime = make_runtime(
        language_engine=LanguageEngine(),
        language_cleanup=cleanup,
    )

    def owner_close() -> None:
        try:
            runtime.close()
        except BaseException as error:
            owner_failures.append(error)

    owner = Thread(target=owner_close)
    owner.start()
    assert cleanup_started.wait(timeout=2)

    def waiter_close() -> None:
        try:
            runtime.close()
        except BaseException as error:
            waiter_failures.append(error)
        finally:
            waiter_finished.set()

    waiter = Thread(target=waiter_close)
    waiter.start()
    assert waiter_finished.wait(timeout=0.05) is False

    release_cleanup.set()
    owner.join(timeout=2)
    waiter.join(timeout=2)

    assert owner.is_alive() is False
    assert waiter.is_alive() is False
    assert len(owner_failures) == 1
    assert isinstance(owner_failures[0], BaseExceptionGroup)
    assert waiter_failures == []
    assert waiter_finished.is_set()

    runtime.close()


def test_close_attempts_every_cleanup_and_groups_all_failures() -> None:
    attempts: list[str] = []

    def fail(name: str, error: BaseException) -> Callable[[], None]:
        def cleanup() -> None:
            attempts.append(name)
            raise error

        return cleanup

    runtime = make_runtime(
        language_engine=LanguageEngine(),
        language_cleanup=fail("language", RuntimeError("language failed")),
        embedding_engine=EmbeddingEngine(),
        embedding_cleanup=fail("embedding", KeyboardInterrupt("embedding failed")),
    )

    with pytest.raises(BaseExceptionGroup) as caught:
        runtime.close()

    assert attempts == ["language", "embedding"]
    assert [str(error) for error in caught.value.exceptions] == [
        "language failed",
        "embedding failed",
    ]

    runtime.close()


def test_body_exception_stays_primary_and_cleanup_failures_become_notes() -> None:
    class BodyFailure(Exception):
        pass

    def cleanup() -> None:
        msg = "cleanup failed"
        raise RuntimeError(msg)

    runtime = make_runtime(
        language_engine=LanguageEngine(),
        language_cleanup=cleanup,
    )

    with pytest.raises(BodyFailure, match="body failed") as caught, runtime:
        msg = "body failed"
        raise BodyFailure(msg)

    notes = caught.value.__notes__
    assert len(notes) == 1
    assert "cleanup failed" in notes[0]

    with pytest.raises(NoActiveRuntimeError):
        current_runtime()
    runtime.close()
