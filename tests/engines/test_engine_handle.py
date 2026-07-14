from threading import Barrier, Lock, Thread

import pytest

from symai.backend.engine_handle import EngineHandle


class ArbitraryEngine:
    pass


def test_engine_handle_accepts_nonlegacy_engine_and_optional_cleanup() -> None:
    engine = ArbitraryEngine()
    closed: list[ArbitraryEngine] = []
    owned = EngineHandle(engine=engine, cleanup=lambda: closed.append(engine))
    borrowed = EngineHandle(engine=engine)

    assert owned.engine is engine
    assert owned.owns_resources is True
    assert borrowed.engine is engine
    assert borrowed.owns_resources is False

    owned.close()
    owned.close()
    borrowed.close()

    assert closed == [engine]
    assert owned.owns_resources is False


def test_engine_handle_consumes_failing_cleanup_exactly_once() -> None:
    attempts = 0

    def fail_cleanup() -> None:
        nonlocal attempts
        attempts += 1
        msg = "cleanup failed"
        raise RuntimeError(msg)

    handle = EngineHandle(engine=ArbitraryEngine(), cleanup=fail_cleanup)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        handle.close()

    handle.close()

    assert attempts == 1
    assert handle.owns_resources is False


def test_engine_handle_cleanup_runs_outside_its_lock() -> None:
    observed_ownership: list[bool] = []
    handle: EngineHandle[ArbitraryEngine]

    def inspect_handle() -> None:
        observed_ownership.append(handle.owns_resources)

    handle = EngineHandle(engine=ArbitraryEngine(), cleanup=inspect_handle)
    handle.close()

    assert observed_ownership == [False]


def test_engine_handle_concurrent_close_consumes_cleanup_once() -> None:
    worker_count = 8
    barrier = Barrier(worker_count)
    attempts = 0
    attempts_lock = Lock()

    def cleanup() -> None:
        nonlocal attempts
        with attempts_lock:
            attempts += 1

    handle = EngineHandle(engine=ArbitraryEngine(), cleanup=cleanup)

    def close_handle() -> None:
        barrier.wait()
        handle.close()

    workers = [Thread(target=close_handle) for _ in range(worker_count)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)

    assert all(not worker.is_alive() for worker in workers)
    assert attempts == 1
    assert handle.owns_resources is False
