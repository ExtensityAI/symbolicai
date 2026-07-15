import pytest

from symai.backend.engine_handle import EngineHandle


class ArbitraryEngine:
    pass


def test_engine_handle_carries_no_cross_thread_synchronization_state() -> None:
    handle = EngineHandle(
        name="primary",
        capability="language_model",
        engine=ArbitraryEngine(),
    )

    assert not hasattr(handle, "_lock")


def test_engine_handle_tags_arbitrary_engine_and_optional_cleanup() -> None:
    engine = ArbitraryEngine()
    closed: list[ArbitraryEngine] = []
    owned = EngineHandle(
        name="primary",
        capability="language_model",
        engine=engine,
        cleanup=lambda: closed.append(engine),
    )
    borrowed = EngineHandle(
        name="vectors",
        capability="embedding",
        engine=engine,
    )

    assert owned.name == "primary"
    assert owned.capability == "language_model"
    assert owned.engine is engine
    assert owned.owns_resources is True
    assert borrowed.name == "vectors"
    assert borrowed.capability == "embedding"
    assert borrowed.engine is engine
    assert borrowed.owns_resources is False

    with pytest.raises(AttributeError):
        owned.engine = ArbitraryEngine()  # type: ignore[reportAttributeAccessIssue]
    with pytest.raises(AttributeError):
        owned.name = "changed"  # type: ignore[reportAttributeAccessIssue]
    with pytest.raises(AttributeError):
        owned.capability = "embedding"  # type: ignore[reportAttributeAccessIssue]

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

    handle = EngineHandle(
        name="primary",
        capability="language_model",
        engine=ArbitraryEngine(),
        cleanup=fail_cleanup,
    )

    with pytest.raises(RuntimeError, match="cleanup failed"):
        handle.close()

    handle.close()

    assert attempts == 1
    assert handle.owns_resources is False


def test_engine_handle_consumes_cleanup_before_invoking_it() -> None:
    observed_ownership: list[bool] = []
    handle: EngineHandle[ArbitraryEngine]

    def inspect_handle() -> None:
        observed_ownership.append(handle.owns_resources)

    handle = EngineHandle(
        name="primary",
        capability="language_model",
        engine=ArbitraryEngine(),
        cleanup=inspect_handle,
    )
    handle.close()

    assert observed_ownership == [False]
