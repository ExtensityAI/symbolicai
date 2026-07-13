import importlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from symai.backend import provider_engines
from symai.backend.base import Engine
from symai.backend.engine_handle import EngineHandle
from symai.backend.settings import SYMAI_CONFIG
from symai.functional import EngineRepository


class StubEngine(Engine):
    def forward(self, *_args, **_kwargs):
        return []

    def prepare(self, argument):
        return argument


@pytest.fixture(autouse=True)
def close_repository():
    EngineRepository.close()
    yield
    EngineRepository.close()


def handle(engine: Engine, close: Callable[[], None]) -> EngineHandle:
    return EngineHandle(engine=engine, cleanup=close)


def test_engine_handle_cleanup_is_optional_and_idempotent():
    engine = StubEngine()
    closed = []
    owned = EngineHandle(engine=engine, cleanup=lambda: closed.append(engine))
    unowned = EngineHandle(engine=engine)

    assert owned.owns_resources is True
    assert unowned.owns_resources is False

    owned.close()
    owned.close()
    unowned.close()

    assert owned.owns_resources is False
    assert closed == [engine]


def test_engine_handle_concurrent_close_runs_cleanup_once():
    engine = StubEngine()
    closed = []
    owned = EngineHandle(engine=engine, cleanup=lambda: closed.append(engine))
    barrier = Barrier(8)

    def close_handle(_index: int) -> None:
        barrier.wait()
        owned.close()

    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(executor.map(close_handle, range(8)))

    assert owned.owns_resources is False
    assert closed == [engine]


def test_repository_replacement_closes_previous_handle():
    first = StubEngine()
    second = StubEngine()
    closed = []
    first_handle = handle(first, lambda: closed.append(first))
    second_handle = handle(second, lambda: closed.append(second))
    EngineRepository.register("contract-test", first_handle)

    EngineRepository.register(
        "contract-test",
        second_handle,
        allow_engine_override=True,
    )

    assert EngineRepository.get("contract-test") is second
    assert first_handle.owns_resources is False
    assert second_handle.owns_resources is True
    assert closed == [first]


def test_repository_same_engine_registration_preserves_existing_handle():
    engine = StubEngine()
    closed = []
    existing = handle(engine, lambda: closed.append(engine))
    incoming = EngineHandle(engine=engine)
    EngineRepository.register("contract-test", existing)

    EngineRepository.register(
        "contract-test",
        incoming,
        allow_engine_override=True,
    )

    assert EngineRepository.get("contract-test") is engine
    assert existing.owns_resources is True
    assert incoming.owns_resources is False
    assert closed == []


def test_repository_same_engine_adopts_cleanup_when_existing_handle_has_none():
    engine = StubEngine()
    closed = []
    owned = handle(engine, lambda: closed.append(engine))
    EngineRepository.register("contract-test", engine)

    EngineRepository.register(
        "contract-test",
        owned,
        allow_engine_override=True,
    )

    assert owned.owns_resources is True

    EngineRepository.close()

    assert owned.owns_resources is False
    assert closed == [engine]


def test_repository_close_releases_all_handles_and_detaches_engines():
    engine = StubEngine()
    closed = []
    EngineRepository.register("contract-test", handle(engine, lambda: closed.append(engine)))

    EngineRepository.close()

    assert "contract-test" not in EngineRepository.list()
    assert closed == [engine]


def test_repository_close_releases_remaining_handles_after_cleanup_failure():
    released = []

    def fail_cleanup():
        msg = "cleanup failed"
        raise RuntimeError(msg)

    failing = handle(StubEngine(), fail_cleanup)
    successful = handle(StubEngine(), lambda: released.append("successful"))
    EngineRepository.register("failing", failing)
    EngineRepository.register("successful", successful)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        EngineRepository.close()

    assert failing.owns_resources is False
    assert successful.owns_resources is False
    assert released == ["successful"]
    assert EngineRepository.list() == {}


def test_concurrent_provider_registration_closes_losing_handle(monkeypatch):
    barrier = Barrier(2)
    created = []
    closed = []

    def create_provider_engine_handle(**_kwargs):
        engine = StubEngine()
        created.append(engine)
        barrier.wait()
        return handle(engine, lambda: closed.append(engine))

    monkeypatch.setattr(
        provider_engines,
        "create_provider_engine_handle",
        create_provider_engine_handle,
    )
    monkeypatch.setitem(
        SYMAI_CONFIG,
        "NEUROSYMBOLIC_ENGINE_MODEL",
        "deepseek:deepseek-v4-flash",
    )
    monkeypatch.setitem(SYMAI_CONFIG, "NEUROSYMBOLIC_ENGINE_API_KEY", "test-key")

    with ThreadPoolExecutor(max_workers=2) as executor:
        engines = tuple(executor.map(lambda _: EngineRepository.get("neurosymbolic"), range(2)))

    assert engines[0] is engines[1]
    assert len(created) == 2
    assert len(closed) == 1
    assert engines[0] not in closed


def test_register_from_package_skips_module_with_missing_dependency(tmp_path, monkeypatch):
    """A module whose optional dependency is missing must be skipped, not abort the group."""
    pkg = tmp_path / "fake_engines_c1"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "broken_engine.py").write_text("import definitely_missing_dependency_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    package = importlib.import_module("fake_engines_c1")
    EngineRepository.register_from_package(package)
