import importlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from symai.backend import provider_runtime
from symai.backend.base import Engine
from symai.backend.engine_lease import EngineLease
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


def lease(engine: Engine, close: Callable[[], None]) -> EngineLease:
    return EngineLease(engine=engine, cleanup=close)


def test_engine_lease_cleanup_is_optional_and_idempotent():
    engine = StubEngine()
    closed = []
    owned = EngineLease(engine=engine, cleanup=lambda: closed.append(engine))
    unowned = EngineLease(engine=engine)

    owned.close()
    owned.close()
    unowned.close()

    assert closed == [engine]


def test_repository_replacement_closes_previous_lease():
    first = StubEngine()
    second = StubEngine()
    closed = []
    EngineRepository.register("contract-test", lease(first, lambda: closed.append(first)))

    EngineRepository.register(
        "contract-test",
        lease(second, lambda: closed.append(second)),
        allow_engine_override=True,
    )

    assert EngineRepository.get("contract-test") is second
    assert closed == [first]


def test_repository_same_engine_registration_preserves_existing_lease():
    engine = StubEngine()
    closed = []
    EngineRepository.register("contract-test", lease(engine, lambda: closed.append(engine)))

    EngineRepository.register(
        "contract-test",
        engine,
        allow_engine_override=True,
    )

    assert EngineRepository.get("contract-test") is engine
    assert closed == []


def test_repository_same_engine_adopts_cleanup_when_existing_lease_has_none():
    engine = StubEngine()
    closed = []
    EngineRepository.register("contract-test", engine)

    EngineRepository.register(
        "contract-test",
        lease(engine, lambda: closed.append(engine)),
        allow_engine_override=True,
    )
    EngineRepository.close()

    assert closed == [engine]


def test_repository_close_releases_all_leases_and_detaches_engines():
    engine = StubEngine()
    closed = []
    EngineRepository.register("contract-test", lease(engine, lambda: closed.append(engine)))

    EngineRepository.close()

    assert "contract-test" not in EngineRepository.list()
    assert closed == [engine]


def test_repository_close_releases_remaining_leases_after_cleanup_failure():
    released = []

    def fail_cleanup():
        msg = "cleanup failed"
        raise RuntimeError(msg)

    EngineRepository.register("failing", lease(StubEngine(), fail_cleanup))
    EngineRepository.register(
        "successful",
        lease(StubEngine(), lambda: released.append("successful")),
    )

    with pytest.raises(RuntimeError, match="cleanup failed"):
        EngineRepository.close()

    assert released == ["successful"]
    assert EngineRepository.list() == {}


def test_concurrent_provider_registration_closes_losing_lease(monkeypatch):
    barrier = Barrier(2)
    created = []
    closed = []

    def create_provider_engine_lease(**_kwargs):
        engine = StubEngine()
        created.append(engine)
        barrier.wait()
        return lease(engine, lambda: closed.append(engine))

    monkeypatch.setattr(
        provider_runtime,
        "create_provider_engine_lease",
        create_provider_engine_lease,
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
