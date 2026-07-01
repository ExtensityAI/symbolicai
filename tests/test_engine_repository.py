import importlib

from symai.functional import EngineRepository


def test_register_from_package_skips_module_with_missing_dependency(tmp_path, monkeypatch):
    """A module whose optional dependency is missing must be skipped, not abort the group."""
    pkg = tmp_path / "fake_engines_c1"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "broken_engine.py").write_text("import definitely_missing_dependency_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    package = importlib.import_module("fake_engines_c1")
    EngineRepository.register_from_package(package)
