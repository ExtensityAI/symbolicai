"""Guard the packaging contract (FIXPLAN §12).

A wheel build and clean-environment install belong in CI, but the failures they catch
are cheap to catch here: an import of a module we never declared, a dependency we
declare but no longer use, and a version that disagrees with itself.
"""

import ast
import sys
import tomllib
from importlib.metadata import requires, version
from pathlib import Path

import pytest

import symai

ROOT = Path(__file__).resolve().parents[1]
DISTRIBUTION = "symbolicai"

# `pydantic_core` is imported directly for the `PydanticUndefined` sentinel, which pydantic
# does not re-export. It is deliberately not declared on its own: pydantic pins an exact
# pydantic-core version, so any range we declared could only ever conflict with that pin.
# The guarantee this relies on is proved below rather than assumed.
_PROVIDED_BY = {"pydantic_core": "pydantic"}


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _declared_dependencies() -> set[str]:
    requirements = _pyproject()["project"]["dependencies"]
    return {
        requirement.split(">=")[0].split("<=")[0].split("==")[0].split("[")[0].strip()
        for requirement in requirements
    }


def _imported_top_level_modules() -> set[str]:
    """Every top-level module `symai` imports, from its own source."""
    modules: set[str] = set()
    for path in (ROOT / "symai").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])

    return modules


def test_every_third_party_import_is_a_declared_dependency() -> None:
    """An undeclared dependency only fails once someone installs the wheel."""
    declared = _declared_dependencies()
    external = {
        module
        for module in _imported_top_level_modules()
        if module != "symai" and module not in sys.stdlib_module_names
    }

    undeclared = {module for module in external if _PROVIDED_BY.get(module, module) not in declared}

    assert undeclared == set()


def test_pydantic_core_is_guaranteed_by_the_declared_pydantic_dependency() -> None:
    """Prove the `_PROVIDED_BY` exemption instead of trusting it.

    symai imports `pydantic_core` without declaring it, which is only safe for as long as
    the pydantic distribution itself requires it.
    """
    required = {
        requirement.split(";")[0].split("==")[0].split(">=")[0].split("(")[0].strip()
        for requirement in requires("pydantic") or ()
    }

    assert "pydantic-core" in required


def test_every_declared_dependency_is_actually_imported() -> None:
    """A dependency no longer used by the shipped surface must not be imposed on users."""
    imported = {_PROVIDED_BY.get(module, module) for module in _imported_top_level_modules()}

    unused = _declared_dependencies() - imported

    assert unused == set()


def test_version_metadata_agrees_across_every_exposed_source() -> None:
    """`pyproject.toml` is the single source; nothing may restate it inconsistently."""
    declared = _pyproject()["project"]["version"]

    assert version(DISTRIBUTION) == declared
    assert symai.__version__ == declared


def test_the_release_is_a_new_major_version() -> None:
    """FIXPLAN §12: the redesign ships as a new major, not as a minor of the old surface."""
    major = int(_pyproject()["project"]["version"].split(".")[0])

    assert major >= 2


def test_the_version_hook_does_not_swallow_unknown_attributes() -> None:
    """A module `__getattr__` that returns for everything would fake any public name.

    Import inertness itself is covered by `test_public_cutover`'s subprocess test; it
    cannot be asserted here, because any other test importing a `symai` submodule binds
    it as an attribute of the package.
    """
    with pytest.raises(AttributeError):
        _ = symai.does_not_exist  # pyright: ignore[reportAttributeAccessIssue]
