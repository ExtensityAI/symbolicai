"""Invariant: symai.utils.Extra stays in bijective sync with pyproject's feature extras."""

from __future__ import annotations

import tomllib
from pathlib import Path

from symai.utils import Extra

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

# `all` aggregates the other extras and `dev` is tooling — neither is a feature extra a
# dependency guard would name (mirrors tests/test_imports._SKIP_GROUPS).
_NON_FEATURE_EXTRAS = {"all", "dev"}


def _pyproject_feature_extras() -> set[str]:
    with _PYPROJECT.open("rb") as fh:
        project = tomllib.load(fh)["project"]

    return set(project.get("optional-dependencies", {})) - _NON_FEATURE_EXTRAS


def test_every_extra_member_is_declared_in_pyproject():
    unknown = {e.value for e in Extra} - _pyproject_feature_extras()
    assert not unknown, f"Extra members not declared in pyproject extras: {sorted(unknown)}"


def test_every_pyproject_feature_extra_has_an_extra_member():
    missing = _pyproject_feature_extras() - {e.value for e in Extra}
    assert not missing, f"pyproject feature extras missing from Extra enum: {sorted(missing)}"
