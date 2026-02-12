"""Import smoke-tests derived automatically from pyproject.toml.

The only manual artifact is _IMPORT_NAME — a mapping for packages whose
PyPI name differs from their Python import name.  Everything else is
read from pyproject.toml at collection time so the tests never drift.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest
import toml

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# PyPI name (lowercased) -> Python import name.
# Only entries where the default heuristic (lowercase, hyphens to underscores) fails.
_IMPORT_NAME = {
    "beautifulsoup4": "bs4",
    "cerebras-cloud-sdk": "cerebras.cloud.sdk",
    "firecrawl-py": "firecrawl",
    "gitpython": "git",
    "google-genai": "google.genai",
    "google_search_results": "serpapi",
    "ipython": "IPython",
    "llama-cpp-python": "llama_cpp",
    "opencv-python": "cv2",
    "openai-whisper": "whisper",
    "parallel-web": "parallel",
    "pdfminer.six": "pdfminer",
    "python-box": "box",
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
    "z3-solver": "z3",
}

_SKIP_GROUPS = {"all", "dev"}

# Captures the package name (with optional extras) before any version specifier.
_RE_PKG = re.compile(r"^([a-zA-Z0-9._-]+(?:\[[^\]]+\])?)")


def _parse_pkg(dep: str) -> str | None:
    """Convert a PEP 508 dependency string to an importable module name."""
    m = _RE_PKG.match(dep)
    if not m:
        return None
    # Strip extras like [server], [all]
    pkg = re.sub(r"\[.*\]", "", m.group(1))
    if pkg.lower() == "symbolicai":
        return None
    key = pkg.lower()
    return _IMPORT_NAME.get(key, key.replace("-", "_"))


def _load_deps():
    """Read pyproject.toml and return (mandatory, {group: [modules]})."""
    project = toml.load(_PYPROJECT)["project"]
    mandatory = [n for d in project["dependencies"] if (n := _parse_pkg(d)) is not None]
    optional = {}
    for group, deps in project.get("optional-dependencies", {}).items():
        if group in _SKIP_GROUPS:
            continue
        modules = [n for d in deps if (n := _parse_pkg(d)) is not None]
        if modules:
            optional[group] = modules
    return mandatory, optional


_mandatory, _optional = _load_deps()


@pytest.mark.parametrize("module_name", _mandatory)
def test_mandatory_dependency(module_name):
    importlib.import_module(module_name)


@pytest.mark.parametrize("group,modules", _optional.items())
def test_optional_dependency(group, modules):
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except ImportError:
            pytest.skip(f"{module_name} (group '{group}') not installed")


def test_symai():
    importlib.import_module("symai")
