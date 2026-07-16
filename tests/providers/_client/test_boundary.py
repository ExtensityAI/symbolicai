"""Importing a client must not drag the framework in behind it.

`tests/test_layering.py` states the same rule statically and catches far more, including
imports nested inside functions. What it cannot see is an import whose *name is computed*:
`importlib.import_module(f"symai.{part}")` is not an edge in any AST. This asks the
interpreter what actually loaded, so those paths are covered too. Keep both — that one
covers breadth, this one covers what static analysis is blind to by construction.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "symai"


def _client_tier_modules() -> list[str]:
    """Every module in the client tier, discovered rather than listed.

    A hardcoded list silently stops covering whatever is added after it was written, which
    is exactly how a fourth provider would inherit none of this file's guarantees.
    """
    providers = PACKAGE / "providers"
    packages = [
        providers / "_client",
        *sorted(
            path / "client"
            for path in providers.iterdir()
            if path.is_dir() and not path.name.startswith("_") and (path / "client").is_dir()
        ),
    ]

    return sorted(
        ".".join(source.relative_to(ROOT).with_suffix("").parts)
        for package in packages
        for source in package.rglob("*.py")
        if source.stem != "__init__"
    )


def test_importing_the_client_tier_loads_no_other_symai_module() -> None:
    modules = _client_tier_modules()
    # Non-vacuity: an empty list would import nothing and trivially observe nothing.
    assert len(modules) > 3

    script = """
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))

for module in json.loads(sys.argv[2]):
    importlib.import_module(module)


def permitted(name):
    # A client may load its own tier and the packages that contain it, and nothing else.
    # The containing packages are permitted by name only: they are empty, and anything
    # they pulled in would still show up here under its own name and be rejected.
    parts = name.split(".")
    if parts[:2] == ["symai", "providers"] and len(parts) <= 3:
        return True
    if parts[:1] == ["symai"] and len(parts) == 1:
        return True
    if parts[:3] == ["symai", "providers", "_client"]:
        return True

    return len(parts) >= 4 and parts[:2] == ["symai", "providers"] and parts[3] == "client"


print(json.dumps(sorted(
    name
    for name in sys.modules
    if (name == "symai" or name.startswith("symai.")) and not permitted(name)
)))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(ROOT), json.dumps(modules)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []
