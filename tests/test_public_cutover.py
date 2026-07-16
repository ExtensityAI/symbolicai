from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import symai.runtime.config as config
import symai.runtime.errors as errors
import symai.runtime.models as models
import symai.runtime.runtime as runtime

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "symai"


def test_runtime_configuration_has_a_clean_module_cutover() -> None:
    assert config.EngineConfig.__name__ == "EngineConfig"
    assert config.RuntimeConfig.__name__ == "RuntimeConfig"
    assert not hasattr(config, "EngineSpec")
    for name in ("NamedEngineConfig", "ProviderEngineConfig", "TransportConfig"):
        assert not hasattr(models, name)


def test_canonical_modules_own_their_public_types() -> None:
    from symai.decoding import decode_text
    from symai.function import Function
    from symai.runtime.runtime import Runtime
    from symai.symbol import Symbol

    assert Function.__module__ == "symai.function"
    assert Runtime.__module__ == "symai.runtime.runtime"
    assert Symbol.__module__ == "symai.symbol"
    assert decode_text.__module__ == "symai.decoding"


def test_runtime_module_exposes_no_ambient_registry_or_provider_clients() -> None:
    for name in (
        "_CURRENT_RUNTIME",
        "current_runtime",
        "NoActiveRuntimeError",
        "Client",
        "EngineHandle",
    ):
        assert not hasattr(runtime, name)
    assert not hasattr(errors, "NoActiveRuntimeError")


def test_runtime_operation_protocols_are_narrow_and_provider_neutral() -> None:
    from importlib import import_module

    engines = import_module("symai.runtime.engines")
    for protocol_name in ("LanguageModelEngine", "EmbeddingEngine"):
        protocol = getattr(engines, protocol_name)
        public_members = {name for name in vars(protocol) if not name.startswith("_")}
        assert public_members == {"close", "execute"}


def test_the_package_root_declares_no_public_surface() -> None:
    """`import symai` binds no name at all — the subprocess test below pins that.

    `__all__` needs its own assertion: it is a dunder, so the subprocess check's
    public-name comparison would not see it.
    """
    import symai

    assert not hasattr(symai, "__all__")


def test_import_symai_is_subprocess_isolated_and_inert(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()
    script = r"""
import builtins
import json
import logging
import os
import sys
import warnings
from pathlib import Path

checkout_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(checkout_root))
temporary_root = Path.cwd().parent
def tree():
    return sorted(str(path.relative_to(temporary_root)) for path in temporary_root.rglob("*"))
seeded_logger_levels = {
    "symai": 7,
    "httpx": 11,
    "urllib3": 13,
    "requests": 17,
}
for logger_name, level in seeded_logger_levels.items():
    logging.getLogger(logger_name).setLevel(level)
before_tree = tree()
before_env = dict(os.environ)
before_root = (logging.getLogger().level, tuple(logging.getLogger().handlers))
before_loggers = {
    name: (logging.getLogger(name).level, tuple(logging.getLogger(name).handlers))
    for name in seeded_logger_levels
}
def forbidden_input(*args, **kwargs):
    raise AssertionError("import attempted to prompt")
builtins.input = forbidden_input
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import symai
result = {
    "tree_unchanged": tree() == before_tree,
    "env_unchanged": dict(os.environ) == before_env,
    "root_logger_unchanged": before_root == (logging.getLogger().level, tuple(logging.getLogger().handlers)),
    "seeded_loggers_unchanged": before_loggers == {
        name: (logging.getLogger(name).level, tuple(logging.getLogger(name).handlers))
        for name in seeded_logger_levels
    },
    "resolved_symai_file": str(Path(symai.__file__).resolve()),
    "public_names": sorted(name for name in vars(symai) if not name.startswith("_")),
    "symai_modules": sorted(
        name for name in sys.modules if name == "symai" or name.startswith("symai.")
    ),
    "warnings": [str(item.message) for item in caught],
    "forbidden_modules": sorted(name for name in sys.modules if name in {
        "symai.backend.settings", "symai.core", "symai.functional", "symai.server",
        "anthropic", "cerebras", "groq", "openai", "google.genai"
    } or name.startswith("symai.server.")),
}
print(json.dumps(result))
"""
    env = os.environ.copy()
    env["HOME"] = str(home)
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(ROOT)],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)
    resolved_symai_file = Path(observed.pop("resolved_symai_file"))
    assert resolved_symai_file.is_relative_to(ROOT / "symai")
    assert observed == {
        "env_unchanged": True,
        "forbidden_modules": [],
        "root_logger_unchanged": True,
        "public_names": [],
        "seeded_loggers_unchanged": True,
        "symai_modules": ["symai"],
        "tree_unchanged": True,
        "warnings": [],
    }


def test_the_package_contains_no_importable_namespace_husk() -> None:
    """A directory without `__init__.py` still imports, as a namespace package.

    Deleting a module's sources does not remove its directory if anything untracked
    survives inside it — a gitignored `__pycache__` outlives `git checkout`, and
    `import symai.backend` then resolves again against an empty namespace. Observed in
    practice after switching to this branch: `symai/backend` and `symai/clients` both
    came back this way.

    This asks the filesystem what Python would resolve rather than checking a list of
    module names someone remembered to write down, so it also covers husks of modules
    deleted later.
    """
    husks = [
        str(path.relative_to(PACKAGE))
        for path in PACKAGE.rglob("*")
        if path.is_dir() and path.name != "__pycache__" and not (path / "__init__.py").exists()
    ]

    assert husks == []


def test_production_never_discovers_modules_by_scanning() -> None:
    """Engines are registered explicitly; nothing may find them by walking the package.

    Reflective discovery is how the ambient registry worked before the cutover: it makes
    the set of available engines depend on what happens to be installed rather than on
    what the caller configured, and it defeats the import laziness the loaders exist for.
    Unlike the deleted names, this stays true for every future release.
    """
    offenders = [
        f"{path.relative_to(PACKAGE)}:{node.lineno}"
        for path in PACKAGE.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"iter_modules", "walk_packages"}
    ]

    assert offenders == []
