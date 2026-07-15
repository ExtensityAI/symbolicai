from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import symai
from symai import decoding, function, loading
from symai.runtime import config, errors, models, runtime

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "symai"

PUBLIC_NAMES = [
    "AmbiguousEngineError",
    "AssistantMessage",
    "AssistantOutputMessage",
    "AuthenticationError",
    "ConstructorDecoder",
    "Content",
    "DecodeError",
    "Decoder",
    "DeveloperMessage",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingVector",
    "EngineCapabilityError",
    "EngineSpec",
    "ErrorMetadata",
    "ExecutionError",
    "FinishReason",
    "Function",
    "ImageContent",
    "ImageDetail",
    "ImplementationId",
    "InvalidResponseError",
    "JsonArray",
    "JsonEntry",
    "JsonObject",
    "JsonObjectResponseFormat",
    "JsonSchemaResponseFormat",
    "JsonValue",
    "LanguageModelOutput",
    "LanguageModelRequest",
    "LanguageModelResponse",
    "LogitBias",
    "MISSING",
    "Message",
    "MetadataLabel",
    "Missing",
    "NoActiveRuntimeError",
    "ProviderId",
    "PydanticDecoder",
    "RateLimitError",
    "RateLimitMetadata",
    "ReasoningConfig",
    "ReasoningEffort",
    "ReasoningFormat",
    "ReasoningSummary",
    "ResponseFormat",
    "ResponseMetadata",
    "Runtime",
    "RuntimeClosedError",
    "RuntimeConfig",
    "RuntimeOwnershipError",
    "SamplingConfig",
    "SymbolicAIRuntimeError",
    "SystemMessage",
    "TextContent",
    "TextDecoder",
    "TextResponseFormat",
    "TokenUsage",
    "TransportError",
    "TypeAdapterDecoder",
    "UnknownEngineError",
    "UnsupportedCapabilityError",
    "UnsupportedFeatureError",
    "UnsupportedModelError",
    "UserMessage",
    "current_runtime",
    "decode_output",
    "load_runtime",
]

DEFINING_MODULE = {
    **{name: config for name in PUBLIC_NAMES if hasattr(config, name)},
    **{name: models for name in PUBLIC_NAMES if hasattr(models, name)},
    **{name: errors for name in PUBLIC_NAMES if hasattr(errors, name)},
    **{
        name: decoding
        for name in PUBLIC_NAMES
        if hasattr(decoding, name)
    },
    "Runtime": runtime,
    "current_runtime": runtime,
    "Function": function,
    "load_runtime": loading,
}

FORBIDDEN_PUBLIC_NAMES = {
    "Argument",
    "Client",
    "EngineHandle",
    "DynamicEngine",
    "Engine",
    "EngineRepository",
    "Expression",
    "NamedEngineConfig",
    "Provider",
    "ProviderEngineConfig",
    "TransportConfig",
    "create_runtime",
    "Symbol",
    "config_manager",
    "run_server",
    "setup_wizard",
}

DELETED_FILES = {
    "runtime/factory.py",
    "components.py",
    "context.py",
    "core.py",
    "functional.py",
    "post_processors.py",
    "pre_processors.py",
    "strategy.py",
    "backend/async_bridge.py",
    "backend/engine_handle.py",
    "backend/base.py",
    "backend/chat_prompts.py",
    "backend/provider_engines.py",
    "backend/request.py",
    "backend/settings.py",
    "backend/streaming.py",
    "backend/transport.py",
    "backend/usage.py",
}

DELETED_TREES = {
    "backend/mixin",
    "backend/engines",
    "extended",
    "models",
    "server",
}

FORBIDDEN_IMPORT_PREFIXES = tuple(
    f"symai.{path.removesuffix('.py').replace('/', '.')}" for path in DELETED_FILES
) + tuple(f"symai.{path.replace('/', '.')}" for path in DELETED_TREES)
FORBIDDEN_MODULE_PATH_FRAGMENTS = FORBIDDEN_IMPORT_PREFIXES


FORBIDDEN_IDENTIFIERS = {
    "Argument",
    "CURRENT_ENGINE_VAR",
    "DynamicEngine",
    "Engine",
    "ENGINE_UNREGISTERED",
    "EngineHandle",
    "EngineRepository",
    "SYMAI_CONFIG",
    "SYMSERVER_CONFIG",
    "run_server",
    "setup_wizard",
}


def test_runtime_configuration_has_a_clean_module_cutover() -> None:
    assert config.EngineSpec.__name__ == "EngineSpec"
    assert config.RuntimeConfig.__name__ == "RuntimeConfig"
    for name in ("NamedEngineConfig", "ProviderEngineConfig", "TransportConfig"):
        assert not hasattr(models, name)


def test_public_api_is_exact_ordered_and_direct() -> None:
    assert symai.__all__ == PUBLIC_NAMES
    assert sorted(PUBLIC_NAMES) == PUBLIC_NAMES
    for name in PUBLIC_NAMES:
        assert getattr(symai, name) is getattr(DEFINING_MODULE[name], name)


def test_public_runtime_surface_exposes_errors_not_handles_or_clients() -> None:
    for name in (
        "AmbiguousEngineError",
        "EngineCapabilityError",
        "RuntimeOwnershipError",
        "UnknownEngineError",
    ):
        assert getattr(symai, name) is getattr(errors, name)

    for name in ("Client", "EngineHandle"):
        assert not hasattr(symai, name)
        assert not hasattr(runtime, name)


def test_runtime_operation_protocols_are_narrow_and_provider_neutral() -> None:
    from importlib import import_module

    engines = import_module("symai.runtime.engines")
    for protocol_name in ("LanguageModelEngine", "EmbeddingEngine"):
        protocol = getattr(engines, protocol_name)
        public_members = {name for name in vars(protocol) if not name.startswith("_")}
        assert public_members == {"close", "execute"}


def test_star_import_is_exact_and_old_names_are_absent() -> None:
    namespace: dict[str, object] = {}
    exec("from symai import *", namespace)
    assert sorted(name for name in namespace if name != "__builtins__") == PUBLIC_NAMES
    for name in FORBIDDEN_PUBLIC_NAMES:
        assert not hasattr(symai, name)


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
        "seeded_loggers_unchanged": True,
        "tree_unchanged": True,
        "warnings": [],
    }


def test_deleted_production_tree_and_adapter_inventory() -> None:
    for path in DELETED_FILES | DELETED_TREES:
        assert not (PACKAGE / path).exists(), path

    assert {path.name for path in (PACKAGE / "providers/_client").glob("*.py")} == {
        "__init__.py",
        "errors.py",
        "headers.py",
        "models.py",
    }
    assert {
        path.name for path in (PACKAGE / "providers/openai/client").glob("*.py")
    } == {
        "__init__.py",
        "_client.py",
        "embeddings.py",
        "errors.py",
        "headers.py",
        "responses.py",
        "transport.py",
    }
    assert {
        path.name for path in (PACKAGE / "providers/openai/engines").glob("*.py")
    } == {"__init__.py", "embedding.py", "responses.py"}
    for provider in ("cerebras", "deepseek"):
        assert {
            path.name for path in (PACKAGE / f"providers/{provider}/client").glob("*.py")
        } == {
            "__init__.py",
            "_client.py",
            "chat.py",
            "errors.py",
            "headers.py",
            "transport.py",
        }
        assert {
            path.name for path in (PACKAGE / f"providers/{provider}/engines").glob("*.py")
        } == {"__init__.py", "chat_completions.py"}


def _production_ast_violations(package: Path) -> list[str]:
    violations: list[str] = []
    for path in package.rglob("*.py"):
        relative = path.relative_to(package)
        package_parts = ("symai", *relative.parts[:-1])
        tree = ast.parse(path.read_text(), filename=str(relative))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_IDENTIFIERS:
                violations.append(f"{relative}:{node.lineno}: name {node.id}")
            elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_IDENTIFIERS:
                violations.append(f"{relative}:{node.lineno}: attribute {node.attr}")
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in FORBIDDEN_IDENTIFIERS:
                    violations.append(f"{relative}:{node.lineno}: definition {node.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(FORBIDDEN_IMPORT_PREFIXES):
                        violations.append(f"{relative}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    ancestor_count = node.level - 1
                    base_parts = package_parts[: len(package_parts) - ancestor_count]
                else:
                    base_parts = ()
                if node.module:
                    base_parts = (*base_parts, *node.module.split("."))
                for alias in node.names:
                    qualified_name = ".".join((*base_parts, *alias.name.split(".")))
                    if qualified_name.startswith(FORBIDDEN_IMPORT_PREFIXES):
                        violations.append(f"{relative}:{node.lineno}: from import {qualified_name}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(fragment in node.value for fragment in FORBIDDEN_MODULE_PATH_FRAGMENTS):
                    violations.append(f"{relative}:{node.lineno}: legacy module path string")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"iter_modules", "walk_packages"}
            ):
                violations.append(f"{relative}:{node.lineno}: reflective package scan")
    return violations


def test_production_ast_has_no_legacy_graph_references() -> None:
    assert _production_ast_violations(PACKAGE) == []


def test_ast_guard_reconstructs_qualified_import_from_names(tmp_path: Path) -> None:
    package = tmp_path / "symai"
    backend = package / "backend"
    runtime_package = package / "runtime"
    backend.mkdir(parents=True)
    runtime_package.mkdir()
    (package / "__init__.py").write_text("")
    (backend / "__init__.py").write_text("")
    (runtime_package / "__init__.py").write_text("")
    (package / "top_level.py").write_text(
        "from symai.backend import settings\n"
        "from symai import core, functional as legacy_functional\n"
    )
    (backend / "relative.py").write_text("from . import settings\nfrom .. import core\n")
    (runtime_package / "relative.py").write_text("from ..backend import provider_engines\n")

    violations = _production_ast_violations(package)

    for qualified_name in (
        "symai.backend.settings",
        "symai.core",
        "symai.functional",
        "symai.backend.provider_engines",
    ):
        assert any(qualified_name in violation for violation in violations)
