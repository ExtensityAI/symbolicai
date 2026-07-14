from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import symai
from symai.runtime import errors, factory, models, runtime

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "symai"

PUBLIC_NAMES = [
    "AssistantMessage",
    "AssistantOutputMessage",
    "AuthenticationError",
    "Content",
    "DeveloperMessage",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingVector",
    "ErrorMetadata",
    "ExecutionError",
    "FinishReason",
    "ImageContent",
    "ImageDetail",
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
    "Message",
    "MetadataLabel",
    "NoActiveRuntimeError",
    "Provider",
    "ProviderEngineConfig",
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
    "SamplingConfig",
    "SymbolicAIRuntimeError",
    "SystemMessage",
    "TextContent",
    "TextResponseFormat",
    "TokenUsage",
    "TransportConfig",
    "TransportError",
    "UnsupportedCapabilityError",
    "UnsupportedFeatureError",
    "UnsupportedModelError",
    "UserMessage",
    "create_runtime",
    "current_runtime",
]

DEFINING_MODULE = {
    **{name: models for name in PUBLIC_NAMES if hasattr(models, name)},
    **{name: errors for name in PUBLIC_NAMES if hasattr(errors, name)},
    "Runtime": runtime,
    "current_runtime": runtime,
    "create_runtime": factory,
}

FORBIDDEN_PUBLIC_NAMES = {
    "Argument",
    "DynamicEngine",
    "Engine",
    "EngineRepository",
    "Expression",
    "Function",
    "Symbol",
    "config_manager",
    "run_server",
    "setup_wizard",
}

DELETED_FILES = {
    "context.py",
    "core.py",
    "functional.py",
    "post_processors.py",
    "pre_processors.py",
    "strategy.py",
    "backend/async_bridge.py",
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
    "backend/engines/drawing",
    "backend/engines/files",
    "backend/engines/formal",
    "backend/engines/index",
    "backend/engines/neurosymbolic",
    "backend/engines/ocr",
    "backend/engines/scrape",
    "backend/engines/search",
    "backend/engines/speech_to_text",
    "backend/engines/symbolic",
    "backend/engines/text_to_speech",
    "extended",
    "models",
    "server",
}

FORBIDDEN_IMPORT_PREFIXES = tuple(
    f"symai.{path.removesuffix('.py').replace('/', '.')}" for path in DELETED_FILES
) + tuple(f"symai.{path.replace('/', '.')}" for path in DELETED_TREES)
FORBIDDEN_MODULE_PATH_FRAGMENTS = (
    *FORBIDDEN_IMPORT_PREFIXES,
    "symai.backend.engines.engine_selenium",
)


FORBIDDEN_IDENTIFIERS = {
    "Argument",
    "CURRENT_ENGINE_VAR",
    "DynamicEngine",
    "Engine",
    "ENGINE_UNREGISTERED",
    "EngineRepository",
    "SYMAI_CONFIG",
    "SYMSERVER_CONFIG",
    "run_server",
    "setup_wizard",
}


def test_public_api_is_exact_ordered_and_direct() -> None:
    assert symai.__all__ == PUBLIC_NAMES
    assert sorted(PUBLIC_NAMES) == PUBLIC_NAMES
    for name in PUBLIC_NAMES:
        assert getattr(symai, name) is getattr(DEFINING_MODULE[name], name)


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

root = Path.cwd().parent
def tree():
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))
before_tree = tree()
before_env = dict(os.environ)
before_root = (logging.getLogger().level, tuple(logging.getLogger().handlers))
before_symai = (logging.getLogger("symai").level, tuple(logging.getLogger("symai").handlers))
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
    "symai_logger_unchanged": before_symai == (logging.getLogger("symai").level, tuple(logging.getLogger("symai").handlers)),
    "warnings": [str(item.message) for item in caught],
    "forbidden_modules": sorted(name for name in sys.modules if name in {
        "symai.backend.settings", "symai.core", "symai.functional", "symai.server",
        "anthropic", "cerebras", "groq", "openai", "google.genai"
    } or name.startswith("symai.server.")),
}
print(json.dumps(result))
"""
    env = os.environ.copy()
    env.update({"HOME": str(home), "PYTHONPATH": str(ROOT)})
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)
    assert observed == {
        "env_unchanged": True,
        "forbidden_modules": [],
        "root_logger_unchanged": True,
        "symai_logger_unchanged": True,
        "tree_unchanged": True,
        "warnings": [],
    }


def test_deleted_production_tree_and_adapter_inventory() -> None:
    for path in DELETED_FILES | DELETED_TREES:
        assert not (PACKAGE / path).exists(), path

    assert {path.name for path in (PACKAGE / "backend/engines/language_model").glob("*.py")} == {
        "__init__.py",
        "cerebras.py",
        "deepseek.py",
        "openai.py",
    }
    assert {path.name for path in (PACKAGE / "backend/engines/embedding").glob("*.py")} == {
        "__init__.py",
        "openai.py",
    }


def test_production_ast_has_no_legacy_graph_references() -> None:
    violations: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE)
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
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(FORBIDDEN_IMPORT_PREFIXES):
                    violations.append(f"{relative}:{node.lineno}: from {node.module}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(fragment in node.value for fragment in FORBIDDEN_MODULE_PATH_FRAGMENTS):
                    violations.append(f"{relative}:{node.lineno}: legacy module path string")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"iter_modules", "walk_packages"}
            ):
                violations.append(f"{relative}:{node.lineno}: reflective package scan")
    assert violations == []
