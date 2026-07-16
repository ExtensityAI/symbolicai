"""Check the documented snippets against the surface the library actually ships.

Nothing else in the suite reads the docs, so a snippet that no longer imports or runs will
not fail anywhere else. Every check here derives what is valid from the current surface,
which is what keeps them useful past any one release.
"""

import ast
import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs" / "source").glob("*.md"))]
_PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _python_blocks(path: Path) -> list[str]:
    return _PYTHON_BLOCK.findall(path.read_text())


def _documents() -> list[pytest.param]:
    return [pytest.param(path, id=str(path.relative_to(ROOT))) for path in DOCS]


@pytest.mark.parametrize("path", _documents())
def test_documented_python_blocks_are_syntactically_valid(path: Path) -> None:
    for index, block in enumerate(_python_blocks(path)):
        compile(block, f"{path.name}:block-{index}", "exec")


@pytest.mark.parametrize("path", _documents())
def test_documented_imports_resolve_against_the_shipped_surface(path: Path) -> None:
    resolved = 0
    for block in _python_blocks(path):
        for node in ast.walk(ast.parse(block)):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if not node.module.startswith("symai"):
                continue

            module = importlib.import_module(node.module)
            for alias in node.names:
                assert hasattr(module, alias.name), (
                    f"{path.name} documents {node.module}.{alias.name}, which is not shipped"
                )
                resolved += 1

    assert resolved or not _python_blocks(path)


def test_documented_configuration_and_handles_work_end_to_end() -> None:
    """Execute the documented setup path: IDs, setting keys, and handle accessors.

    Import checks alone would not catch a wrong implementation ID or setting key. No
    network call happens here — only configuration, construction, and selection.
    """
    from pydantic import SecretStr

    from symai.loading import load_runtime
    from symai.runtime.config import EngineConfig, RuntimeConfig

    key = SecretStr("test-key")
    config = RuntimeConfig(
        language_models={
            "chat": EngineConfig(
                implementation="openai:responses",
                settings={"api_key": key, "model": "gpt-5.4"},
            ),
            "tenant-b": EngineConfig(
                implementation="cerebras:chat-completions",
                settings={"api_key": key, "model": "gpt-oss-120b"},
            ),
        },
        embeddings={
            "vectors": EngineConfig(
                implementation="openai:embeddings",
                settings={"api_key": key, "model": "text-embedding-3-small"},
            )
        },
    )

    with load_runtime(config) as runtime:
        assert runtime.language_model("chat") == runtime.language_model("chat")
        assert runtime.language_model("tenant-b") != runtime.language_model("chat")
        assert runtime.embedding("vectors") is not None
