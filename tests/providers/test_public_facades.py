import ast
import importlib
import pkgutil
from pathlib import Path

from symai.providers import cerebras, deepseek, openai


def test_openai_facade_exposes_concise_public_names() -> None:
    assert openai.Client.__name__ == "Client"
    assert openai.Client.__module__ == "symai.providers.openai.client._client"
    assert openai.ResponsesEngine.__name__ == "ResponsesEngine"
    assert openai.EmbeddingEngine.__name__ == "EmbeddingEngine"
    assert not any(name.startswith("OpenAI") for name in vars(openai))


def test_cerebras_facade_exposes_concise_public_names() -> None:
    assert cerebras.Client.__name__ == "Client"
    assert cerebras.Client.__module__ == "symai.providers.cerebras.client._client"
    assert cerebras.ChatCompletionsEngine.__name__ == "ChatCompletionsEngine"
    assert not any(name.startswith("Cerebras") for name in vars(cerebras))


def test_deepseek_facade_exposes_concise_public_names() -> None:
    assert deepseek.Client.__name__ == "Client"
    assert deepseek.Client.__module__ == "symai.providers.deepseek.client._client"
    assert deepseek.ChatCompletionsEngine.__name__ == "ChatCompletionsEngine"
    assert not any(name.startswith("DeepSeek") for name in vars(deepseek))


def test_provider_client_packages_do_not_import_symbolic_runtime_layers() -> None:
    forbidden_prefixes = ("symai.runtime", "symai.function", "symai.symbol")

    for provider_name in ("openai", "cerebras", "deepseek"):
        client_package = importlib.import_module(f"symai.providers.{provider_name}.client")

        for module_info in pkgutil.walk_packages(
            client_package.__path__, f"{client_package.__name__}."
        ):
            module = importlib.import_module(module_info.name)
            module_path = getattr(module, "__file__", None)
            if module_path is None or not module_path.endswith(".py"):
                continue

            tree = ast.parse(Path(module_path).read_text())
            imported_modules = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module is not None
            }
            imported_modules.update(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )

            assert not any(
                imported == prefix or imported.startswith(f"{prefix}.")
                for imported in imported_modules
                for prefix in forbidden_prefixes
            ), module_info.name
