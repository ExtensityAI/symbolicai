import ast
import importlib
import pkgutil
from pathlib import Path


def test_client_packages_do_not_import_symbolicai_backend():
    clients = importlib.import_module("symai.clients")

    for module_info in pkgutil.walk_packages(clients.__path__, f"{clients.__name__}."):
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
            imported == "symai.backend" or imported.startswith("symai.backend.")
            for imported in imported_modules
        ), module_info.name
