import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module_name",
    (
        "symai.loading",
        "symai.providers.openai.loading",
        "symai.providers.cerebras.loading",
        "symai.providers.deepseek.loading",
    ),
)
def test_loading_modules_do_not_import_provider_clients_or_engines(
    module_name: str,
) -> None:
    script = """
import importlib
import sys

importlib.import_module(sys.argv[1])

forbidden = (
    "symai.providers.openai.client",
    "symai.providers.openai.engines",
    "symai.providers.cerebras.client",
    "symai.providers.cerebras.engines",
    "symai.providers.deepseek.client",
    "symai.providers.deepseek.engines",
)
loaded = tuple(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
)
if loaded:
    raise AssertionError(f"heavy provider modules imported: {loaded}")
"""

    subprocess.run([sys.executable, "-c", script, module_name], check=True)


@pytest.mark.parametrize(
    "module_name",
    (
        "symai.contract.models",
        "symai.contract.validation",
        "symai.contract.remedy",
        "symai.contract.contract",
        "symai.contract.decorator",
    ),
)
def test_contract_modules_do_not_import_provider_implementations(
    module_name: str,
) -> None:
    script = """
import importlib
import sys

importlib.import_module(sys.argv[1])

loaded = tuple(name for name in sys.modules if name.startswith("symai.providers."))
if loaded:
    raise AssertionError(f"provider modules imported: {loaded}")
"""

    subprocess.run([sys.executable, "-c", script, module_name], check=True)
