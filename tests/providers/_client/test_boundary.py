import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_client_tier_imports_without_loading_runtime_modules():
    script = """
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))

modules = (
    "symai.providers._client.client",
    "symai.providers._client.errors",
    "symai.providers._client.headers",
    "symai.providers._client.models",
    "symai.providers._client.settings",
    "symai.providers._client.transport",
)
for module in modules:
    importlib.import_module(module)
print(json.dumps(sorted(name for name in sys.modules if name.startswith("symai.runtime"))))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(ROOT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []
