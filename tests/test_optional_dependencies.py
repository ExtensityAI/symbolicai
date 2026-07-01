import builtins
import subprocess
import sys

import numpy as np
import pytest

from symai.ops.measures import calculate_frechet_distance


def test_frechet_distance_raises_clear_error_without_scipy(monkeypatch):
    """The frechet kernel imports scipy lazily and names the extra when it's absent."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            msg = "scipy not installed"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="cluster"):
        calculate_frechet_distance(np.zeros(3), np.eye(3), np.zeros(3), np.eye(3))


def test_frechet_distance_works_with_scipy():
    d = calculate_frechet_distance(np.zeros(3), np.eye(3), np.zeros(3), np.eye(3))
    assert d == pytest.approx(0.0, abs=1e-6)


def test_import_symai_does_not_eagerly_load_sklearn_or_scipy():
    """A bare `import symai` must not pull scikit-learn or scipy (both now optional)."""
    code = (
        "import sys, symai; "
        "leaked = [m for m in ('sklearn', 'scipy') if m in sys.modules]; "
        "assert not leaked, leaked"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
