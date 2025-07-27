"""Unit tests for the `naive_webscraping` interface.

The real `naive_webscraping` interface eventually spins up the full SymbolicAI
execution chain which includes hitting the network via the `RequestsEngine`.
For an isolated unit-test we don't want to make outgoing HTTP requests.  We
therefore monkey-patch the `symai.core.scrape` decorator used by
`naive_webscraping.__call__` so that we can intercept the call **before** it
reaches the engine layer.  This lets us assert that the decorator receives the
expected arguments (the URL and any keyword arguments) and that the wrapper
returns its result back to the caller unchanged.

Only the behaviour that lives inside `naive_webscraping` is tested here – we
deliberately do **not** test the network code sitting behind the web-scraping
engine.
"""

# mypy: ignore-errors
# ruff: noqa: D401, ANN001, ANN003, D103 – tests are intentionally lightweight

from __future__ import annotations

import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Dep-dependency stubs
# ---------------------------------------------------------------------------
# ``naive_webscraping`` depends on ``symai.backend.engines.webscraping.
# engine_requests`` which in turn imports external libraries that may not be
# available in the test environment (e.g. ``trafilatura`` and ``bs4``).
# We pre-emptively create minimal stub modules so the import succeeds without
# requiring those heavy dependencies.

_stubbed_modules: dict[str, types.ModuleType] = {}


def _install_stub(name: str, attrs: dict[str, object] | None = None) -> None:  # noqa: D401
    """Install a dummy module into :pydata:`sys.modules`."""

    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    _stubbed_modules[name] = mod


# Minimal stubs for libraries that might not be installed in the CI runtime.
# Install stubs only when the real modules are absent so as not to interfere
# with environments where they *are* available.

if "trafilatura" not in sys.modules:
    _install_stub(
        "trafilatura",
        {
            "load_html": lambda *_, **__: "",  # type: ignore[override]
            "extract": lambda *_, **__: "",  # type: ignore[override]
        },
    )

# BeautifulSoup is used only for parsing meta refresh in the real engine.  A
# skeletal class implementation is sufficient for import-time purposes.

if "bs4" not in sys.modules:

    class _DummySoup:  # noqa: D401
        def __init__(self, *_, **__):  # noqa: D401
            pass

    _install_stub("bs4", {"BeautifulSoup": _DummySoup})

# ``requests`` is usually available but we stub it if missing to guarantee an
# import-safe environment.

if "requests" not in sys.modules:
    _install_stub("requests")

# Now that the external requirements are stubbed we can safely import the code
# under test.

from symai.extended.interfaces.naive_webscraping import naive_webscraping  # noqa: E402
from symai import core  # noqa: E402


@pytest.fixture()
def patch_core_scrape(monkeypatch):
    """Patch ``core.scrape`` with a fake decorator for the duration of a test.

    The fake decorator records the arguments it was called with so the test can
    assert on them.  It also returns a wrapper that, when invoked, returns a
    sentinel value so that the outer interface call has something deterministic
    to propagate.
    """

    call_recorder = {}

    def fake_scrape(url: str, **scrape_kwargs):  # type: ignore[override]
        """Replacement for :pyfunc:`symai.core.scrape`.  Acts like the real
        decorator but avoids any further SymbolicAI processing.
        """

        # Record the arguments with which the decorator was applied so that the
        # test can make assertions later on.
        call_recorder["url"] = url
        call_recorder["kwargs"] = scrape_kwargs

        # The real ``core.scrape`` returns a decorator; so does our fake.
        def decorator(func):  # noqa: D401 – short lambda-style docstring not needed
            # We need to return a *wrapper* that will be executed by
            # ``naive_webscraping.__call__``.  This wrapper should accept the
            # ``self`` instance and propagate any positional/keyword arguments
            # but ultimately just return a deterministic sentinel value.

            def wrapper(instance, *args, **kwargs):  # noqa: D401
                # Record the instance the wrapper gets called with – this should
                # be the exact ``naive_webscraping`` object created in the test.
                call_recorder["instance"] = instance
                call_recorder["wrapper_args"] = args
                call_recorder["wrapper_kwargs"] = kwargs

                # Return a unique, easily-identifiable sentinel value so the
                # outer call can assert on it.
                return "<mock-scrape-result>"

            return wrapper

        return decorator

    # Monkey-patch ``core.scrape`` for the duration of the test.
    monkeypatch.setattr(core, "scrape", fake_scrape, raising=True)

    # Provide access to the shared state inside the test.
    yield call_recorder


def test_naive_webscraping_delegates_to_core_scrape(patch_core_scrape):
    """``naive_webscraping`` must delegate all work to ``core.scrape``.

    The test verifies that:

    1. The URL and any additional keyword arguments are forwarded verbatim to
       ``core.scrape``.
    2. The *instance* that eventually reaches the wrapper returned by
       ``core.scrape`` is the same ``naive_webscraping`` object created in the
       test (this ensures the `self` propagation is correct).
    3. Whatever the wrapper returns is propagated back to the original caller
       unchanged.
    """

    url = "https://example.com"
    extra_kwargs = {"output_format": "text"}

    scraper = naive_webscraping()

    # Call the interface – this should invoke our patched ``core.scrape``.
    result = scraper(url, **extra_kwargs)

    # 1. The decorator received the expected URL and keyword arguments.
    assert patch_core_scrape["url"] == url
    assert patch_core_scrape["kwargs"] == extra_kwargs

    # 2. The wrapper should have been called with *this* instance.
    assert patch_core_scrape["instance"] is scraper

    # 3. The result returned by the wrapper is relayed back intact.
    assert result == "<mock-scrape-result>"
