from importlib.metadata import EntryPoint

import pytest

import symai.interfaces as interfaces_module
from symai import Expression
from symai.interfaces import Interface


class SampleExpression(Expression):
    """Stand-in for a third-party plugin: a real Expression subclass, which is
    exactly what a `symai.expressions` entry point is expected to point at."""


@pytest.fixture(autouse=True)
def _isolate_eps_cache():
    """`_expression_eps` is process-cached; clear around every test so injected
    entry points never leak between tests."""
    interfaces_module._expression_eps.cache_clear()
    yield
    interfaces_module._expression_eps.cache_clear()


def install_entry_point(monkeypatch, name, target="test_interfaces:SampleExpression"):
    ep = EntryPoint(name=name, value=target, group="symai.expressions")
    monkeypatch.setattr(
        interfaces_module,
        "entry_points",
        lambda group=None: [ep] if group == "symai.expressions" else [],
        raising=False,
    )


def test_load_expression_returns_class_from_entry_point(monkeypatch):
    install_entry_point(monkeypatch, "myplugin")
    assert interfaces_module.load_expression("myplugin") is SampleExpression


def test_load_expression_returns_none_when_not_installed(monkeypatch):
    install_entry_point(monkeypatch, "myplugin")
    assert interfaces_module.load_expression("not_a_plugin") is None


def test_interface_instantiates_installed_expression_plugin(monkeypatch):
    install_entry_point(monkeypatch, "myplugin")
    assert isinstance(Interface("myplugin"), SampleExpression)


def test_bundled_interface_shadows_installed_plugin(monkeypatch):
    # entry point named "file" collides with the bundled `file` interface
    install_entry_point(monkeypatch, "file")
    result = Interface("file")
    assert not isinstance(result, SampleExpression)
    assert type(result).__name__ == "file"


def test_interface_unknown_name_raises_clear_error():
    with pytest.raises(ValueError, match="No interface or installed"):
        Interface("definitely_not_a_real_interface")
