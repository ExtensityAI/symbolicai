import logging
from functools import cache
from importlib.metadata import entry_points
from pydoc import locate

from .imports import Import
from .symbol import Expression

logger = logging.getLogger(__name__)


@cache
def _expression_eps():
    """Installed `symai.expressions` plugins, keyed by entry-point name.

    Cached because `entry_points()` scans all distribution metadata and
    `Interface(...)` is a hot path. Names are a flat namespace across every
    installed distribution, so duplicates are resolved first-wins, loudly."""
    eps = {}
    for ep in entry_points(group="symai.expressions"):
        if ep.name in eps:
            logger.warning("Duplicate symai.expressions plugin %r; keeping first.", ep.name)
            continue

        eps[ep.name] = ep
    return eps


def load_expression(name: str):
    """Resolve an installed `symai.expressions` plugin to its class, or None.

    None means "not installed under that name" — callers fall back to the
    bundled `symai.extended.interfaces` registry."""
    ep = _expression_eps().get(name)
    if ep is None:
        return None

    return ep.load()


class Interface(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def __new__(cls, module: str, *args, **kwargs):
        module = str(module)
        # if `/` in module, assume github repo; else assume local module
        if "/" in module:
            return Import(module)
        name = module.lower().replace("-", "_")
        cls._module = name
        cls.module_path = f"symai.extended.interfaces.{name}"
        expression_cls = Interface.load_module_class(cls.module_path, name) or load_expression(name)
        if expression_cls is None:
            msg = f"No interface or installed symai.expressions plugin named {name!r}."
            raise ValueError(msg)

        return expression_cls(*args, **kwargs)

    def __call__(self, *_args, **_kwargs):
        msg = f"Interface {self._module} is not callable."
        raise NotImplementedError(msg)

    @staticmethod
    def load_module_class(module_path, class_name):
        module_ = locate(module_path)
        if module_ is None:
            return None

        return getattr(module_, class_name, None)
