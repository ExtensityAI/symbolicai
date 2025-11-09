from . import symdev as _symdev
from . import sympkg as _sympkg
from . import symrun as _symrun

__all__ = []
_seen_names = set()


def _export_module(module, seen_names: set[str] = _seen_names) -> None:
    public_names = getattr(module, "__all__", None)
    if public_names is None:
        public_names = [name for name in dir(module) if not name.startswith("_")]
    for name in public_names:
        globals()[name] = getattr(module, name)
        if name not in seen_names:
            __all__.append(name)
            seen_names.add(name)


for _module in (_symdev, _sympkg, _symrun):
    _export_module(_module)


del _export_module
del _module
del _seen_names
del _symdev
del _sympkg
del _symrun
