from . import similarity as _similarity

__all__ = getattr(_similarity, "__all__", None) # noqa
if __all__ is None:
    __all__ = [name for name in dir(_similarity) if not name.startswith("_")]

for _name in __all__:
    globals()[_name] = getattr(_similarity, _name)

del _name
del _similarity
