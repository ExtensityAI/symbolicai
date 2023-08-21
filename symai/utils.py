import inspect
import sys
import warnings
import functools
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as PPool


def parallel(worker=mp.cpu_count()//2):
    def dec(function):
        @functools.wraps(function)
        def _dec(*args, **kwargs):
            with PPool(worker) as pool:
                map_obj = pool.map(function, *args, **kwargs)
            return map_obj
        return _dec
    return dec


def ignore_exception(exception=Exception, default=None):
    """ Decorator for ignoring exception from a function
    e.g.   @ignore_exception(DivideByZero)
    e.g.2. ignore_exception(DivideByZero)(Divide)(2/0)
    """
    def dec(function):
        def _dec(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except exception:
                return default
        return _dec
    return dec


def prep_as_str(x):
    return f"'{str(x)}'" if ignore_exception()(int)(str(x)) is None else str(x)


def deprecated(message):
    def deprecated_decorator(func):
        def deprecated_func(*args, **kwargs):
            warnings.warn("{} is a deprecated function. {}".format(func.__name__, message),
                          category=DeprecationWarning,
                          stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)
            return func(*args, **kwargs)
        return deprecated_func
    return deprecated_decorator


class Args:
    def __init__(self, skip_none: bool = False, **kwargs):
        # for each key set an attribute
        for key, value in kwargs.items():
            if value is not None or not skip_none:
                if not key.startswith('_'):
                    setattr(self, key, value)


class CustomUserWarning:
    def __init__(self, message: str, stacklevel: int = 1) -> None:
        caller   = inspect.getframeinfo(inspect.stack()[stacklevel][0])
        lineno   = caller.lineno
        filename = caller.filename
        filename = filename[filename.find('symbolicai'):]
        print(f"{filename}:{lineno}: {UserWarning.__name__}: {message}", file=sys.stderr)

