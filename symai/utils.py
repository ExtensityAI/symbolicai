import inspect
import sys


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


class CustomUserWarning:
    def __init__(self, message: str, stacklevel: int = 1) -> None:
        caller   = inspect.getframeinfo(inspect.stack()[stacklevel][0])
        lineno   = caller.lineno
        filename = caller.filename
        filename = filename[filename.find('symbolicai'):]
        print(f"{filename}:{lineno}: {UserWarning.__name__}: {message}", file=sys.stderr)

