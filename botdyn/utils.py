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
