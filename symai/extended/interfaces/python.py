from ... import core
from ...symbol import Expression, Result


class python(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def __call__(self, expr: str, **kwargs) -> Result:
        sym = self._to_symbol(expr)
        @core.execute(**kwargs)
        def _func(_) -> Result:
            pass
        return _func(sym)
