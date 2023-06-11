from ...symbol import Symbol, Expression
from ... import core


class file(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, path: Symbol, **kwargs) -> "file":
        path = self._to_symbol(path)
        @core.opening(path=path.value, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
