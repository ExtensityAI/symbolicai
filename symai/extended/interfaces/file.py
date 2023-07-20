from ... import core
from ...symbol import Expression, Symbol


class file(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, path: Symbol, **kwargs) -> "file":
        path = self._to_symbol(path)
        @core.opening(path=path.value, **kwargs)
        def _func(_) -> str:
            pass
        return self.sym_return_type(_func(self))
