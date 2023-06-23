from ...symbol import Symbol, Expression
from ... import core


class dall_e(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, sym: Symbol, operation: str = 'create', **kwargs) -> "dall_e":
        sym = self._to_symbol(sym)
        @core.draw(operation=operation, **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(sym))
