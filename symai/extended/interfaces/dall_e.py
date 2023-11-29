from ... import core
from ...symbol import Expression, Symbol
from ...backend.engines.imagerendering.engine_dall_e import DalleResult


class dall_e(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, sym: Symbol, operation: str = 'create', **kwargs) -> DalleResult:
        sym = self._to_symbol(sym)
        @core.draw(operation=operation, **kwargs)
        def _func(_) -> DalleResult:
            pass
        return _func(sym)
