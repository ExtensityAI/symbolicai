from ... import core
from ...symbol import Expression, Symbol
from ...backend.engines.drawing.engine_bfl import FluxResult


class flux(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, sym: Symbol, operation: str = 'create', **kwargs) -> FluxResult:
        sym = self._to_symbol(sym)
        @core.draw(operation=operation, **kwargs)
        def _func(_) -> FluxResult:
            pass
        return _func(sym)
