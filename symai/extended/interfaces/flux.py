from ... import core
from ...symbol import Expression, Symbol
from ...backend.engines.drawing.engine_bfl import FluxResult


class flux(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def __call__(self, prompt: str, operation: str = 'create', **kwargs) -> FluxResult:
        prompt = self._to_symbol(prompt)
        @core.draw(operation=operation, **kwargs)
        def _func(_) -> FluxResult:
            pass
        return _func(prompt)
