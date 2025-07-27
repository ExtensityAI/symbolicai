from ... import core
from ...symbol import Expression, Symbol
from ...backend.engines.drawing.engine_gpt_image import GPTImageResult


class dall_e(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def __call__(self, sym: Symbol, operation: str = 'create', **kwargs) -> GPTImageResult:
        sym = self._to_symbol(sym)
        @core.draw(operation=operation, **kwargs)
        def _func(_) -> GPTImageResult:
            pass
        return _func(sym)
