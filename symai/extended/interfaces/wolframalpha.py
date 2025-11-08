from ... import core
from ...backend.engines.symbolic.engine_wolframalpha import WolframResult
from ...symbol import Expression


class wolframalpha(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def __call__(self, expr: str, **kwargs) -> WolframResult:
        @core.expression(**kwargs)
        def _func(_, expr: str) -> WolframResult:
            pass
        return _func(self, expr)
