from symai import core
from symai.backend.engines.formal.engine_axiom import AxiomResult
from symai.symbol import Expression


class axiom(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def __call__(self, expr, tool="check", config=None, **kwargs) -> AxiomResult:
        @core.formal(tool=tool, config=config or {}, **kwargs)
        def _func(_, expr):
            pass

        return _func(self, expr)
