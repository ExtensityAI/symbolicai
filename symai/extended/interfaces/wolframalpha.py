from ... import core
from ...symbol import Expression
from ...backend.engines.symbolic.engine_wolframalpha import WolframResult


class wolframalpha(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = Expression.command(engines=['symbolic'], expression_engine='wolframalpha')

    def __call__(self, expr: str, **kwargs) -> WolframResult:
        @core.expression(expression_engine=self.engine, **kwargs)
        def _func(_, expr: str) -> WolframResult:
            pass
        return _func(self, expr)
