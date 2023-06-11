from ...symbol import Expression
from ... import core


class wolframalpha(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = Expression.command(engines=['symbolic'], expression_engine='wolframalpha')

    def __call__(self, expr: str, **kwargs) -> "wolframalpha":
        @core.expression(expression_engine=self.engine, **kwargs)
        def _func(_, expr: str):
            pass
        return self._sym_return_type(_func(self, expr))
