import symai as ai


class wolframalpha(ai.Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = ai.Expression.command(engines=['symbolic'], expression_engine='wolframalpha')

    def __call__(self, expr: str, **kwargs) -> "wolframalpha":
        @ai.expression(expression_engine=self.engine, **kwargs)
        def _func(_, expr: str):
            pass
        return self._sym_return_type(_func(self, expr))
