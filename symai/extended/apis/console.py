import symai as ai


class console(ai.Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> "console":
        kwargs['handler'] = lambda x: print(*x['args'])
        @ai.output(**kwargs)
        def _func(_, *args):
            pass
        return self._sym_return_type(_func(self, *args))
