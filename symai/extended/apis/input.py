import symai as ai


class input(ai.Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, message: str = "Please add more information", **kwargs) -> "input":
        @ai.userinput(**kwargs)
        def _func(_, message) -> str:
            pass
        return self._sym_return_type(_func(self, message))
