from ... import core
from ...symbol import Expression


class input(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, message: str = "Please add more information", **kwargs) -> "input":
        @core.userinput(**kwargs)
        def _func(_, message) -> str:
            pass
        return self.sym_return_type(_func(self, message))
