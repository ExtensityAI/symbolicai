from ... import core
from ...symbol import Expression
from ...utils import UserMessage


class console(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def __call__(self, *args, **kwargs) -> "console":
        def _handler(payload):
            args_ = payload.get('args', ())
            message = ' '.join(str(arg) for arg in args_)
            UserMessage(message)

        kwargs['handler'] = _handler
        @core.output(**kwargs)
        def _func(_, *args):
            pass
        return self.sym_return_type(_func(self, *args))
