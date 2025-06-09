from typing import List

from ... import core
from ...symbol import Expression


class blip_2(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def __call__(self, image: str = None, query: str = None, **kwargs) -> "blip_2":
        @core.caption(image=image, prompt=query, **kwargs)
        def _func(_) -> str:
            pass
        return self.sym_return_type(_func(self))
