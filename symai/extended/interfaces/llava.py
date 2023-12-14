from typing import List

from ... import core
from ...symbol import Expression


class llava(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, image: str, query: str, **kwargs) -> "llava":
        @core.caption(image=image, prompt=query, **kwargs)
        def _func(_) -> str:
            pass
        return self.sym_return_type(_func(self))
