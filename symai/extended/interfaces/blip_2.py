from typing import List, Optional

import numpy as np

from ... import core
from ...symbol import Expression


class blip_2(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, image: str = None, text: List[str] = None, **kwargs) -> "blip_2":
        @core.caption(image=image, prompt=text, **kwargs)
        def _func(_) -> str:
            pass
        return self.sym_return_type(_func(self))
