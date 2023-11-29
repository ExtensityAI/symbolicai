from typing import List

import numpy as np

from ... import core
from ...symbol import Expression


class clip(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, image: str = None, text: List[str] = None, **kwargs) -> "clip":
        @core.text_vision(image=image, text=text, **kwargs)
        def _func(_) -> np.ndarray:
            pass
        return self.sym_return_type(_func(self))
