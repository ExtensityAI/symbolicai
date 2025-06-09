from typing import List, Union

import numpy as np

from ... import core
from ...symbol import Expression


class clip(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def __call__(self, image: str | bytes = None, text: List[str] = None, **kwargs) -> "clip":
        @core.text_vision(image=image, text=text, **kwargs)
        def _func(_) -> np.ndarray:
            pass
        return self.sym_return_type(_func(self))
