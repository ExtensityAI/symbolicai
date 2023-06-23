from typing import List, Optional

import numpy as np

from ... import core
from ...symbol import Expression


class clip(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, image: Optional[str] = None, text: Optional[List[str]] = None, **kwargs) -> "clip":
        @core.vision(image=image, prompt=text, **kwargs)
        def _func(_) -> np.ndarray:
            pass
        return self._sym_return_type(_func(self))
