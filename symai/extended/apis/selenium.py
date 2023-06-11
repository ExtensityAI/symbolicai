import symai as ai


class selenium(ai.Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, url: str, pattern: str = '', **kwargs) -> "selenium":
        @ai.fetch(url=url, pattern=pattern, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))
