from ... import core
from ...symbol import Expression
from ...backend.engines.crawler.engine_selenium import SeleniumResult


class selenium(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, url: str, pattern: str = '', **kwargs) -> SeleniumResult:
        @core.fetch(url=url, pattern=pattern, **kwargs)
        def _func(_) -> SeleniumResult:
            pass
        return _func(self)
