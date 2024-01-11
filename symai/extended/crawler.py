from typing import List

from ..components import Clean, Sequence, Stream
from ..symbol import Expression, Symbol
from ..interfaces import Interface


class Crawler(Expression):
    def __init__(self, filters: List[Expression] = [], **kwargs):
        super().__init__(**kwargs)
        filters = filters if isinstance(filters, List) or isinstance(filters, tuple) else [filters]
        self.crawler = Interface('selenium')
        self.data_stream = Stream(Sequence(
            Clean(),
            *filters
        ))

    def forward(self, url: str, pattern='www', **kwargs) -> Symbol:
        res = self.crawler(url=url, pattern=pattern, **kwargs)
        vals = list(self.data_stream(res, **kwargs))
        return Symbol(vals)
