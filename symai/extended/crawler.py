from typing import List

from ..components import Clean, Sequence, Stream
from ..symbol import Expression, Symbol
from symai.interfaces import Interface


class Crawler(Expression):
    def __init__(self, filters: List[Expression] = []):
        super().__init__()
        filters = filters if isinstance(filters, List) or isinstance(filters, tuple) else [filters]
        self.fetch = Interface('selenium')
        self.data_stream = Stream(Sequence(
            Clean(),
            *filters
        ))

    def forward(self, url: str, pattern='www', **kwargs) -> Symbol:
        res = self.fetch(url=url, pattern=pattern, **kwargs)
        vals = list(self.data_stream(res, **kwargs))
        return Symbol(vals)
