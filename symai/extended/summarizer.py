from typing import List

from ..components import Clean, Outline, Sequence, Stream, Translate
from ..symbol import Expression, Symbol


class Summarizer(Expression):
    def __init__(self, filters: List[Expression] = [], **kwargs):
        super().__init__(**kwargs)
        filters = filters if isinstance(filters, List) or isinstance(filters, tuple) else [filters]
        self.data_stream = Stream(Sequence(
            Clean(),
            Translate(),
            Outline(),
            *filters,
        ))

    def forward(self, sym: Symbol, **kwargs) -> Symbol:
        vals = list(self.data_stream(sym, **kwargs))
        if len(vals) == 1:
            res = {str(vals[0]): Expression(vals[0]).compose(**kwargs)}
            return self.sym_return_type(res)
        res = Expression(vals).cluster()
        sym = res.map()
        summary = {}
        for k, v in sym.value.items():
            summary[k] = Expression(v).compose(**kwargs)
        return self.sym_return_type(summary)
