import logging

from ... import Expression, Symbol


class longtext(Expression):
    def __init__(self, expr: Expression, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expr = expr
        self.logger = logging.getLogger(__name__)

    def __call__(self, sym: Symbol, *args, **kwargs):
        sym = self._to_symbol(sym)
        res = sym.stream(self.expr, *args, **kwargs)
        combined = list(res)
        return self._to_symbol(combined)
