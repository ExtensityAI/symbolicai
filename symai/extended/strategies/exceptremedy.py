import logging

from ... import Expression, Symbol


class ExceptRemedy(Expression):
    def __init__(self, expr: Expression, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expr = expr
        self.logger = logging.getLogger(__name__)

    def forward(self, ex: Exception, context: Symbol, max_tokens = 4000, *args, **kwargs):
        self.logger.warn(f"Error: {ex}")
        sym = self._to_symbol(ex)
        if sym.contains('maximum context length'):
            token_size = sym.extract("tokens in your prompt")
            token_size = token_size.cast(int)
            max_tokens = max_tokens - token_size
            self.logger.warn(f"Try to remedy the exceeding of the maximum token limitation! Set max_tokens to {max_tokens}, tokens in prompt {token_size}")
        context = self._to_symbol(context)
        res = context.stream(self.expr, max_tokens=max_tokens, *args, **kwargs)
        combined = list(res)
        return self._to_symbol(combined)
