import logging

from ... import core_ext
from ...symbol import Expression, Symbol


class MaxTokensExceptRemedy(Expression):
    def __init__(self, expr: Expression, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expr = expr
        self.logger = logging.getLogger(__name__)

    @core_ext.bind(engine='neurosymbolic', property='max_tokens')
    def _max_tokens(self): pass

    def forward(self, ex: Exception, context: Symbol, *args, **kwargs):
        self.logger.warn(f"Error: {ex}")
        sym = self._to_symbol(ex)
        if sym.contains('maximum context length'):
            token_size = sym.extract("tokens in your prompt")
            token_size = token_size.cast(int)
            max_tokens = self.max_tokens() - token_size
            self.logger.warn(f"Try to remedy the exceeding of the maximum token limitation! Set max_tokens to {max_tokens}, tokens in prompt {token_size}")
        context = self._to_symbol(context)
        res = context.stream(self.expr, *args, **kwargs)
        combined = list(res)
        return self._to_symbol(combined)
