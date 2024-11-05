SUPPORTED_MODELS = [
    # https://docs.anthropic.com/en/docs/about-claude/models
    'claude-3-5-sonnet-latest',
    'claude-3-5-haiku-latest',
    'claude-3-5-sonnet-20241022',
    'claude-3-5-sonnet-20240620',
    'claude-3-opus-latest',
    'claude-3-opus-20240229',
    'claude-3-sonnet-20240229',
    'claude-3-haiku-20240307',
]

class AnthropicMixin:
    def api_max_context_tokens(self):
        if self.model == 'claude-3-5-sonnet-latest' or \
           self.model == 'claude-3-5-sonnet-20241022' or \
           self.model == 'claude-3-5-sonnet-20240620' or \
           self.model == 'claude-3-opus-latest' or \
           self.model == 'claude-3-opus-20240229' or \
           self.model == 'claude-3-sonnet-20240229' or \
           self.model == 'claude-3-haiku-20240307':
            return 200_000

    def api_max_response_tokens(self):
        if self.model == 'claude-3-5-sonnet-latest' or \
           self.model == 'claude-3-5-sonnet-20241022' or \
           self.model == 'claude-3-5-haiku-latest':
            return 8_192
        if self.model == 'claude-3-5-sonnet-20240620' or \
           self.model == 'claude-3-opus-latest' or \
           self.model == 'clade-3-opus-20240229' or \
           self.model == 'claude-3-sonnet-20240229' or \
           self.model == 'claude-3-haiku-20240307':
            return 4_096

