# https://api-docs.deepseek.com/quick_start/pricing
SUPPORTED_CHAT_MODELS = []
SUPPORTED_REASONING_MODELS = [
    'deepseek-reasoner'
]

class DeepSeekMixin:
    def api_max_context_tokens(self):
        if self.model == 'deepseek-reasoner':
            return 64_000

    def api_max_response_tokens(self):
        if self.model == 'deepseek-reasoner':
            return 8_000
