# https://ai.google.dev/gemini-api/docs/models/gemini
SUPPORTED_CHAT_MODELS = []
SUPPORTED_REASONING_MODELS = [
    'gemini-2.5-pro-preview-05-06',
    'gemini-2.5-flash-preview-05-20',
]

class GoogleMixin:
    def api_max_context_tokens(self):
        if self.model.startswith('gemini-2.5-'):
            return 1_048_576

    def api_max_response_tokens(self):
        if self.model == 'gemini-2.5-':
            return 65_536
