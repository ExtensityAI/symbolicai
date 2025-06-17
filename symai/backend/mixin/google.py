# https://ai.google.dev/gemini-api/docs/models/gemini
SUPPORTED_CHAT_MODELS = []
SUPPORTED_REASONING_MODELS = [
    # Check the latest snapshots; ie. *-06-05, etc
    'gemini-2.5-pro',
    'gemini-2.5-flash',
]

class GoogleMixin:
    def api_max_context_tokens(self):
        if self.model.startswith('gemini-2.5-'):
            return 1_048_576

    def api_max_response_tokens(self):
        if self.model == 'gemini-2.5-':
            return 65_536
