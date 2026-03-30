# https://ai.google.dev/gemini-api/docs/models/gemini
SUPPORTED_CHAT_MODELS = [
    "gemini-3.1-flash-lite-preview",
]
SUPPORTED_REASONING_MODELS = [
    # Check the latest snapshots; ie. *-06-05, etc
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]


class GoogleMixin:
    def api_max_context_tokens(self):
        if self.model.startswith(("gemini-2.5-", "gemini-3")):
            return 1_048_576
        return None

    def api_max_response_tokens(self):
        if self.model.startswith(("gemini-2.5-", "gemini-3")):
            return 65_536
        return None
