# https://ai.google.dev/gemini-api/docs/models/gemini
SUPPORTED_CHAT_MODELS = [
    "gemini-3.1-flash-lite-preview",
]
SUPPORTED_REASONING_MODELS = [
    # Check the latest snapshots; ie. *-06-05, etc
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]
SUPPORTED_EMBEDDING_MODELS = [
    "gemini-embedding-001",
    "gemini-embedding-2",
]


class GoogleMixin:
    def api_embedding_dims(self):
        if self.model in SUPPORTED_EMBEDDING_MODELS:
            return 3072
        return None

    def api_max_context_tokens(self):
        if self.model == "gemini-embedding-001":
            return 2048
        if self.model == "gemini-embedding-2":
            return 8192
        if self.model.startswith(("gemini-2.5-", "gemini-3")):
            return 1_048_576
        return None

    def api_max_response_tokens(self):
        if self.model.startswith(("gemini-2.5-", "gemini-3")):
            return 65_536
        return None
