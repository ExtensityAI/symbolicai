# https://docs.anthropic.com/en/docs/about-claude/models
SUPPORTED_CHAT_MODELS = [
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-latest",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]
SUPPORTED_REASONING_MODELS = [
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-opus-4-1",
    "claude-opus-4-0",
    "claude-sonnet-4-0",
    "claude-3-7-sonnet-latest",
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
]

LONG_CONTEXT_1M_TOKENS = 1_000_000
LONG_CONTEXT_1M_BETA_HEADER = "context-1m-2025-08-07"
LONG_CONTEXT_1M_MODELS = {
    "claude-opus-4-6",
    "claude-sonnet-4-5",
}


class AnthropicMixin:
    def supports_long_context_1m(self, model: str) -> bool:
        return model in LONG_CONTEXT_1M_MODELS

    def long_context_beta_header(self) -> str:
        return LONG_CONTEXT_1M_BETA_HEADER

    def api_max_context_tokens(self, long_context_1m: bool = False, model: str | None = None):
        selected_model = self.model if model is None else model
        if long_context_1m and self.supports_long_context_1m(selected_model):
            return LONG_CONTEXT_1M_TOKENS
        if (
            selected_model == "claude-opus-4-6"
            or selected_model == "claude-opus-4-5"
            or selected_model == "claude-opus-4-1"
            or selected_model == "claude-opus-4-0"
            or selected_model == "claude-sonnet-4-0"
            or selected_model == "claude-3-7-sonnet-latest"
            or selected_model == "claude-haiku-4-5"
            or selected_model == "claude-sonnet-4-5"
            or selected_model == "claude-3-5-sonnet-latest"
            or selected_model == "claude-3-5-sonnet-20241022"
            or selected_model == "claude-3-5-sonnet-20240620"
            or selected_model == "claude-3-opus-latest"
            or selected_model == "claude-3-opus-20240229"
            or selected_model == "claude-3-sonnet-20240229"
            or selected_model == "claude-3-haiku-20240307"
        ):
            return 200_000
        return None

    def api_max_response_tokens(self):
        if self.model == "claude-opus-4-6":
            return 128_000

        if (
            self.model == "claude-opus-4-5"
            or self.model == "claude-sonnet-4-0"
            or self.model == "claude-3-7-sonnet-latest"
            or self.model == "claude-haiku-4-5"
            or self.model == "claude-sonnet-4-5"
        ):
            return 64_000
        if self.model == "claude-opus-4-1" or self.model == "claude-opus-4-0":
            return 32_000
        if (
            self.model == "claude-3-5-sonnet-latest"
            or self.model == "claude-3-5-sonnet-20241022"
            or self.model == "claude-3-5-haiku-latest"
        ):
            return 8_192
        if (
            self.model == "claude-3-5-sonnet-20240620"
            or self.model == "claude-3-opus-latest"
            or self.model == "claude-3-opus-20240229"
            or self.model == "claude-3-sonnet-20240229"
            or self.model == "claude-3-haiku-20240307"
        ):
            return 4_096
        return None
