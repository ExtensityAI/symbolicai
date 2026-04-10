SUPPORTED_CHAT_MODELS: list[str] = [
    "cerebras:qwen-3-235b-a22b-instruct-2507",
]

SUPPORTED_REASONING_MODELS: list[str] = [
    "cerebras:zai-glm-4.6",
    "cerebras:gpt-oss-120b",
    "cerebras:qwen-3-32b",
]


class CerebrasMixin:
    def api_max_context_tokens(self):
        model = getattr(self, "model", "")
        if "qwen-3-235b" in model:
            return 65_536
        if "qwen-3-32b" in model:
            return 32_768
        # gpt-oss-120b, zai-glm-4.6, and fallback
        return 131_000

    def api_max_response_tokens(self):
        model = getattr(self, "model", "")
        if "qwen-3-32b" in model:
            return 32_768
        if "qwen-3-235b" in model:
            return 65_536
        # gpt-oss-120b, zai-glm-4.6, and fallback
        return 131_000
