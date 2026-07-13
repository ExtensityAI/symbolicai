from symai.backend.engines.neurosymbolic.engine_anthropic_claudeX_chat import ClaudeXChatEngine
from symai.backend.engines.neurosymbolic.engine_anthropic_claudeX_reasoning import (
    ClaudeXReasoningEngine,
)
from symai.backend.engines.neurosymbolic.engine_google_geminiX_reasoning import (
    GeminiXReasoningEngine,
)
from symai.backend.engines.neurosymbolic.engine_groq import GroqEngine
from symai.backend.engines.neurosymbolic.engine_openrouter import OpenRouterEngine
from symai.backend.integrations.cerebras.engines.neurosymbolic import CerebrasEngine
from symai.backend.integrations.deepseek.engines.neurosymbolic import DeepSeekEngine
from symai.backend.integrations.openai.engines.neurosymbolic import OpenAIResponsesEngine
from symai.backend.mixin import (
    ANTHROPIC_CHAT_MODELS,
    ANTHROPIC_REASONING_MODELS,
    CEREBRAS_CHAT_MODELS,
    CEREBRAS_REASONING_MODELS,
    DEEPSEEK_MODELS,
    GOOGLE_CHAT_MODELS,
    GOOGLE_REASONING_MODELS,
    GROQ_REASONING_MODELS,
    OPENAI_MODELS,
    OPENROUTER_CHAT_MODELS,
    OPENROUTER_REASONING_MODELS,
)

# create the mapping
ENGINE_MAPPING = {
    **dict.fromkeys(ANTHROPIC_CHAT_MODELS, ClaudeXChatEngine),
    **dict.fromkeys(ANTHROPIC_REASONING_MODELS, ClaudeXReasoningEngine),
    **dict.fromkeys(CEREBRAS_CHAT_MODELS, CerebrasEngine),
    **dict.fromkeys(CEREBRAS_REASONING_MODELS, CerebrasEngine),
    **dict.fromkeys(DEEPSEEK_MODELS, DeepSeekEngine),
    **dict.fromkeys(GOOGLE_CHAT_MODELS, GeminiXReasoningEngine),
    **dict.fromkeys(GOOGLE_REASONING_MODELS, GeminiXReasoningEngine),
    **dict.fromkeys(OPENAI_MODELS, OpenAIResponsesEngine),
    **dict.fromkeys(GROQ_REASONING_MODELS, GroqEngine),
    **dict.fromkeys(OPENROUTER_CHAT_MODELS, OpenRouterEngine),
    **dict.fromkeys(OPENROUTER_REASONING_MODELS, OpenRouterEngine),
}

__all__ = [
    "ANTHROPIC_CHAT_MODELS",
    "ANTHROPIC_REASONING_MODELS",
    "CEREBRAS_CHAT_MODELS",
    "CEREBRAS_REASONING_MODELS",
    "DEEPSEEK_MODELS",
    "ENGINE_MAPPING",
    "GOOGLE_CHAT_MODELS",
    "GOOGLE_REASONING_MODELS",
    "GROQ_REASONING_MODELS",
    "OPENAI_MODELS",
    "OPENROUTER_CHAT_MODELS",
    "OPENROUTER_REASONING_MODELS",
    "ClaudeXChatEngine",
    "ClaudeXReasoningEngine",
    "DeepSeekEngine",
    "GeminiXReasoningEngine",
    "GroqEngine",
    "OpenAIResponsesEngine",
    "OpenRouterEngine",
]
