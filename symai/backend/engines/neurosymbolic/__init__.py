from ...mixin import (
                      ANTHROPIC_CHAT_MODELS,
                      ANTHROPIC_REASONING_MODELS,
                      DEEPSEEK_CHAT_MODELS,
                      DEEPSEEK_REASONING_MODELS,
                      GOOGLE_CHAT_MODELS,
                      GOOGLE_REASONING_MODELS,
                      GROQ_CHAT_MODELS,
                      GROQ_REASONING_MODELS,
                      OPENAI_CHAT_MODELS,
                      OPENAI_REASONING_MODELS,
)
from .engine_anthropic_claudeX_chat import ClaudeXChatEngine
from .engine_anthropic_claudeX_reasoning import ClaudeXReasoningEngine
from .engine_deepseekX_reasoning import DeepSeekXReasoningEngine
from .engine_google_geminiX_reasoning import GeminiXReasoningEngine
from .engine_groq import GroqEngine
from .engine_openai_gptX_chat import GPTXChatEngine
from .engine_openai_gptX_reasoning import GPTXReasoningEngine

# create the mapping
ENGINE_MAPPING = {
    **dict.fromkeys(ANTHROPIC_CHAT_MODELS, ClaudeXChatEngine),
    **dict.fromkeys(ANTHROPIC_REASONING_MODELS, ClaudeXReasoningEngine),
    **dict.fromkeys(DEEPSEEK_REASONING_MODELS, DeepSeekXReasoningEngine),
    **dict.fromkeys(GOOGLE_REASONING_MODELS, GeminiXReasoningEngine),
    **dict.fromkeys(OPENAI_CHAT_MODELS, GPTXChatEngine),
    **dict.fromkeys(OPENAI_REASONING_MODELS, GPTXReasoningEngine),
    **dict.fromkeys(GROQ_CHAT_MODELS, GroqEngine),
    **dict.fromkeys(GROQ_REASONING_MODELS, GroqEngine),
}

__all__ = [
                      "ANTHROPIC_CHAT_MODELS",
                      "ANTHROPIC_REASONING_MODELS",
                      "DEEPSEEK_CHAT_MODELS",
                      "DEEPSEEK_REASONING_MODELS",
                      "ENGINE_MAPPING",
                      "GOOGLE_CHAT_MODELS",
                      "GOOGLE_REASONING_MODELS",
                      "GROQ_CHAT_MODELS",
                      "GROQ_REASONING_MODELS",
                      "OPENAI_CHAT_MODELS",
                      "OPENAI_REASONING_MODELS",
                      "ClaudeXChatEngine",
                      "ClaudeXReasoningEngine",
                      "DeepSeekXReasoningEngine",
                      "GPTXChatEngine",
                      "GPTXReasoningEngine",
                      "GeminiXReasoningEngine",
                      "GroqEngine",
]
