from ...mixin import (ANTHROPIC_CHAT_MODELS, ANTHROPIC_REASONING_MODELS,
                      DEEPSEEK_CHAT_MODELS, DEEPSEEK_REASONING_MODELS,
                      GOOGLE_CHAT_MODELS, GOOGLE_REASONING_MODELS,
                      OPENAI_CHAT_MODELS, OPENAI_REASONING_MODELS)
from .engine_anthropic_claudeX_chat import ClaudeXChatEngine
from .engine_anthropic_claudeX_reasoning import ClaudeXReasoningEngine
from .engine_deepseekX_reasoning import DeepSeekXReasoningEngine
from .engine_google_geminiX_reasoning import GeminiXReasoningEngine
from .engine_openai_gptX_chat import GPTXChatEngine
from .engine_openai_gptX_reasoning import GPTXReasoningEngine

# create the mapping
ENGINE_MAPPING = {
    **{model_name: ClaudeXChatEngine for model_name in ANTHROPIC_CHAT_MODELS},
    **{model_name: ClaudeXReasoningEngine for model_name in ANTHROPIC_REASONING_MODELS},
    **{model_name: DeepSeekXReasoningEngine for model_name in DEEPSEEK_REASONING_MODELS},
    **{model_name: GeminiXReasoningEngine for model_name in GOOGLE_REASONING_MODELS},
    **{model_name: GPTXChatEngine for model_name in OPENAI_CHAT_MODELS},
    **{model_name: GPTXReasoningEngine for model_name in OPENAI_REASONING_MODELS},
}
