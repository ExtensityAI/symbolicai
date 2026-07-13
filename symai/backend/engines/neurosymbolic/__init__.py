from symai.backend.engines.language_model.cerebras import (
    REGISTERED_MODELS as CEREBRAS_MODELS,
)
from symai.backend.engines.language_model.cerebras import (
    LanguageModelEngine as CerebrasLanguageModelEngine,
)
from symai.backend.engines.language_model.deepseek import REGISTERED_MODELS as DEEPSEEK_MODELS
from symai.backend.engines.language_model.deepseek import (
    LanguageModelEngine as DeepSeekLanguageModelEngine,
)
from symai.backend.engines.language_model.openai import REGISTERED_MODELS as OPENAI_MODELS
from symai.backend.engines.language_model.openai import (
    LanguageModelEngine as OpenAILanguageModelEngine,
)
from symai.backend.engines.neurosymbolic.engine_anthropic_claudeX_chat import ClaudeXChatEngine
from symai.backend.engines.neurosymbolic.engine_anthropic_claudeX_reasoning import (
    ClaudeXReasoningEngine,
)
from symai.backend.engines.neurosymbolic.engine_google_geminiX_reasoning import (
    GeminiXReasoningEngine,
)
from symai.backend.engines.neurosymbolic.engine_groq import GroqEngine
from symai.backend.engines.neurosymbolic.engine_openrouter import OpenRouterEngine
from symai.backend.mixin import (
    ANTHROPIC_CHAT_MODELS,
    ANTHROPIC_REASONING_MODELS,
    GOOGLE_CHAT_MODELS,
    GOOGLE_REASONING_MODELS,
    GROQ_REASONING_MODELS,
    OPENROUTER_CHAT_MODELS,
    OPENROUTER_REASONING_MODELS,
)

# create the mapping
ENGINE_MAPPING = {
    **dict.fromkeys(ANTHROPIC_CHAT_MODELS, ClaudeXChatEngine),
    **dict.fromkeys(ANTHROPIC_REASONING_MODELS, ClaudeXReasoningEngine),
    **dict.fromkeys(CEREBRAS_MODELS, CerebrasLanguageModelEngine),
    **dict.fromkeys(DEEPSEEK_MODELS, DeepSeekLanguageModelEngine),
    **dict.fromkeys(GOOGLE_CHAT_MODELS, GeminiXReasoningEngine),
    **dict.fromkeys(GOOGLE_REASONING_MODELS, GeminiXReasoningEngine),
    **dict.fromkeys(OPENAI_MODELS, OpenAILanguageModelEngine),
    **dict.fromkeys(GROQ_REASONING_MODELS, GroqEngine),
    **dict.fromkeys(OPENROUTER_CHAT_MODELS, OpenRouterEngine),
    **dict.fromkeys(OPENROUTER_REASONING_MODELS, OpenRouterEngine),
}
