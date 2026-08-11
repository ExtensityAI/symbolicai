from symai.backend.engines.neurosymbolic.anthropic import (
    SUPPORTED_CHAT_MODELS as ANTHROPIC_CHAT_MODELS,
)
from symai.backend.engines.neurosymbolic.anthropic import (
    SUPPORTED_REASONING_MODELS as ANTHROPIC_REASONING_MODELS,
)
from symai.backend.engines.neurosymbolic.anthropic import AnthropicEngine
from symai.backend.engines.neurosymbolic.cerebras import (
    SUPPORTED_CHAT_MODELS as CEREBRAS_CHAT_MODELS,
)
from symai.backend.engines.neurosymbolic.cerebras import (
    SUPPORTED_REASONING_MODELS as CEREBRAS_REASONING_MODELS,
)
from symai.backend.engines.neurosymbolic.cerebras import CerebrasEngine
from symai.backend.engines.neurosymbolic.deepseek import SUPPORTED_MODELS as DEEPSEEK_MODELS
from symai.backend.engines.neurosymbolic.deepseek import DeepseekEngine
from symai.backend.engines.neurosymbolic.google import SUPPORTED_CHAT_MODELS as GOOGLE_CHAT_MODELS
from symai.backend.engines.neurosymbolic.google import (
    SUPPORTED_REASONING_MODELS as GOOGLE_REASONING_MODELS,
)
from symai.backend.engines.neurosymbolic.google import GoogleEngine
from symai.backend.engines.neurosymbolic.groq import (
    SUPPORTED_REASONING_MODELS as GROQ_REASONING_MODELS,
)
from symai.backend.engines.neurosymbolic.groq import GroqEngine
from symai.backend.engines.neurosymbolic.openai import SUPPORTED_OPENAI_MODELS as OPENAI_MODELS
from symai.backend.engines.neurosymbolic.openai import OpenAIEngine
from symai.backend.engines.neurosymbolic.openrouter import (
    SUPPORTED_CHAT_MODELS as OPENROUTER_CHAT_MODELS,
)
from symai.backend.engines.neurosymbolic.openrouter import (
    SUPPORTED_REASONING_MODELS as OPENROUTER_REASONING_MODELS,
)
from symai.backend.engines.neurosymbolic.openrouter import OpenRouterEngine
from symai.backend.engines.neurosymbolic.orcarouter import (
    SUPPORTED_CHAT_MODELS as ORCAROUTER_CHAT_MODELS,
)
from symai.backend.engines.neurosymbolic.orcarouter import (
    SUPPORTED_REASONING_MODELS as ORCAROUTER_REASONING_MODELS,
)
from symai.backend.engines.neurosymbolic.orcarouter import OrcaRouterEngine

# create the mapping
ENGINE_MAPPING = {
    **dict.fromkeys(ANTHROPIC_CHAT_MODELS, AnthropicEngine),
    **dict.fromkeys(ANTHROPIC_REASONING_MODELS, AnthropicEngine),
    **dict.fromkeys(CEREBRAS_CHAT_MODELS, CerebrasEngine),
    **dict.fromkeys(CEREBRAS_REASONING_MODELS, CerebrasEngine),
    **dict.fromkeys(DEEPSEEK_MODELS, DeepseekEngine),
    **dict.fromkeys(GOOGLE_CHAT_MODELS, GoogleEngine),
    **dict.fromkeys(GOOGLE_REASONING_MODELS, GoogleEngine),
    **dict.fromkeys(OPENAI_MODELS, OpenAIEngine),
    **dict.fromkeys(GROQ_REASONING_MODELS, GroqEngine),
    **dict.fromkeys(OPENROUTER_CHAT_MODELS, OpenRouterEngine),
    **dict.fromkeys(OPENROUTER_REASONING_MODELS, OpenRouterEngine),
    **dict.fromkeys(ORCAROUTER_CHAT_MODELS, OrcaRouterEngine),
    **dict.fromkeys(ORCAROUTER_REASONING_MODELS, OrcaRouterEngine),
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
    "ORCAROUTER_CHAT_MODELS",
    "ORCAROUTER_REASONING_MODELS",
    "AnthropicEngine",
    "DeepseekEngine",
    "GoogleEngine",
    "GroqEngine",
    "OpenAIEngine",
    "OpenRouterEngine",
    "OrcaRouterEngine",
]
