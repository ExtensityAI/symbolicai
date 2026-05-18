from types import SimpleNamespace
from unittest.mock import patch

from anthropic._types import NOT_GIVEN

import symai.backend.mixin.anthropic as anthropic_mixin
import symai.backend.mixin.openai as openai_mixin

DUMMY_KEY = "sk-test-not-a-real-key"


class AnthropicProbe(anthropic_mixin.AnthropicMixin):
    def __init__(self, model):
        self.model = model


class OpenAIProbe(openai_mixin.OpenAIMixin):
    def __init__(self, model):
        self.model = model


def test_new_models_are_registered():
    assert "claude-opus-4-7" in anthropic_mixin.SUPPORTED_REASONING_MODELS
    assert "gpt-5.5" in openai_mixin.SUPPORTED_REASONING_MODELS
    assert "gpt-5.5-2026-04-23" in openai_mixin.SUPPORTED_REASONING_MODELS


def test_new_model_token_limits():
    claude = AnthropicProbe("claude-opus-4-7")
    assert claude.api_max_context_tokens() == 1_000_000
    assert claude.api_max_response_tokens() == 128_000

    for model in ("gpt-5.5", "gpt-5.5-2026-04-23"):
        openai = OpenAIProbe(model)
        assert openai.api_max_context_tokens() == 1_050_000
        assert openai.api_max_response_tokens() == 128_000


def test_claude_opus_4_7_uses_reasoning_engine():
    from symai.backend.engines.neurosymbolic import engine_anthropic_claudeX_chat as chat_mod
    from symai.backend.engines.neurosymbolic import (
        engine_anthropic_claudeX_reasoning as reasoning_mod,
    )

    with patch.object(reasoning_mod.anthropic, "Anthropic"):
        reasoning_engine = reasoning_mod.ClaudeXReasoningEngine(
            api_key=DUMMY_KEY,
            model="claude-opus-4-7",
        )
    with patch.object(chat_mod.anthropic, "Anthropic"):
        chat_engine = chat_mod.ClaudeXChatEngine(
            api_key=DUMMY_KEY,
            model="claude-opus-4-7",
        )

    assert reasoning_engine.id() == "neurosymbolic"
    assert chat_engine.id() != "neurosymbolic"


def test_gpt_5_5_uses_reasoning_engine():
    from symai.backend.engines.neurosymbolic import engine_openai_gptX_reasoning as mod

    for model in ("gpt-5.5", "gpt-5.5-2026-04-23"):
        with patch.object(mod.openai, "Client"):
            engine = mod.GPTXReasoningEngine(api_key=DUMMY_KEY, model=model)

        assert engine.id() == "neurosymbolic"
        assert engine.max_context_tokens == 1_050_000
        assert engine.max_response_tokens == 128_000


def test_claude_opus_4_7_without_thinking_omits_sampling_params():
    from symai.backend.engines.neurosymbolic import engine_anthropic_claudeX_reasoning as mod

    with patch.object(mod.anthropic, "Anthropic"):
        engine = mod.ClaudeXReasoningEngine(api_key=DUMMY_KEY, model="claude-opus-4-7")

    argument = SimpleNamespace(kwargs={}, prop=SimpleNamespace(response_format=None))
    payload = engine._prepare_request_payload(argument)

    assert payload["thinking"] is NOT_GIVEN
    assert payload["temperature"] is NOT_GIVEN
    assert payload["top_p"] is NOT_GIVEN
    assert payload["top_k"] is NOT_GIVEN


def test_claude_opus_4_7_thinking_is_adaptive():
    from symai.backend.engines.neurosymbolic import engine_anthropic_claudeX_reasoning as mod

    with patch.object(mod.anthropic, "Anthropic"):
        engine = mod.ClaudeXReasoningEngine(api_key=DUMMY_KEY, model="claude-opus-4-7")

    argument = SimpleNamespace(
        kwargs={"thinking": {"type": "enabled", "budget_tokens": 4096, "effort": "high"}},
        prop=SimpleNamespace(response_format=None),
    )
    payload = engine._prepare_request_payload(argument)

    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"] == {"effort": "high"}
