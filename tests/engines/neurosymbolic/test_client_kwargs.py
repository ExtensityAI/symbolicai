"""
Verify that the six neurosymbolic engines forward ``client_timeout`` and
``client_max_retries`` from their ``__init__`` kwargs into the underlying SDK
client constructor.

These tests are offline: the real SDK client constructors are patched so we
never open a connection, and the only thing under test is the wiring between
engine kwargs and SDK kwargs. They do NOT prove the SDK itself honors the
params — that's a separate guarantee we rely on the SDK vendors for.
"""

from unittest.mock import patch

import pytest

TIMEOUT = 5.0
MAX_RETRIES = 1
DUMMY_KEY = "sk-test-not-a-real-key"


# ---------- Anthropic ----------

def test_claude_chat_forwards_client_kwargs():
    from symai.backend.engines.neurosymbolic import engine_anthropic_claudeX_chat as mod

    with patch.object(mod.anthropic, "Anthropic") as sdk_ctor:
        mod.ClaudeXChatEngine(
            api_key=DUMMY_KEY,
            model="claude-3-5-sonnet-latest",
            client_timeout=TIMEOUT,
            client_max_retries=MAX_RETRIES,
        )

    assert sdk_ctor.call_count == 1
    kwargs = sdk_ctor.call_args.kwargs
    assert kwargs["timeout"] == TIMEOUT
    assert kwargs["max_retries"] == MAX_RETRIES
    assert kwargs["api_key"] == DUMMY_KEY


def test_claude_chat_omits_kwargs_when_not_supplied():
    from symai.backend.engines.neurosymbolic import engine_anthropic_claudeX_chat as mod

    with patch.object(mod.anthropic, "Anthropic") as sdk_ctor:
        mod.ClaudeXChatEngine(api_key=DUMMY_KEY, model="claude-3-5-sonnet-latest")

    kwargs = sdk_ctor.call_args.kwargs
    assert "timeout" not in kwargs
    assert "max_retries" not in kwargs


def test_claude_reasoning_forwards_client_kwargs():
    from symai.backend.engines.neurosymbolic import engine_anthropic_claudeX_reasoning as mod

    with patch.object(mod.anthropic, "Anthropic") as sdk_ctor:
        mod.ClaudeXReasoningEngine(
            api_key=DUMMY_KEY,
            model="claude-sonnet-4-5",
            client_timeout=TIMEOUT,
            client_max_retries=MAX_RETRIES,
        )

    kwargs = sdk_ctor.call_args.kwargs
    assert kwargs["timeout"] == TIMEOUT
    assert kwargs["max_retries"] == MAX_RETRIES


# ---------- OpenAI Chat ----------

def test_openai_chat_forwards_client_kwargs():
    from symai.backend.engines.neurosymbolic import engine_openai_gptX_chat as mod

    with patch.object(mod.openai, "Client") as sdk_ctor:
        mod.GPTXChatEngine(
            api_key=DUMMY_KEY,
            model="gpt-4o-mini",
            client_timeout=TIMEOUT,
            client_max_retries=MAX_RETRIES,
        )

    assert sdk_ctor.call_count == 1
    kwargs = sdk_ctor.call_args.kwargs
    assert kwargs["timeout"] == TIMEOUT
    assert kwargs["max_retries"] == MAX_RETRIES
    assert kwargs["api_key"] == DUMMY_KEY


def test_openai_chat_omits_kwargs_when_not_supplied():
    from symai.backend.engines.neurosymbolic import engine_openai_gptX_chat as mod

    with patch.object(mod.openai, "Client") as sdk_ctor:
        mod.GPTXChatEngine(api_key=DUMMY_KEY, model="gpt-4o-mini")

    kwargs = sdk_ctor.call_args.kwargs
    assert "timeout" not in kwargs
    assert "max_retries" not in kwargs


# ---------- OpenAI Reasoning ----------

def test_openai_reasoning_forwards_client_kwargs():
    from symai.backend.engines.neurosymbolic import engine_openai_gptX_reasoning as mod

    with patch.object(mod.openai, "Client") as sdk_ctor:
        mod.GPTXReasoningEngine(
            api_key=DUMMY_KEY,
            model="o4-mini",
            client_timeout=TIMEOUT,
            client_max_retries=MAX_RETRIES,
        )

    kwargs = sdk_ctor.call_args.kwargs
    assert kwargs["timeout"] == TIMEOUT
    assert kwargs["max_retries"] == MAX_RETRIES


# ---------- OpenAI Responses ----------

def test_openai_responses_forwards_client_kwargs():
    """Responses engine wraps client_timeout in httpx.Timeout(..., connect=10.0).
    Here we only assert the SDK sees max_retries and a Timeout object whose
    read value matches what we passed. A dedicated Timeout-shape assertion
    lives in the integration test for this engine."""
    from symai.backend.engines.neurosymbolic import engine_openai_responses as mod

    with patch.object(mod.openai, "Client") as sdk_ctor:
        mod.OpenAIResponsesEngine(
            api_key=DUMMY_KEY,
            model="responses:gpt-5-mini",
            client_timeout=TIMEOUT,
            client_max_retries=MAX_RETRIES,
        )

    kwargs = sdk_ctor.call_args.kwargs
    assert kwargs["max_retries"] == MAX_RETRIES
    # httpx.Timeout is what's passed; inspect its read value.
    timeout_obj = kwargs["timeout"]
    assert timeout_obj.read == TIMEOUT
    assert timeout_obj.connect == 10.0


def test_openai_responses_defaults_when_kwargs_omitted():
    from symai.backend.engines.neurosymbolic import engine_openai_responses as mod

    with patch.object(mod.openai, "Client") as sdk_ctor:
        mod.OpenAIResponsesEngine(api_key=DUMMY_KEY, model="responses:gpt-5-mini")

    kwargs = sdk_ctor.call_args.kwargs
    assert kwargs["max_retries"] == 3
    timeout_obj = kwargs["timeout"]
    assert timeout_obj.read == 600.0
    assert timeout_obj.connect == 10.0


# ---------- Cerebras ----------

def test_cerebras_forwards_client_kwargs():
    from symai.backend.engines.neurosymbolic import engine_cerebras as mod

    with patch.object(mod, "Cerebras") as sdk_ctor:
        mod.CerebrasEngine(
            api_key=DUMMY_KEY,
            model="cerebras:qwen-3-32b",
            client_timeout=TIMEOUT,
            client_max_retries=MAX_RETRIES,
        )

    assert sdk_ctor.call_count == 1
    kwargs = sdk_ctor.call_args.kwargs
    assert kwargs["timeout"] == TIMEOUT
    assert kwargs["max_retries"] == MAX_RETRIES
    assert kwargs["api_key"] == DUMMY_KEY


def test_cerebras_omits_kwargs_when_not_supplied():
    from symai.backend.engines.neurosymbolic import engine_cerebras as mod

    with patch.object(mod, "Cerebras") as sdk_ctor:
        mod.CerebrasEngine(api_key=DUMMY_KEY, model="cerebras:qwen-3-32b")

    kwargs = sdk_ctor.call_args.kwargs
    assert "timeout" not in kwargs
    assert "max_retries" not in kwargs
