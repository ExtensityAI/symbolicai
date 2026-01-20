import os
import time

import pytest

from symai import Expression, PrimitiveDisabler, Symbol
from symai.backend.mixin import (
    ANTHROPIC_CHAT_MODELS,
    ANTHROPIC_REASONING_MODELS,
    CEREBRAS_CHAT_MODELS,
    CEREBRAS_REASONING_MODELS,
    DEEPSEEK_CHAT_MODELS,
    DEEPSEEK_REASONING_MODELS,
    GOOGLE_CHAT_MODELS,
    GOOGLE_REASONING_MODELS,
    GROQ_CHAT_MODELS,
    GROQ_REASONING_MODELS,
    OPENAI_CHAT_MODELS,
    OPENAI_REASONING_MODELS,
    OPENAI_RESPONSES_MODELS,
)
from symai.backend.settings import SYMAI_CONFIG
from symai.components import ChonkieChunker, DynamicEngine, FileReader, MetadataTracker
from symai.extended import Interface
from symai.functional import EngineRepository
from symai.utils import RuntimeInfo

NEUROSYMBOLIC_MODEL = SYMAI_CONFIG.get("NEUROSYMBOLIC_ENGINE_MODEL", "")
SEARCH_MODEL = SYMAI_CONFIG.get("SEARCH_ENGINE_MODEL", "")

SUPPORTED_NEUROSYMBOLIC_MODELS = (
    OPENAI_CHAT_MODELS
    + OPENAI_REASONING_MODELS
    + OPENAI_RESPONSES_MODELS
    + ANTHROPIC_CHAT_MODELS
    + ANTHROPIC_REASONING_MODELS
    + GOOGLE_CHAT_MODELS
    + GOOGLE_REASONING_MODELS
    + DEEPSEEK_CHAT_MODELS
    + DEEPSEEK_REASONING_MODELS
    + GROQ_CHAT_MODELS
    + GROQ_REASONING_MODELS
    + CEREBRAS_CHAT_MODELS
    + CEREBRAS_REASONING_MODELS
)

NEUROSYMBOLIC_ENGINE = NEUROSYMBOLIC_MODEL in SUPPORTED_NEUROSYMBOLIC_MODELS
# Search interface currently targets OpenAI responses/web_search tooling.
SEARCH_ENGINE = SEARCH_MODEL in OPENAI_CHAT_MODELS + OPENAI_REASONING_MODELS


def estimate_cost(info: RuntimeInfo, pricing: dict) -> float:
    input_cost = (info.prompt_tokens - info.cached_tokens) * pricing.get("input", 0)
    cached_input_cost = info.cached_tokens * pricing.get("cached_input", 0)
    output_cost = info.completion_tokens * pricing.get("output", 0)
    call_cost = info.total_calls * pricing.get("calls", 0)
    return input_cost + cached_input_cost + output_cost + call_cost


@pytest.mark.skipif(
    not NEUROSYMBOLIC_ENGINE and not SEARCH_ENGINE,
    reason="Requires a configured model supported by MetadataTracker (neurosymbolic and/or OpenAI search).",
)
def test_metadata_tracker_with_runtimeinfo():
    sym = Symbol("You're supposed to go off the rails and say something pretty crazy.")
    start_time = time.perf_counter()
    with MetadataTracker() as tracker:
        did_neurosymbolic = False
        did_search = False

        res = None
        search_res = None

        if NEUROSYMBOLIC_ENGINE:
            res = sym.query("What is the meaning of life?")
            did_neurosymbolic = True

        # Only attempt OpenAI web-search if configured; otherwise keep the test provider-agnostic.
        if SEARCH_ENGINE:
            try:
                search = Interface("openai_search")
                search_res = search("What is the meaning of life?")
                did_search = True
            except Exception:
                # If search isn't configured (e.g., missing key) we still want to validate MetadataTracker.
                did_search = False
    end_time = time.perf_counter()
    assert did_neurosymbolic or did_search, "Expected at least one engine call to be executed."
    if did_neurosymbolic:
        assert res is not None and res.value is not None, "Expected non-None output from neurosymbolic query."
    if did_search:
        assert search_res is not None and search_res.value is not None, "Expected non-None output from search."

    # Approximate $/token pricing used only for unit-test cost estimation.
    # Keep this table in sync with the "standard" OpenAI models supported by SymbolicAI.
    openai_pricing_per_model = {
        # GPT-3.5
        "gpt-3.5-turbo": {"input": 0.50 / 1e6, "cached_input": 0.13 / 1e6, "output": 1.50 / 1e6},
        "gpt-3.5-turbo-16k": {"input": 0.50 / 1e6, "cached_input": 0.13 / 1e6, "output": 1.50 / 1e6},
        "gpt-3.5-turbo-1106": {"input": 0.50 / 1e6, "cached_input": 0.13 / 1e6, "output": 1.50 / 1e6},
        "gpt-3.5-turbo-0613": {"input": 0.50 / 1e6, "cached_input": 0.13 / 1e6, "output": 1.50 / 1e6},
        # GPT-4 / Turbo
        "gpt-4": {"input": 30.0 / 1e6, "cached_input": 7.5 / 1e6, "output": 60.0 / 1e6},
        "gpt-4-0613": {"input": 30.0 / 1e6, "cached_input": 7.5 / 1e6, "output": 60.0 / 1e6},
        "gpt-4-1106-preview": {"input": 10.0 / 1e6, "cached_input": 2.5 / 1e6, "output": 30.0 / 1e6},
        "gpt-4-turbo": {"input": 10.0 / 1e6, "cached_input": 2.5 / 1e6, "output": 30.0 / 1e6},
        "gpt-4-turbo-2024-04-09": {"input": 10.0 / 1e6, "cached_input": 2.5 / 1e6, "output": 30.0 / 1e6},
        # GPT-4o
        "gpt-4o": {"input": 5.0 / 1e6, "cached_input": 1.25 / 1e6, "output": 15.0 / 1e6},
        "gpt-4o-2024-11-20": {"input": 5.0 / 1e6, "cached_input": 1.25 / 1e6, "output": 15.0 / 1e6},
        "gpt-4o-mini": {"input": 0.15 / 1e6, "cached_input": 0.04 / 1e6, "output": 0.60 / 1e6},
        "chatgpt-4o-latest": {"input": 5.0 / 1e6, "cached_input": 1.25 / 1e6, "output": 15.0 / 1e6},
        # GPT-4.1 family
        "gpt-4.1": {"input": 2.0 / 1e6, "cached_input": 0.5 / 1e6, "output": 8.0 / 1e6},
        "gpt-4.1-mini": {"input": 0.40 / 1e6, "cached_input": 0.10 / 1e6, "output": 1.60 / 1e6},
        "gpt-4.1-nano": {"input": 0.10 / 1e6, "cached_input": 0.03 / 1e6, "output": 0.40 / 1e6},
        # GPT-5 chat aliases
        "gpt-5-chat-latest": {"input": 5.0 / 1e6, "cached_input": 1.25 / 1e6, "output": 20.0 / 1e6},
        "gpt-5.1-chat-latest": {"input": 5.0 / 1e6, "cached_input": 1.25 / 1e6, "output": 20.0 / 1e6},
        "gpt-5.2-chat-latest": {"input": 5.0 / 1e6, "cached_input": 1.25 / 1e6, "output": 20.0 / 1e6},
        # Reasoning models (approximate)
        "o3-mini": {"input": 1.0 / 1e6, "cached_input": 0.25 / 1e6, "output": 4.0 / 1e6},
        "o4-mini": {"input": 1.0 / 1e6, "cached_input": 0.25 / 1e6, "output": 4.0 / 1e6},
        "o1": {"input": 10.0 / 1e6, "cached_input": 2.5 / 1e6, "output": 40.0 / 1e6},
        "o3": {"input": 10.0 / 1e6, "cached_input": 2.5 / 1e6, "output": 40.0 / 1e6},
        # GPT-5 reasoning family (approximate)
        "gpt-5": {"input": 5.0 / 1e6, "cached_input": 1.25 / 1e6, "output": 20.0 / 1e6},
        "gpt-5.1": {"input": 5.0 / 1e6, "cached_input": 1.25 / 1e6, "output": 20.0 / 1e6},
        "gpt-5.2": {"input": 5.0 / 1e6, "cached_input": 1.25 / 1e6, "output": 20.0 / 1e6},
        "gpt-5-mini": {"input": 0.40 / 1e6, "cached_input": 0.10 / 1e6, "output": 1.60 / 1e6},
        "gpt-5-nano": {"input": 0.10 / 1e6, "cached_input": 0.03 / 1e6, "output": 0.40 / 1e6},
        # Responses-only model aliases (approximate)
        "gpt-5-pro": {"input": 10.0 / 1e6, "cached_input": 2.5 / 1e6, "output": 40.0 / 1e6},
        "o3-pro": {"input": 15.0 / 1e6, "cached_input": 3.75 / 1e6, "output": 60.0 / 1e6},
        "gpt-5.2-pro": {"input": 10.0 / 1e6, "cached_input": 2.5 / 1e6, "output": 40.0 / 1e6},
    }

    # Approximate pricing for other providers (used only to estimate cost; not meant to be exact).
    anthropic_pricing_per_model = {
        # Claude 3.5 family (approximate)
        "claude-3-5-sonnet-latest": {"input": 3.0 / 1e6, "cached_input": 0.75 / 1e6, "output": 15.0 / 1e6},
        "claude-3-5-haiku-latest": {"input": 1.0 / 1e6, "cached_input": 0.25 / 1e6, "output": 5.0 / 1e6},
        # Claude 4.x / 3.7 reasoning-ish (approximate)
        "claude-3-7-sonnet-latest": {"input": 3.0 / 1e6, "cached_input": 0.75 / 1e6, "output": 15.0 / 1e6},
        "claude-sonnet-4-0": {"input": 3.0 / 1e6, "cached_input": 0.75 / 1e6, "output": 15.0 / 1e6},
        "claude-opus-4-0": {"input": 15.0 / 1e6, "cached_input": 3.75 / 1e6, "output": 75.0 / 1e6},
        "claude-opus-4-1": {"input": 15.0 / 1e6, "cached_input": 3.75 / 1e6, "output": 75.0 / 1e6},
        "claude-opus-4-5": {"input": 15.0 / 1e6, "cached_input": 3.75 / 1e6, "output": 75.0 / 1e6},
        "claude-sonnet-4-5": {"input": 3.0 / 1e6, "cached_input": 0.75 / 1e6, "output": 15.0 / 1e6},
        "claude-haiku-4-5": {"input": 1.0 / 1e6, "cached_input": 0.25 / 1e6, "output": 5.0 / 1e6},
    }

    google_pricing_per_model = {
        # Gemini 2.5 family (approximate)
        "gemini-2.5-pro": {"input": 2.5 / 1e6, "cached_input": 0.63 / 1e6, "output": 10.0 / 1e6},
        "gemini-2.5-flash": {"input": 0.5 / 1e6, "cached_input": 0.13 / 1e6, "output": 2.0 / 1e6},
    }

    deepseek_pricing_per_model = {
        "deepseek-reasoner": {"input": 0.6 / 1e6, "cached_input": 0.15 / 1e6, "output": 2.4 / 1e6},
    }

    # Groq/Cerebras pricing is provider-specific and can vary by routed model; keep conservative defaults.
    groq_pricing_default = {"input": 0.3 / 1e6, "cached_input": 0.08 / 1e6, "output": 1.2 / 1e6}
    cerebras_pricing_default = {"input": 0.3 / 1e6, "cached_input": 0.08 / 1e6, "output": 1.2 / 1e6}

    pricing: dict[tuple[str, str | None], dict] = {}
    search_call_cost = 27.5 / 1e3  # $/call (per 1k calls); applied to search-like engines

    # Ensure we cover all supported OpenAI models (chat + reasoning) and their responses: variants.
    for model_name in set(OPENAI_CHAT_MODELS + OPENAI_REASONING_MODELS):
        model_pricing = openai_pricing_per_model.get(model_name)
        if model_pricing is None:
            # Fallback: treat unknown-but-supported models like a mid-tier chat model.
            model_pricing = openai_pricing_per_model["gpt-4.1-mini"]
        pricing[("GPTXChatEngine", model_name)] = dict(model_pricing)
        pricing[("GPTXReasoningEngine", model_name)] = dict(model_pricing)
        pricing[("GPTXSearchEngine", model_name)] = {**model_pricing, "calls": search_call_cost}
        pricing[("OpenAIResponsesEngine", f"responses:{model_name}")] = dict(model_pricing)

    # Add explicit responses-only aliases.
    for model_name in ("gpt-5-pro", "o3-pro", "gpt-5.2-pro"):
        model_pricing = openai_pricing_per_model[model_name]
        pricing[("OpenAIResponsesEngine", f"responses:{model_name}")] = dict(model_pricing)

    usage_per_engine = RuntimeInfo.from_tracker(tracker, 0)
    # Add fallback pricing for any other supported engines observed in the tracker.
    for engine_model in usage_per_engine.keys():
        if engine_model in pricing:
            continue
        engine_name, model_name = engine_model
        if engine_name in ("ClaudeXChatEngine", "ClaudeXReasoningEngine"):
            engine_pricing = dict(
                anthropic_pricing_per_model.get(
                    model_name or "", {"input": 3.0 / 1e6, "cached_input": 0.75 / 1e6, "output": 15.0 / 1e6}
                )
            )
        elif engine_name == "GeminiXReasoningEngine":
            engine_pricing = dict(
                google_pricing_per_model.get(
                    model_name or "", {"input": 2.0 / 1e6, "cached_input": 0.5 / 1e6, "output": 8.0 / 1e6}
                )
            )
        elif engine_name == "DeepSeekXReasoningEngine":
            engine_pricing = dict(
                deepseek_pricing_per_model.get(
                    model_name or "", {"input": 0.6 / 1e6, "cached_input": 0.15 / 1e6, "output": 2.4 / 1e6}
                )
            )
        elif engine_name == "GroqEngine":
            engine_pricing = dict(groq_pricing_default)
        elif engine_name == "CerebrasEngine":
            engine_pricing = dict(cerebras_pricing_default)
        elif engine_name == "ParallelEngine":
            # ParallelEngine does not have per-model token accounting; keep cost estimation stable.
            engine_pricing = {"input": 0.0, "cached_input": 0.0, "output": 0.0}
        elif engine_name in ("GPTXSearchEngine",):
            # Should already be covered above, but keep safe.
            engine_pricing = {**openai_pricing_per_model["gpt-4.1-mini"], "calls": search_call_cost}
        else:
            engine_pricing = {"input": 1.0 / 1e6, "cached_input": 0.25 / 1e6, "output": 2.0 / 1e6}

        if engine_name == "GPTXSearchEngine":
            engine_pricing.setdefault("calls", search_call_cost)
        pricing[(engine_name, model_name)] = engine_pricing

    # total_elapsed_time, prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens, total_calls, total_tokens, cost_estimate
    usage = RuntimeInfo(0, 0, 0, 0, 0, 0, 0, 0)
    for (engine_name, model_name), data in usage_per_engine.items():
        usage += RuntimeInfo.estimate_cost(data, estimate_cost, pricing=pricing[(engine_name, model_name)])
    # Add total elapsed time to the actual elapsed time
    usage.total_elapsed_time = end_time - start_time

    assert usage.total_elapsed_time > 0, "Expected elapsed time to be greater than 0."
    assert usage.prompt_tokens > 0, "Expected prompt tokens to be greater than 0."
    assert usage.completion_tokens > 0, "Expected completion tokens to be greater than 0."
    assert usage.total_tokens > 0, "Expected total tokens to be greater than 0."
    assert usage.cost_estimate >= 0, "Expected cost estimate to be non-negative."
    # Prompt caching may be enabled server-side and can yield non-zero cached tokens.
    assert usage.cached_tokens >= 0, "Expected cached tokens to be non-negative."
    assert usage.cached_tokens <= usage.prompt_tokens, "Cached tokens cannot exceed prompt tokens."
    assert usage.reasoning_tokens >= 0, "Expected reasoning tokens to be non-negative."
    assert usage.reasoning_tokens <= usage.total_tokens, "Reasoning tokens cannot exceed total tokens."
    expected_calls = int(did_neurosymbolic) + int(did_search)
    assert usage.total_calls == expected_calls, f"Expected total_calls to be {expected_calls}."


def test_disable_primitives():
    sym1 = Symbol("This is a test")
    sym2 = Symbol("This is another test")
    sym3 = Expression("This is a test")

    with PrimitiveDisabler():
        # disable primitives
        assert all([
            sym1.query('Is this a test?') is None,
            sym2.query('Is this a test?') is None,
            sym3.query('Is this a test?') is None,
            ])

    # re-enable primitives
    assert all([
        sym1.query('Is this a test?') is not None,
        sym2.query('Is this a test?') is not None,
        sym3.query('Is this a test?') is not None,
        ])


def test_dynamic_engine_switching():
    # Fetch API keys from environment variables:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    if openai_api_key is None or anthropic_api_key is None:
        raise OSError(
            "Missing environment variables: OPENAI_API_KEY and/or ANTHROPIC_API_KEY"
        )

    dynamic_engine1 = DynamicEngine('o1', openai_api_key)
    dynamic_engine2 = DynamicEngine('claude-3-5-sonnet-latest', anthropic_api_key)

    # Test with dynamic_engine1
    with dynamic_engine1:
        # EngineRepository.get should return the engine_instance from dynamic_engine1
        engine_in_context = EngineRepository.get('neurosymbolic')
        assert engine_in_context is dynamic_engine1.engine_instance, \
            "Should use dynamic_engine1 inside its context."

        # Simple check that the Symbol query does not return None
        resp1 = Symbol("You are an OpenAI model. "
                       "You have to tell which model you are specifically."
                      ).query("Which model are you?")
        assert resp1 is not None, \
            "Expected some non-None output from dynamic engine 1 within its context."

    # Once we're outside the dynamic_engine1 context, it should revert to default or another engine
    outside_engine1 = EngineRepository.get('neurosymbolic')
    assert outside_engine1 != dynamic_engine1.engine_instance, \
        "Should revert to a different engine outside dynamic_engine1 context."

    # Test with dynamic_engine2
    with dynamic_engine2:
        engine_in_context = EngineRepository.get('neurosymbolic')
        assert engine_in_context is dynamic_engine2.engine_instance, \
            "Should use dynamic_engine2 inside its context."

        resp2 = Symbol("You are an OpenAI model. "
                       "You have to tell which model you are specifically."
                      ).query("Which model are you?")
        assert resp2 is not None, \
            "Expected some non-None output from dynamic engine 2 within its context."

    # Outside of dynamic_engine2 context, it again reverts
    outside_engine2 = EngineRepository.get('neurosymbolic')
    assert outside_engine2 != dynamic_engine2.engine_instance, \
        "Should revert to a different engine outside dynamic_engine2 context."


def test_chonkie_chunker_pdf():
    """Test ChonkieChunker with a PDF file."""
    pdf_path = ""

    # Skip test if PDF file doesn't exist
    if not os.path.exists(pdf_path):
        pytest.skip(f"PDF file not found: {pdf_path}")

    # Check if chonkie is available by trying to import it
    try:
        from chonkie import RecursiveChunker
        from tokenizers import Tokenizer
    except ImportError:
        pytest.skip("chonkie library is not installed. Please install it with `pip install chonkie tokenizers`")

    # Read the PDF file using FileReader
    file_reader = FileReader()
    pdf_content_list = file_reader(pdf_path)
    assert pdf_content_list.value is not None, "PDF content should not be None"
    assert isinstance(pdf_content_list.value, list), "FileReader should return a list"
    assert len(pdf_content_list.value) > 0, "PDF content list should not be empty"

    # Get the first (and only) file content
    pdf_content = Symbol(pdf_content_list.value[0])
    assert len(str(pdf_content)) > 0, "PDF content should not be empty"

    # Create chunker and chunk the PDF
    chunker = ChonkieChunker()
    chunks = chunker(pdf_content, chunker_name="RecursiveChunker")

    # Verify chunks were created
    assert chunks.value is not None, "Chunks should not be None"
    assert isinstance(chunks.value, list), "Chunks should be a list"
    assert len(chunks.value) > 0, "Should create at least one chunk"

    # Verify each chunk is a non-empty string
    for chunk in chunks.value:
        assert isinstance(chunk, str), "Each chunk should be a string"
        assert len(chunk) > 0, "Each chunk should not be empty"
