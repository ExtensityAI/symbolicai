import os
import time

import pytest

from symai import Expression, PrimitiveDisabler, Symbol
from symai.backend.mixin import OPENAI_CHAT_MODELS, OPENAI_REASONING_MODELS
from symai.backend.settings import SYMAI_CONFIG
from symai.components import DynamicEngine, MetadataTracker
from symai.extended import Interface
from symai.functional import EngineRepository
from symai.symbol import Metadata
from symai.utils import RuntimeInfo

NEUROSYMBOLIC_ENGINE = SYMAI_CONFIG.get('NEUROSYMBOLIC_ENGINE_MODEL', '') in OPENAI_CHAT_MODELS + OPENAI_REASONING_MODELS
SEARCH_ENGINE = SYMAI_CONFIG.get('SEARCH_ENGINE_MODEL', '') in OPENAI_CHAT_MODELS + OPENAI_REASONING_MODELS


def estimate_cost(info: dict, pricing: dict) -> float:
    input_cost = (info.prompt_tokens - info.cached_tokens) * pricing.get("input", 0)
    cached_input_cost = info.cached_tokens * pricing.get("cached_input", 0)
    output_cost = info.completion_tokens * pricing.get("output", 0)
    return input_cost + cached_input_cost + output_cost


@pytest.mark.skipif(not NEUROSYMBOLIC_ENGINE and not SEARCH_ENGINE, reason="Tested only for OpenAI backend for now.")
def test_metadata_tracker_with_runtimeinfo():
    sym = Symbol("You're supposed to go off the rails and say something pretty crazy.")
    search = Interface('openai_search')
    start_time = time.perf_counter()
    with MetadataTracker() as tracker:
        res = sym.query("What is the meaning of life?")
        search_res = search("What is the meaning of life?")
    end_time = time.perf_counter()
    assert res.value is not None and search_res.value is not None, "Expected some non-None output from the query."

    dummy_pricing = {
        "GPTXChatEngine": {
            "input": 0.000002,
            "cached_input": 0.000001,
            "output": 0.000002
        },
        "GPTXSearchEngine": {
            "input": 0.000002,
            "cached_input": 0.000001,
            "output": 0.000002
        }
    }
    usage_per_engine = RuntimeInfo.from_tracker(tracker, 0)
    usage = RuntimeInfo(0, 0, 0, 0, 0, 0)
    for engine_name, data in usage_per_engine.items():
        usage += RuntimeInfo.estimate_cost(data, estimate_cost, pricing=dummy_pricing[engine_name])
    # Add total elapsed time to the actual elapsed time
    usage.total_elapsed_time = end_time - start_time

    assert usage.total_elapsed_time > 0, "Expected elapsed time to be greater than 0."
    assert usage.prompt_tokens > 0, "Expected prompt tokens to be greater than 0."
    assert usage.completion_tokens > 0, "Expected completion tokens to be greater than 0."
    assert usage.total_tokens > 0, "Expected total tokens to be greater than 0."
    assert usage.cost_estimate > 0, "Expected cost estimate to be greater than 0."
    assert usage.cached_tokens == 0, "Expected cached tokens to be 0."


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
        raise EnvironmentError(
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
