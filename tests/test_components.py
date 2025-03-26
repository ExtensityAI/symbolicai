import os

import pytest

from symai import Expression, PrimitiveDisabler, Symbol
from symai.components import DynamicEngine
from symai.functional import EngineRepository


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


if __name__ == "__main__":
    pytest.main()
