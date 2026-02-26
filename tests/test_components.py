import os

import pytest

from symai import Expression, PrimitiveDisabler, Symbol
from symai.backend.settings import SYMAI_CONFIG
from symai.components import ChonkieChunker, DynamicEngine, FileReader, MetadataTracker
from symai.functional import EngineRepository
from symai.utils import RuntimeInfo

NEUROSYMBOLIC_MODEL = SYMAI_CONFIG.get("NEUROSYMBOLIC_ENGINE_MODEL", "")
NEUROSYMBOLIC_ENGINE = bool(NEUROSYMBOLIC_MODEL)


def estimate_cost(info: RuntimeInfo, pricing: dict) -> float:
    input_cost = (info.prompt_tokens - info.cached_tokens) * pricing.get("input", 0)
    cached_input_cost = info.cached_tokens * pricing.get("cached_input", 0)
    output_cost = info.completion_tokens * pricing.get("output", 0)
    call_cost = info.total_calls * pricing.get("calls", 0)
    return input_cost + cached_input_cost + output_cost + call_cost


@pytest.mark.skipif(not NEUROSYMBOLIC_ENGINE, reason="Requires a configured neurosymbolic engine.")
def test_metadata_tracker_with_runtimeinfo():
    sym = Symbol("You're supposed to go off the rails and say something pretty crazy.")
    with MetadataTracker() as tracker:
        res = sym.query("What is the meaning of life?")

    assert res is not None and res.value is not None

    dummy_pricing = {"input": 1.0, "cached_input": 0.5, "output": 2.0}
    usage_per_engine = RuntimeInfo.from_tracker(tracker, 0)
    usage = RuntimeInfo(0, 0, 0, 0, 0, 0, 0, 0)
    for _, data in usage_per_engine.items():
        usage += RuntimeInfo.estimate_cost(data, estimate_cost, pricing=dummy_pricing)

    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.cost_estimate > 0


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

    dynamic_engine1 = DynamicEngine('o3', openai_api_key)
    dynamic_engine2 = DynamicEngine('claude-opus-4-6', anthropic_api_key)

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
