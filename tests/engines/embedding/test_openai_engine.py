import time

import pytest

from symai import Symbol
from symai.backend.settings import SYMAI_CONFIG

MODEL = SYMAI_CONFIG.get("EMBEDDING_ENGINE_MODEL", "")
IS_OPENAI = MODEL.startswith("text-embedding")
SKIP_MSG = f"EMBEDDING_ENGINE_MODEL={MODEL!r} is not an OpenAI model"


@pytest.mark.skipif(not IS_OPENAI, reason=SKIP_MSG)
def test_batched_embedding():
    result = Symbol(["hello", "world"]).embed()
    assert len(result.value) == 2


@pytest.mark.skipif(not IS_OPENAI, reason=SKIP_MSG)
def test_truncated_embedding():
    result = Symbol(["hello"]).embed(new_dim=128)
    assert len(result.value[0]) == 128


@pytest.mark.skipif(not IS_OPENAI, reason=SKIP_MSG)
class TestEmbedBatching:
    """Verify that a single batched embed call is faster than N sequential ones."""

    TEXTS = [
        "Machine learning transforms data into insights.",
        "Python is the dominant language for data science.",
        "Neural networks learn hierarchical representations.",
        "Qdrant stores and retrieves dense vector embeddings.",
        "Transformers revolutionized natural language processing.",
        "Embeddings map semantic meaning into vector space.",
        "Cosine similarity measures the angle between vectors.",
        "Batching amortizes the fixed HTTP round-trip cost.",
    ]

    def test_batch_embed_faster_than_sequential(self):
        t0 = time.perf_counter()
        for text in self.TEXTS:
            Symbol(text).embed()
        sequential_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        Symbol(self.TEXTS).embed()
        batch_time = time.perf_counter() - t0

        print(f"\nSequential ({len(self.TEXTS)} calls): {sequential_time:.3f}s")
        print(f"Batch      (1 call):              {batch_time:.3f}s")
        print(f"Speedup: {sequential_time / batch_time:.1f}x")

        assert sequential_time > batch_time, (
            f"Batch embed ({batch_time:.3f}s) should be faster than "
            f"sequential embed ({sequential_time:.3f}s)"
        )
