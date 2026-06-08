import time
from pathlib import Path

import pytest
from google.genai.types import Content, Part

from symai import Symbol
from symai.backend.settings import SYMAI_CONFIG

MODEL = SYMAI_CONFIG.get("EMBEDDING_ENGINE_MODEL", "")
IS_GEMINI = "gemini-embedding" in MODEL
SKIP_MSG = f"EMBEDDING_ENGINE_MODEL={MODEL!r} is not a Gemini embedding model"

API_KEY = SYMAI_CONFIG.get("EMBEDDING_ENGINE_API_KEY", "")
HAS_API_KEY = bool(API_KEY)
SAMPLE_IMAGE = Path(__file__).parent.parent.parent / "data" / "sample.png"


@pytest.mark.skipif(not IS_GEMINI, reason=SKIP_MSG)
def test_single_embedding():
    result = Symbol("hello world").embed()
    assert len(result.value) == 1
    assert isinstance(result.value[0], list)
    assert len(result.value[0]) > 0
    assert all(isinstance(x, float) for x in result.value[0])


@pytest.mark.skipif(not IS_GEMINI, reason=SKIP_MSG)
def test_batched_embedding():
    result = Symbol(["hello", "world"]).embed()
    # gemini-embedding-001 returns individual embeddings
    # gemini-embedding-2 aggregates multiple inputs into one embedding
    if "gemini-embedding-2" in MODEL:
        assert len(result.value) >= 1
    else:
        assert len(result.value) == 2
    assert all(isinstance(emb, list) for emb in result.value)
    assert all(len(emb) > 0 for emb in result.value)


@pytest.mark.skipif(not IS_GEMINI, reason=SKIP_MSG)
def test_truncated_embedding():
    result = Symbol(["hello"]).embed(new_dim=128)
    assert len(result.value[0]) == 128


@pytest.mark.skipif(not IS_GEMINI, reason=SKIP_MSG)
class TestGeminiEmbedBatching:
    """Verify that a single batched embed call is faster than N sequential ones."""

    TEXTS = [
        "Machine learning transforms data into insights.",
        "Python is the dominant language for data science.",
        "Neural networks learn hierarchical representations.",
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


@pytest.mark.skipif(not HAS_API_KEY, reason="No EMBEDDING_ENGINE_API_KEY configured")
@pytest.mark.skipif("gemini-embedding-2" not in MODEL, reason="Requires gemini-embedding-2 model")
class TestGeminiEmbedding2Multimodal:
    """Tests for gemini-embedding-2 multimodal capabilities using Symbol.embed()."""

    def test_text_embedding(self):
        """Test text embedding via Symbol.embed()."""
        result = Symbol("hello world").embed()
        assert len(result.value) == 1
        assert len(result.value[0]) == 3072
        assert all(isinstance(x, float) for x in result.value[0])

    def test_image_embedding_from_bytes(self):
        """Test image embedding from raw bytes."""
        image_bytes = SAMPLE_IMAGE.read_bytes()
        result = Symbol(image_bytes).embed()
        assert len(result.value) == 1
        assert len(result.value[0]) == 3072
        assert all(isinstance(x, float) for x in result.value[0])

    def test_image_embedding_from_part(self):
        """Test image embedding from Google Part object."""
        image_bytes = SAMPLE_IMAGE.read_bytes()
        image_part = Part.from_bytes(data=image_bytes, mime_type="image/png")
        result = Symbol(image_part).embed()
        assert len(result.value) == 1
        assert len(result.value[0]) == 3072
        assert all(isinstance(x, float) for x in result.value[0])

    def test_multimodal_embedding_text_and_image(self):
        """Test combined text and image embedding."""
        image_bytes = SAMPLE_IMAGE.read_bytes()
        content = Content(
            parts=[
                Part.from_text(text="Describe this image"),
                Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ]
        )
        result = Symbol(content).embed()
        assert len(result.value) == 1
        assert len(result.value[0]) == 3072
        assert all(isinstance(x, float) for x in result.value[0])

    def test_truncated_multimodal_embedding(self):
        """Test dimension reduction with multimodal input."""
        image_bytes = SAMPLE_IMAGE.read_bytes()
        image_part = Part.from_bytes(data=image_bytes, mime_type="image/png")
        result = Symbol(image_part).embed(new_dim=768)
        assert len(result.value[0]) == 768

    def test_batch_mixed_inputs(self):
        """Test batch embedding with mixed text and image inputs."""
        image_bytes = SAMPLE_IMAGE.read_bytes()
        image_part = Part.from_bytes(data=image_bytes, mime_type="image/png")
        
        # Batch with text and image
        result = Symbol(["hello world", image_part]).embed()
        # Note: gemini-embedding-2 aggregates multiple inputs into one embedding
        assert len(result.value) >= 1
        assert all(len(emb) == 3072 for emb in result.value)
