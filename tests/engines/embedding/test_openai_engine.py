import time

import pytest

from symai.backend.engines.embedding.openai import EmbeddingEngine
from symai.backend.engines.embedding.openai.models import (
    API_PINNED,
    OPENAI_EMBEDDING_MODEL_SPECS,
    OPENAI_EMBEDDINGS_URL,
    OpenAIEmbeddingResponse,
)
from tests.engines.embedding.interface import (
    EmbeddingTestInterface,
    normalized_vector,
)


class TestOpenAIEmbeddingEngine(EmbeddingTestInterface):
    engine_cls = EmbeddingEngine
    response_cls = OpenAIEmbeddingResponse
    default_model = "text-embedding-3-small"
    expected_dims = OPENAI_EMBEDDING_MODEL_SPECS[default_model][1]
    wire_url = OPENAI_EMBEDDINGS_URL
    auth_header_name = "Authorization"
    auth_header_prefix = "Bearer "
    api_pinned = API_PINNED
    api_pinned_module = "symai.backend.engines.embedding.openai.models"
    keys_log_section = "openai"
    keys_log_pattern = r'"(sk-proj-[^"]+)"'
    supports_usage = True

    def mock_response_json(self):
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": normalized_vector(0.11),
                    "index": 0,
                },
                {
                    "object": "embedding",
                    "embedding": normalized_vector(0.07),
                    "index": 1,
                },
            ],
            "model": self.default_model,
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        }

    def response_dropping_required(self, payload):
        # usage is required: the MetadataTracker "EmbeddingEngine" branch reads
        # raw_output.usage.prompt_tokens / .total_tokens — dropping it must fail fast.
        payload.pop("usage")
        return payload

    def expected_request_body_subset(self):
        return {"model": self.default_model, "input": list(self.MOCK_ENTRIES)}

    def test_wire_payload_omits_dimensions(self):
        # NOTE: `dimensions` is intentionally NOT sent — new_dim truncation is
        # client-side (L2 re-normalized); the wire payload is exactly {model, input}.
        api, _output, _metadata = self.forward_through_mock(new_dim=128)

        assert "dimensions" not in api.last_body

    TIMING_TEXTS = (
        "Machine learning transforms data into insights.",
        "Python is the dominant language for data science.",
        "Neural networks learn hierarchical representations.",
        "Qdrant stores and retrieves dense vector embeddings.",
        "Transformers revolutionized natural language processing.",
        "Embeddings map semantic meaning into vector space.",
        "Cosine similarity measures the angle between vectors.",
        "Batching amortizes the fixed HTTP round-trip cost.",
    )

    @pytest.mark.engine_live
    def test_live_batch_embed_faster_than_sequential(self, engine_api_mode):
        """Perf smoke: one batched embed call beats N sequential ones (live only)."""
        api_key = self.require_live(engine_api_mode)
        engine = self.make_live_engine(api_key)

        def embed(entries):
            argument = self.make_argument(entries=entries)
            engine.prepare(argument)
            return engine.forward(argument)[0][0]

        t0 = time.perf_counter()
        for text in self.TIMING_TEXTS:
            assert len(embed([text])) == 1
        sequential_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        vectors = embed(list(self.TIMING_TEXTS))
        batch_time = time.perf_counter() - t0

        assert len(vectors) == len(self.TIMING_TEXTS)

        # NOTE: tolerant perf smoke — batching must not be slower than sequential,
        # but no minimum speedup factor is demanded.
        assert sequential_time > batch_time, (
            f"Batch embed ({batch_time:.3f}s) should be faster than "
            f"sequential embed ({sequential_time:.3f}s)"
        )
