import time
from pathlib import Path

import httpx
import pytest

from symai.backend.engines.embedding.gemini import GeminiEmbeddingEngine
from symai.backend.engines.embedding.gemini.models import (
    API_PINNED,
    GEMINI_API_BASE,
    GEMINI_EMBEDDING_MODEL_SPECS,
    GeminiBatchEmbedResponse,
)
from tests.engines.embedding.interface import (
    EmbeddingTestInterface,
    normalized_vector,
)
from tests.engines.mock_api import MockAPI

SAMPLE_IMAGE = Path(__file__).parent.parent.parent / "data" / "sample.png"
MULTIMODAL_MODEL = "gemini-embedding-2"
MULTIMODAL_DIMS = GEMINI_EMBEDDING_MODEL_SPECS[MULTIMODAL_MODEL][1]


class TestGeminiEmbeddingEngine(EmbeddingTestInterface):
    engine_cls = GeminiEmbeddingEngine
    response_cls = GeminiBatchEmbedResponse
    default_model = "gemini-embedding-001"
    expected_dims = GEMINI_EMBEDDING_MODEL_SPECS[default_model][1]
    wire_url = f"{GEMINI_API_BASE}/models/{default_model}:batchEmbedContents"
    auth_header_name = "x-goog-api-key"
    auth_header_prefix = ""
    api_pinned = API_PINNED
    api_pinned_module = "symai.backend.engines.embedding.gemini.models"
    api_key_env = "GOOGLE_API_KEY"
    supports_usage = False  # GeminiBatchEmbedResponse carries no usage; no tracker branch

    def mock_response_json(self):
        return {
            "embeddings": [
                {"values": normalized_vector(0.11)},
                {"values": normalized_vector(0.07)},
            ]
        }

    def response_dropping_required(self, payload):
        # embeddings is required (min 1 item) — dropping it must fail typed parsing.
        payload.pop("embeddings")
        return payload

    def expected_request_body_subset(self):
        return {
            "requests": [
                {
                    "model": f"models/{self.default_model}",
                    "content": {"parts": [{"text": "hello"}]},
                    "taskType": "SEMANTIC_SIMILARITY",
                },
                {
                    "model": f"models/{self.default_model}",
                    "content": {"parts": [{"text": "world"}]},
                    "taskType": "SEMANTIC_SIMILARITY",
                },
            ]
        }

    def test_new_dim_sent_on_wire_as_output_dimensionality(self):
        # Unlike OpenAI (client-side only), Gemini steers the server via
        # outputDimensionality per request entry AND re-normalizes client-side.
        api, output, _metadata = self.forward_through_mock(new_dim=4)

        for entry in api.last_body["requests"]:
            assert entry["outputDimensionality"] == 4
        assert all(len(vector) == 4 for vector in output[0])

    def test_no_mutable_new_dim_stash(self):
        # Regression: build_request used to stash new_dim on self._new_dim for
        # parse_response — a reentrancy hazard on a shared engine instance. new_dim
        # is now read back from the call argument in parse_response (like the old
        # SDK engine and the embedding/openai reference).
        engine = self.make_engine()
        assert not hasattr(engine, "_new_dim")

        _api, output, _metadata = self.forward_through_mock(new_dim=4)

        assert not hasattr(engine, "_new_dim")
        assert all(len(vector) == 4 for vector in output[0])

    def test_call_request_reraises_transport_error(self):
        engine = self.make_engine()
        argument = self.make_argument()

        with MockAPI(
            engine,
            lambda request: httpx.Response(
                500, json={"error": {"message": "boom"}}, request=request
            ),
        ):
            engine.prepare(argument)
            with pytest.raises(Exception, match="boom"):
                engine.forward(argument)

    @pytest.mark.engine_live
    def test_live_single_embedding(self, engine_api_mode):
        api_key = self.require_live(engine_api_mode)
        engine = self.make_live_engine(api_key)

        argument = self.make_argument(entries=["hello world"])
        engine.prepare(argument)
        output, _metadata = engine.forward(argument)

        assert len(output[0]) == 1
        assert isinstance(output[0][0], list)
        assert len(output[0][0]) == self.expected_dims
        assert all(isinstance(x, float) for x in output[0][0])

    TIMING_TEXTS = (
        "Machine learning transforms data into insights.",
        "Python is the dominant language for data science.",
        "Neural networks learn hierarchical representations.",
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


class TestGeminiEmbedding2Multimodal(TestGeminiEmbeddingEngine):
    """Live multimodal capabilities of gemini-embedding-2 (raw bytes)."""

    default_model = MULTIMODAL_MODEL
    expected_dims = MULTIMODAL_DIMS
    wire_url = f"{GEMINI_API_BASE}/models/{MULTIMODAL_MODEL}:batchEmbedContents"

    def embed_live(self, entries, api_key, **kwargs):
        engine = self.make_live_engine(api_key)
        argument = self.make_argument(entries=entries, kwargs=kwargs)
        engine.prepare(argument)
        return engine.forward(argument)[0][0]

    @pytest.mark.engine_live
    def test_live_text_embedding(self, engine_api_mode):
        api_key = self.require_live(engine_api_mode)

        vectors = self.embed_live(["hello world"], api_key)

        assert len(vectors) == 1
        assert len(vectors[0]) == MULTIMODAL_DIMS
        assert all(isinstance(x, float) for x in vectors[0])

    @pytest.mark.engine_live
    def test_live_image_embedding_from_bytes(self, engine_api_mode):
        api_key = self.require_live(engine_api_mode)

        vectors = self.embed_live([SAMPLE_IMAGE.read_bytes()], api_key)

        assert len(vectors) == 1
        assert len(vectors[0]) == MULTIMODAL_DIMS
        assert all(isinstance(x, float) for x in vectors[0])

    def test_part_or_content_inputs_raise_type_error(self):
        engine = self.make_engine()
        argument = self.make_argument(entries=[{"not": "str-or-bytes"}])
        engine.prepare(argument)

        with pytest.raises(TypeError, match="str and bytes"):
            engine.forward(argument)

    @pytest.mark.engine_live
    def test_live_truncated_multimodal_embedding(self, engine_api_mode):
        api_key = self.require_live(engine_api_mode)
        vectors = self.embed_live([SAMPLE_IMAGE.read_bytes()], api_key, new_dim=768)

        assert len(vectors[0]) == 768

    @pytest.mark.engine_live
    def test_live_batch_mixed_inputs(self, engine_api_mode):
        api_key = self.require_live(engine_api_mode)
        vectors = self.embed_live(["hello world", SAMPLE_IMAGE.read_bytes()], api_key)

        # NOTE: the batch endpoint embeds every request entry separately.
        assert len(vectors) >= 1
        assert all(len(emb) == MULTIMODAL_DIMS for emb in vectors)
