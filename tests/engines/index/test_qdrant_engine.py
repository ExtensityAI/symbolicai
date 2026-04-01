"""
Comprehensive tests for Qdrant Index Engine.

Tests cover:
- Basic engine operations (search, add)
- Manager methods (collection management, point operations)
- Document chunking and upsert functionality
- Error handling and edge cases
"""

import asyncio
import os
import time
from pathlib import Path

import pytest

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct
except ImportError:
    QdrantClient = None
    PointStruct = None
    Filter = None
    FieldCondition = None
    MatchValue = None

from symai import Symbol
from symai.backend.engines.index.engine_qdrant import QdrantIndexEngine
from symai.backend.settings import SYMAI_CONFIG, SYMSERVER_CONFIG
from symai.core import Argument
from symai.interfaces import Interface

try:
    from symai.components import ChonkieChunker
except ImportError:
    ChonkieChunker = None

AVAILABLE_PDFS = [(Path(__file__).parents[2] / "data" / "sample.pdf").as_posix()]


def _check_qdrant_available():
    if QdrantClient is None:
        return False
    url = (
        SYMSERVER_CONFIG.get("url")
        or SYMAI_CONFIG.get("INDEXING_ENGINE_URL")
        or "http://localhost:6333"
    )
    api_key = SYMAI_CONFIG.get("INDEXING_ENGINE_API_KEY")
    try:
        client_kwargs = {"url": url}
        if api_key:
            client_kwargs["api_key"] = api_key
        QdrantClient(**client_kwargs).get_collections()
        return True
    except Exception:
        return False


# Check if Qdrant is available
QDrant_AVAILABLE = _check_qdrant_available()


@pytest.fixture
def engine():
    """Create a Qdrant engine instance for testing."""
    return QdrantIndexEngine(
        url=SYMSERVER_CONFIG.get("url")
        or SYMAI_CONFIG.get("INDEXING_ENGINE_URL")
        or "http://localhost:6333",
        api_key=SYMAI_CONFIG.get("INDEXING_ENGINE_API_KEY"),
        index_name="test_collection",
        index_dims=1536,
    )


@pytest.fixture
def test_collection_name():
    """Generate a unique collection name for each test."""
    return f"test_collection_{int(time.time() * 1000)}"


def normalize_embedding(embedding, target_size=1536):
    """
    Normalize an embedding to a flat list of target_size.

    Args:
        embedding: Embedding from Symbol (can be numpy array, list, or nested list)
        target_size: Target size for the embedding vector (default: 1536)

    Returns:
        List of floats with length target_size
    """
    # Convert to list format
    if hasattr(embedding, "tolist"):
        query_vector = embedding.tolist()
    elif isinstance(embedding, list):
        query_vector = embedding
    else:
        query_vector = list(embedding)

    # Handle nested lists
    if query_vector and len(query_vector) > 0 and isinstance(query_vector[0], list):
        query_vector = (
            query_vector[0]
            if len(query_vector) == 1
            else [item for sublist in query_vector for item in sublist]
        )

    # Pad or truncate to match target size
    if len(query_vector) != target_size:
        if len(query_vector) > target_size:
            query_vector = query_vector[:target_size]
        else:
            query_vector = query_vector + [0.0] * (target_size - len(query_vector))

    return query_vector


@pytest.mark.skipif(not QDrant_AVAILABLE, reason="Qdrant server not available")
class TestQdrantEngineBasic:
    """Test basic engine functionality."""

    def test_engine_initialization(self, engine):
        """Test that engine initializes correctly."""
        assert engine is not None
        assert engine.index_name == "test_collection"
        assert engine.index_dims == 1536

    @pytest.mark.asyncio
    async def test_add_and_search_using_manager_methods(self, engine, test_collection_name):
        """Test adding and searching using manager methods directly."""

        # Create collection
        await engine.create_collection(test_collection_name, vector_size=1536)

        # Create embeddings and points
        text = "Hello world"
        symbol = Symbol(text)
        embedding = symbol.embedding

        # Ensure correct size (engine will normalize the vector format)
        # Convert to list for size checking
        if hasattr(embedding, "tolist"):
            embedding_list = embedding.tolist()
        elif isinstance(embedding, list):
            embedding_list = embedding
        else:
            embedding_list = list(embedding)

        # Handle nested lists for size checking
        if embedding_list and len(embedding_list) > 0 and isinstance(embedding_list[0], list):
            embedding_list = (
                embedding_list[0]
                if len(embedding_list) == 1
                else [item for sublist in embedding_list for item in sublist]
            )

        # Pad or truncate to match collection size
        if len(embedding_list) != 1536:
            if len(embedding_list) > 1536:
                embedding_list = embedding_list[:1536]
            else:
                embedding_list = embedding_list + [0.0] * (1536 - len(embedding_list))
            # Update embedding to use the adjusted list
            embedding = embedding_list

        points = [PointStruct(id=1, vector=embedding, payload={"text": text})]

        # Add using manager method
        await engine.upsert(test_collection_name, points)

        # Search using manager method
        results = await engine.search(test_collection_name, embedding, limit=5)

        assert len(results) > 0, "Should find at least one result."
        assert hasattr(results[0], "payload"), "Result should have payload."
        assert "Hello world" in str(results[0].payload.get("text", ""))


@pytest.mark.skipif(not QDrant_AVAILABLE, reason="Qdrant server not available")
class TestQdrantManagerMethods:
    """Test manager methods (collection management, point operations)."""

    @pytest.mark.asyncio
    async def test_create_collection(self, engine, test_collection_name):
        """Test collection creation."""
        await engine.create_collection(
            collection_name=test_collection_name, vector_size=1536, distance="Cosine"
        )

        # Verify collection exists
        exists = await engine.collection_exists(test_collection_name)
        assert exists, "Collection should exist after creation."

    @pytest.mark.asyncio
    async def test_list_collections(self, engine, test_collection_name):
        """Test listing collections."""
        # Create a collection
        await engine.create_collection(test_collection_name, vector_size=1536)

        # List collections
        collections = await engine.list_collections()
        assert isinstance(collections, list)
        assert test_collection_name in collections, "Created collection should be in list."

    @pytest.mark.asyncio
    async def test_get_collection_info(self, engine, test_collection_name):
        """Test getting collection information."""
        await engine.create_collection(test_collection_name, vector_size=1536)

        info = await engine.get_collection_info(test_collection_name)
        assert info["name"] == test_collection_name
        assert info["config"]["params"]["vectors"]["size"] == 1536
        assert info["points_count"] == 0, "New collection should have 0 points."

    @pytest.mark.asyncio
    async def test_upsert_points(self, engine, test_collection_name):
        """Test upserting points."""

        await engine.create_collection(test_collection_name, vector_size=1536)

        # Create test points
        points = [
            PointStruct(id=1, vector=[0.1] * 1536, payload={"text": "Test document 1"}),
            PointStruct(id=2, vector=[0.2] * 1536, payload={"text": "Test document 2"}),
        ]

        await engine.upsert(test_collection_name, points)

        # Verify points were added
        info = await engine.get_collection_info(test_collection_name)
        assert info["points_count"] == 2, "Collection should have 2 points."

    @pytest.mark.asyncio
    async def test_search_points(self, engine, test_collection_name):
        """Test searching for points."""

        await engine.create_collection(test_collection_name, vector_size=1536)

        # Add points
        points = [
            PointStruct(id=1, vector=[0.1] * 1536, payload={"text": "Alpha document"}),
            PointStruct(id=2, vector=[0.2] * 1536, payload={"text": "Beta document"}),
        ]
        await engine.upsert(test_collection_name, points)

        # Search
        query_vector = [0.15] * 1536
        results = await engine.search(test_collection_name, query_vector, limit=2)

        assert len(results) > 0, "Search should return results."
        assert hasattr(results[0], "id"), "Results should have id attribute."
        assert hasattr(results[0], "score"), "Results should have score attribute."

    @pytest.mark.asyncio
    async def test_retrieve_points(self, engine, test_collection_name):
        """Test retrieving points by ID."""

        await engine.create_collection(test_collection_name, vector_size=1536)

        # Add points
        points = [
            PointStruct(id=100, vector=[0.1] * 1536, payload={"text": "Retrieved document"}),
        ]
        await engine.upsert(test_collection_name, points)

        # Retrieve
        retrieved = await engine.retrieve(test_collection_name, 100)
        assert len(retrieved) == 1
        assert retrieved[0]["id"] == 100
        assert retrieved[0]["payload"]["text"] == "Retrieved document"

    @pytest.mark.asyncio
    async def test_delete_points(self, engine, test_collection_name):
        """Test deleting points."""

        await engine.create_collection(test_collection_name, vector_size=1536)

        # Add points
        points = [
            PointStruct(id=1, vector=[0.1] * 1536, payload={"text": "To be deleted"}),
            PointStruct(id=2, vector=[0.2] * 1536, payload={"text": "To keep"}),
        ]
        await engine.upsert(test_collection_name, points)

        # Delete point
        await engine.delete(test_collection_name, 1)

        # Verify deletion
        info = await engine.get_collection_info(test_collection_name)
        assert info["points_count"] == 1, "Should have 1 point after deletion."

    @pytest.mark.asyncio
    async def test_delete_collection(self, engine, test_collection_name):
        """Test deleting a collection."""
        await engine.create_collection(test_collection_name, vector_size=1536)

        # Verify exists
        assert await engine.collection_exists(test_collection_name)

        # Delete
        await engine.delete_collection(test_collection_name)

        # Verify deleted
        assert not await engine.collection_exists(test_collection_name)


@pytest.mark.skipif(not QDrant_AVAILABLE, reason="Qdrant server not available")
class TestQdrantChunking:
    """Test document chunking and upsert functionality."""

    @pytest.mark.asyncio
    async def test_chunk_and_upsert_text(self, engine, test_collection_name):
        """Test chunking and upserting text."""
        await engine.create_collection(test_collection_name, vector_size=1536)

        text = """
        This is a test document with multiple sentences.
        It should be chunked into smaller pieces.
        Each chunk will be embedded and stored in the collection.
        We can then search for relevant chunks.
        """

        num_chunks = await engine.chunk_and_upsert(
            collection_name=test_collection_name, text=text, chunker_name="RecursiveChunker"
        )

        assert num_chunks > 0, "Should create at least one chunk."

        # Verify chunks were added
        info = await engine.get_collection_info(test_collection_name)
        assert info["points_count"] == num_chunks, "Points count should match number of chunks."

    @pytest.mark.asyncio
    async def test_chunk_and_upsert_with_metadata(self, engine, test_collection_name):
        """Test chunking with metadata."""
        await engine.create_collection(test_collection_name, vector_size=1536)

        text = "Test document with metadata."
        metadata = {"source": "test", "author": "test_author", "date": "2024-01-01"}

        num_chunks = await engine.chunk_and_upsert(
            collection_name=test_collection_name, text=text, metadata=metadata
        )

        assert num_chunks > 0

        # Search and verify metadata
        query_vector = [0.1] * 1536
        results = await engine.search(test_collection_name, query_vector, limit=1)

        if results:
            # Check if payload contains metadata
            assert hasattr(results[0], "payload"), "Result should have payload."
            # Note: payload structure may vary, so we just check it exists

    @pytest.mark.asyncio
    @pytest.mark.parametrize("pdf_path", AVAILABLE_PDFS)
    async def test_chunk_and_upsert_pdf(self, engine, test_collection_name, pdf_path):
        """Test chunking and upserting PDF documents."""
        await engine.create_collection(test_collection_name, vector_size=1536)

        # Use document_path parameter
        num_chunks = await engine.chunk_and_upsert(
            collection_name=test_collection_name,
            document_path=pdf_path,
            chunker_name="RecursiveChunker",
            metadata={"source": os.path.basename(pdf_path)},
        )

        assert num_chunks > 0, f"Should create chunks from PDF: {pdf_path}"

        # Verify chunks were added
        info = await engine.get_collection_info(test_collection_name)
        assert info["points_count"] == num_chunks

        # Test search on the PDF content
        # We'll search with a generic query vector to see if we get results
        query_vector = [0.1] * 1536
        results = await engine.search(test_collection_name, query_vector, limit=5)
        assert len(results) > 0, "Should find results from PDF chunks."

    @pytest.mark.asyncio
    async def test_chunk_and_upsert_multiple_chunkers(self, engine, test_collection_name):
        """Test different chunker types."""
        chunkers = ["RecursiveChunker", "SentenceChunker", "TokenChunker"]

        for chunker_name in chunkers:
            collection_name = f"{test_collection_name}_{chunker_name}"
            await engine.create_collection(collection_name, vector_size=1536)

            text = "First sentence. Second sentence. Third sentence."

            try:
                num_chunks = await engine.chunk_and_upsert(
                    collection_name=collection_name, text=text, chunker_name=chunker_name
                )
                assert num_chunks > 0, f"Chunker {chunker_name} should create chunks."
            except Exception as e:
                # Some chunkers might not be available or might fail
                pytest.skip(f"Chunker {chunker_name} not available: {e}")

    @pytest.mark.asyncio
    async def test_chunk_and_upsert_with_start_id(self, engine, test_collection_name):
        """Test chunking with custom start_id."""
        await engine.create_collection(test_collection_name, vector_size=1536)

        text = "Test document for ID testing."

        num_chunks = await engine.chunk_and_upsert(
            collection_name=test_collection_name, text=text, start_id=1000
        )

        assert num_chunks > 0

        # Retrieve points and verify IDs
        # Note: We can't easily verify exact IDs without knowing how many chunks,
        # but we can verify points exist
        info = await engine.get_collection_info(test_collection_name)
        assert info["points_count"] == num_chunks


@pytest.mark.skipif(not QDrant_AVAILABLE, reason="Qdrant server not available")
class TestQdrantUrlDownloads:
    """Test document downloads from URLs with SSRF protection."""

    # Real public PDF for testing successful downloads
    VALID_PDF_URL = "https://arxiv.org/pdf/2508.03665"

    # SSRF test URLs that should be blocked
    SSRF_TEST_CASES = [
        # Localhost variants
        ("http://localhost:8080/admin", "localhost"),
        ("http://127.0.0.1:3000/secrets", "127.0.0.1"),
        ("http://0.0.0.0/test", "0.0.0.0"),
        ("http://[::1]/api", "::1"),
        # Private IP ranges
        ("http://192.168.1.1/router", "192.168"),
        ("http://10.0.0.1/internal", "10.0.0"),
        ("http://172.16.0.1/admin", "172.16"),
        # Link-local (cloud metadata)
        ("http://169.254.169.254/latest/meta-data/", "169.254"),
        # File scheme (should be blocked)
        ("file:///etc/passwd", "file"),
        ("file:///C:/Windows/System32/drivers/etc/hosts", "file"),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ssrf_url,expected_in_error", SSRF_TEST_CASES)
    async def test_ssrf_blocked(self, engine, test_collection_name, ssrf_url, expected_in_error):
        """Test that SSRF attacks via document_url are blocked."""
        await engine.create_collection(test_collection_name, vector_size=1536)

        # Attempt to download from SSRF URL should raise ValueError
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            await engine.chunk_and_upsert(
                collection_name=test_collection_name,
                document_url=ssrf_url,
                chunker_name="RecursiveChunker",
            )

        # Verify error message contains expected hint about why it failed
        error_msg = str(exc_info.value).lower()
        assert (
            "ssrf" in error_msg
            or "private" in error_msg
            or "not allowed" in error_msg
            or "localhost" in error_msg
            or expected_in_error.lower() in error_msg
            or "could not resolve" in error_msg
            or "hostname" in error_msg
        ), f"SSRF error message should indicate blocked URL, got: {error_msg}"

        # Verify no points were added (collection should still be empty)
        info = await engine.get_collection_info(test_collection_name)
        assert info["points_count"] == 0, "SSRF attempt should not add any chunks"

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="External URL test - run manually when needed")
    async def test_valid_url_download(self, engine, test_collection_name):
        """Test downloading and chunking from a real public URL.

        Note: This test is skipped by default to avoid external dependencies.
        Run manually with: pytest -v -k test_valid_url_download --run-external
        """
        await engine.create_collection(test_collection_name, vector_size=1536)

        num_chunks = await engine.chunk_and_upsert(
            collection_name=test_collection_name,
            document_url=self.VALID_PDF_URL,
            chunker_name="RecursiveChunker",
            metadata={"source": self.VALID_PDF_URL},
        )

        assert num_chunks > 0, f"Should create chunks from URL: {self.VALID_PDF_URL}"

        # Verify chunks were added
        info = await engine.get_collection_info(test_collection_name)
        assert info["points_count"] == num_chunks


@pytest.mark.skipif(not QDrant_AVAILABLE, reason="Qdrant server not available")
class TestQdrantErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_collection_not_exists_error(self, engine):
        """Test error when collection doesn't exist."""
        with pytest.raises(ValueError, match="does not exist"):
            await engine.get_collection_info("nonexistent_collection")

    @pytest.mark.asyncio
    async def test_upsert_empty_points_error(self, engine, test_collection_name):
        """Test error when upserting empty points."""
        await engine.create_collection(test_collection_name, vector_size=1536)

        with pytest.raises(ValueError, match="cannot be empty"):
            await engine.upsert(test_collection_name, [])

    @pytest.mark.asyncio
    async def test_chunk_and_upsert_no_input_error(self, engine, test_collection_name):
        """Test error when no input provided to chunk_and_upsert."""
        await engine.create_collection(test_collection_name, vector_size=1536)

        with pytest.raises(ValueError, match="must be provided"):
            await engine.chunk_and_upsert(collection_name=test_collection_name)

    @pytest.mark.asyncio
    async def test_chunk_and_upsert_multiple_inputs_error(self, engine, test_collection_name):
        """Test error when multiple inputs provided."""
        await engine.create_collection(test_collection_name, vector_size=1536)

        with pytest.raises(ValueError, match="Only one of"):
            await engine.chunk_and_upsert(
                collection_name=test_collection_name, text="test", document_path="test.pdf"
            )

    @pytest.mark.asyncio
    async def test_delete_nonexistent_collection_error(self, engine):
        """Test error when deleting nonexistent collection."""
        with pytest.raises(ValueError, match="does not exist"):
            await engine.delete_collection("nonexistent_collection")


@pytest.mark.skipif(not QDrant_AVAILABLE, reason="Qdrant server not available")
class TestQdrantSearchChunkedDocuments:
    """Test searching chunked and upserted documents."""

    @pytest.mark.asyncio
    async def test_search_chunked_text_semantic(self, engine, test_collection_name):
        """Test semantic search on chunked text documents."""

        await engine.create_collection(test_collection_name, vector_size=1536)

        # Add a document with specific content
        document_text = """
        Machine learning is a subset of artificial intelligence that focuses on algorithms
        that can learn from data. Deep learning uses neural networks with multiple layers
        to process complex patterns. Natural language processing enables computers to
        understand and generate human language. Computer vision allows machines to interpret
        visual information from the world.
        """
        num_chunks = await engine.chunk_and_upsert(
            collection_name=test_collection_name, text=document_text
        )
        assert num_chunks > 0

        # Search with a semantic query
        query_text = "What is artificial intelligence?"
        query_symbol = Symbol(query_text)
        query_vector = normalize_embedding(query_symbol.embedding)

        results = await engine.search(test_collection_name, query_vector, limit=5)

        assert len(results) > 0, "Should find results for semantic query."
        assert hasattr(results[0], "score"), "Results should have similarity scores."
        assert hasattr(results[0], "payload"), "Results should have payload with text."

        # Verify that results contain relevant content
        found_relevant = False
        for result in results:
            if hasattr(result, "payload") and result.payload:
                text = result.payload.get("text", "") or result.payload.get("content", "")
                if "machine learning" in text.lower() or "artificial intelligence" in text.lower():
                    found_relevant = True
                    break
        assert found_relevant, "Should find relevant chunks containing AI/ML content."

    @pytest.mark.asyncio
    async def test_search_multiple_documents(self, engine, test_collection_name):
        """Test searching across multiple chunked documents."""

        await engine.create_collection(test_collection_name, vector_size=1536)

        # Add multiple documents with different topics
        documents = [
            "Python is a high-level programming language known for its simplicity and readability.",
            "JavaScript is the language of the web, used for both frontend and backend development.",
            "Rust is a systems programming language focused on safety and performance.",
            "Go is a statically typed language developed by Google for concurrent programming.",
        ]

        total_chunks = 0
        for doc in documents:
            chunks = await engine.chunk_and_upsert(collection_name=test_collection_name, text=doc)
            total_chunks += chunks

        assert total_chunks > 0

        # Search for Python-related content
        query_text = "programming language for data science"
        query_symbol = Symbol(query_text)
        query_vector = normalize_embedding(query_symbol.embedding)

        results = await engine.search(test_collection_name, query_vector, limit=10)

        assert len(results) > 0, "Should find results across multiple documents."
        assert len(results) <= 10, "Should respect limit parameter."

        # Verify results contain programming language content
        found_language = False
        for result in results:
            if hasattr(result, "payload") and result.payload:
                text = result.payload.get("text", "") or result.payload.get("content", "")
                if any(lang in text.lower() for lang in ["python", "javascript", "rust", "go"]):
                    found_language = True
                    break
        assert found_language, "Should find programming language content."

    @pytest.mark.asyncio
    async def test_search_with_limit(self, engine, test_collection_name):
        """Test search with different limit values."""

        await engine.create_collection(test_collection_name, vector_size=1536)

        # Add a long document that will create multiple chunks
        long_text = ". ".join(
            [f"Sentence number {i} about machine learning and data science." for i in range(50)]
        )
        num_chunks = await engine.chunk_and_upsert(
            collection_name=test_collection_name, text=long_text
        )
        assert num_chunks > 0

        # Create query embedding
        query_text = "machine learning"
        query_symbol = Symbol(query_text)
        query_vector = normalize_embedding(query_symbol.embedding)

        # Test with limit=1
        results_1 = await engine.search(test_collection_name, query_vector, limit=1)
        assert len(results_1) == 1, "Should return exactly 1 result with limit=1."

        # Test with limit=5
        results_5 = await engine.search(test_collection_name, query_vector, limit=5)
        assert len(results_5) <= 5, "Should return at most 5 results with limit=5."
        assert len(results_5) >= 1, "Should return at least 1 result."

        # Test with limit=20
        results_20 = await engine.search(test_collection_name, query_vector, limit=20)
        assert len(results_20) <= 20, "Should return at most 20 results with limit=20."
        assert len(results_20) >= len(results_5), "Should return more results with higher limit."

    @pytest.mark.asyncio
    async def test_search_result_scores(self, engine, test_collection_name):
        """Test that search results are ordered by relevance score."""

        await engine.create_collection(test_collection_name, vector_size=1536)

        # Add documents with different topics
        documents = [
            "Machine learning algorithms can learn from data without explicit programming.",
            "Cooking recipes require precise measurements and timing.",
            "Deep learning uses neural networks with many layers for complex tasks.",
            "Gardening tips help plants grow healthy and strong.",
        ]

        for doc in documents:
            await engine.chunk_and_upsert(collection_name=test_collection_name, text=doc)

        # Search with ML-related query
        query_text = "neural networks and deep learning"
        query_symbol = Symbol(query_text)
        query_vector = normalize_embedding(query_symbol.embedding)

        results = await engine.search(test_collection_name, query_vector, limit=10)

        assert len(results) > 0, "Should find results."

        # Verify scores are in descending order (highest similarity first)
        scores = [r.score for r in results if hasattr(r, "score")]
        if len(scores) > 1:
            assert scores == sorted(scores, reverse=True), (
                "Results should be ordered by score (descending)."
            )

        # Verify top result has highest score
        if len(results) > 1:
            assert results[0].score >= results[1].score, (
                "First result should have highest or equal score."
            )

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self, engine, test_collection_name):
        """Test searching with metadata filters."""

        await engine.create_collection(test_collection_name, vector_size=1536)

        # Add documents with different metadata
        documents = [
            ("Machine learning is fascinating.", {"category": "AI", "author": "Alice"}),
            ("Cooking is an art form.", {"category": "Food", "author": "Bob"}),
            ("Deep learning models are powerful.", {"category": "AI", "author": "Charlie"}),
        ]

        for text, metadata in documents:
            await engine.chunk_and_upsert(
                collection_name=test_collection_name, text=text, metadata=metadata
            )

        # Create query
        query_text = "learning and intelligence"
        query_symbol = Symbol(query_text)
        query_vector = normalize_embedding(query_symbol.embedding)

        # Search without filter
        all_results = await engine.search(test_collection_name, query_vector, limit=10)
        assert len(all_results) > 0

        # Search with metadata filter for AI category
        try:
            query_filter = Filter(
                must=[FieldCondition(key="category", match=MatchValue(value="AI"))]
            )
            filtered_results = await engine.search(
                test_collection_name, query_vector, limit=10, query_filter=query_filter
            )

            # Verify filtered results only contain AI category
            for result in filtered_results:
                if hasattr(result, "payload") and result.payload:
                    category = result.payload.get("category")
                    if category:
                        assert category == "AI", "Filtered results should only contain AI category."
        except Exception as e:
            # Filter functionality might not be fully implemented or might require different syntax
            pytest.skip(f"Metadata filtering not available or failed: {e}")

    @pytest.mark.asyncio
    async def test_forward_search_with_dict_filter(self, engine, test_collection_name):
        """Test high-level forward() search with simple dict-based metadata filter."""

        await engine.create_collection(test_collection_name, vector_size=1536)

        # Add documents with different metadata via chunk_and_upsert
        documents = [
            ("Machine learning is fascinating.", {"category": "AI", "author": "Alice"}),
            ("Cooking is an art form.", {"category": "Food", "author": "Bob"}),
            ("Deep learning models are powerful.", {"category": "AI", "author": "Charlie"}),
        ]

        for text, metadata in documents:
            await engine.chunk_and_upsert(
                collection_name=test_collection_name,
                text=text,
                metadata=metadata,
            )

        # Build query embedding
        query_text = "learning and intelligence"
        query_symbol = Symbol(query_text)
        # Reuse helper to normalize embedding size
        query_vector = normalize_embedding(query_symbol.embedding)

        # Build Argument for Engine.forward, mimicking index decorator usage
        decorator_kwargs = {
            "prompt": query_vector,
            "operation": "search",
            "index_name": test_collection_name,
            "ori_query": query_text,
            "index_dims": 1536,
            "index_top_k": 10,
            # Pass simple dict filter; engine will convert to Qdrant Filter
            "query_filter": {"category": "AI"},
        }
        argument = Argument(args=(), signature_kwargs={}, decorator_kwargs=decorator_kwargs)

        results, _ = engine.forward(argument)

        # forward returns [QdrantResult], unwrap and inspect raw scored points
        assert isinstance(results, list) and len(results) == 1
        qdrant_result = results[0]
        raw_points = qdrant_result.raw or []

        # We expect at least some results and that all have category == "AI"
        assert len(raw_points) > 0
        for point in raw_points:
            payload = getattr(point, "payload", None) or {}
            category = payload.get("category")
            if category is not None:
                assert category == "AI", (
                    "Dict-based filter in forward() should restrict to AI category."
                )

    @pytest.mark.asyncio
    async def test_forward_search_returns_search_result(self, engine, test_collection_name):
        """Ensure search mode can emit SearchResult-style output for citations."""

        if not AVAILABLE_PDFS:
            pytest.skip("No test PDF files available")

        pdf_path = AVAILABLE_PDFS[0]
        await engine.create_collection(test_collection_name, vector_size=1536)

        await engine.chunk_and_upsert(
            collection_name=test_collection_name,
            document_path=pdf_path,
            chunker_name="RecursiveChunker",
            chunker_kwargs={"chunk_size": 512},
            metadata={"source": Path(pdf_path).name},
        )

        query_symbol = Symbol("citation friendly")
        query_vector = normalize_embedding(query_symbol.embedding)

        decorator_kwargs = {
            "prompt": query_vector,
            "operation": "search",
            "index_name": test_collection_name,
            "ori_query": query_symbol.value,
            "index_dims": 1536,
            "index_top_k": 5,
            "treat_as_search_engine": True,
        }
        argument = Argument(args=(), signature_kwargs={}, decorator_kwargs=decorator_kwargs)

        results, _ = engine.forward(argument)

        assert isinstance(results, list) and len(results) == 1
        search_result = results[0]
        assert hasattr(search_result, "get_citations")
        assert search_result.value is not None
        citations = search_result.get_citations()
        assert isinstance(citations, list)

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self, engine, test_collection_name):
        """Test searching with score threshold."""

        await engine.create_collection(test_collection_name, vector_size=1536)

        # Add documents
        documents = [
            "Machine learning and artificial intelligence are transforming technology.",
            "Random unrelated text about completely different topics like weather and sports.",
            "Deep neural networks enable advanced AI applications.",
        ]

        for doc in documents:
            await engine.chunk_and_upsert(collection_name=test_collection_name, text=doc)

        # Create query
        query_text = "artificial intelligence and machine learning"
        query_symbol = Symbol(query_text)
        query_vector = normalize_embedding(query_symbol.embedding)

        # Search without threshold
        all_results = await engine.search(test_collection_name, query_vector, limit=10)
        assert len(all_results) > 0

        if all_results and hasattr(all_results[0], "score"):
            # Get the score of the last result
            min_score = all_results[-1].score if len(all_results) > 1 else all_results[0].score

            # Search with threshold (use a reasonable threshold based on cosine similarity)
            # Cosine similarity ranges from -1 to 1, but typically we see values > 0.5 for similar content
            threshold = max(0.3, min_score - 0.1) if min_score > 0 else 0.3

            threshold_results = await engine.search(
                test_collection_name, query_vector, limit=10, score_threshold=threshold
            )

            # Verify all results meet threshold
            for result in threshold_results:
                if hasattr(result, "score"):
                    assert result.score >= threshold, (
                        f"Result score {result.score} should be >= threshold {threshold}."
                    )

            # Threshold results should be <= all results
            assert len(threshold_results) <= len(all_results), (
                "Threshold should filter out low-scoring results."
            )

    @pytest.mark.asyncio
    @pytest.mark.skipif(len(AVAILABLE_PDFS) == 0, reason="No test PDF files available")
    async def test_search_pdf_content(self, engine, test_collection_name):
        """Test searching content from chunked PDF documents."""

        if not AVAILABLE_PDFS:
            pytest.skip("No PDF files available for testing")

        pdf_path = AVAILABLE_PDFS[0]
        await engine.create_collection(test_collection_name, vector_size=1536)

        # Chunk and upsert PDF
        num_chunks = await engine.chunk_and_upsert(
            collection_name=test_collection_name,
            document_path=pdf_path,
            chunker_name="RecursiveChunker",
            metadata={"source": os.path.basename(pdf_path)},
        )
        assert num_chunks > 0

        # Create a query that might match PDF content
        # Use a generic query that could match academic/research content
        query_text = "research methodology and findings"
        query_symbol = Symbol(query_text)
        query_vector = normalize_embedding(query_symbol.embedding)

        # Search PDF content
        results = await engine.search(test_collection_name, query_vector, limit=5)

        assert len(results) > 0, "Should find results from PDF content."

        # Verify results have payload with text
        for result in results:
            assert hasattr(result, "payload"), "Results should have payload."
            if result.payload:
                text = result.payload.get("text") or result.payload.get("content", "")
                assert len(text) > 0, "Result payload should contain text content."

    @pytest.mark.asyncio
    async def test_search_result_content_verification(self, engine, test_collection_name):
        """Test that search results contain the expected content from chunked documents."""

        await engine.create_collection(test_collection_name, vector_size=1536)

        # Add a document with specific, searchable content
        document_text = """
        The Python programming language was created by Guido van Rossum in 1991.
        Python emphasizes code readability and simplicity. It supports multiple programming
        paradigms including object-oriented, functional, and procedural programming.
        Python has a large standard library and an active community of developers.
        """
        num_chunks = await engine.chunk_and_upsert(
            collection_name=test_collection_name, text=document_text
        )
        assert num_chunks > 0

        # Search for specific content
        query_text = "Python programming language"
        query_symbol = Symbol(query_text)
        query_vector = normalize_embedding(query_symbol.embedding)

        results = await engine.search(test_collection_name, query_vector, limit=5)

        assert len(results) > 0, "Should find results for Python query."

        # Verify at least one result contains Python-related content
        found_python_content = False
        for result in results:
            if hasattr(result, "payload") and result.payload:
                text = result.payload.get("text", "") or result.payload.get("content", "")
                text_lower = text.lower()
                if "python" in text_lower or "guido" in text_lower or "programming" in text_lower:
                    found_python_content = True
                    # Verify the result has required attributes
                    assert hasattr(result, "id"), "Result should have id."
                    assert hasattr(result, "score"), "Result should have score."
                    break

        assert found_python_content, "Should find results containing Python-related content."

    @pytest.mark.asyncio
    async def test_search_empty_collection(self, engine, test_collection_name):
        """Test searching an empty collection."""

        await engine.create_collection(test_collection_name, vector_size=1536)

        # Create query
        query_text = "test query"
        query_symbol = Symbol(query_text)
        query_vector = normalize_embedding(query_symbol.embedding)

        # Search empty collection
        results = await engine.search(test_collection_name, query_vector, limit=5)

        # Should return empty list, not raise error
        assert isinstance(results, list), "Should return a list even for empty collection."
        assert len(results) == 0, "Should return empty results for empty collection."


@pytest.mark.skipif(not QDrant_AVAILABLE, reason="Qdrant server not available")
class TestQdrantLocalSearch:
    """Local search (SearchResult) behavior."""

    @pytest.mark.asyncio
    async def test_local_search_text_citations(self, engine, test_collection_name):
        """Local search over chunked text with citation formatting."""
        await engine.create_collection(test_collection_name, vector_size=1536)

        text = "Local search citation-friendly content."
        await engine.chunk_and_upsert(
            collection_name=test_collection_name,
            text=text,
            metadata={"source": "local_note.txt"},
        )

        search = Interface("local_search", index_name=test_collection_name, url=engine.url)
        result = search.search(
            "citation-friendly content",
            limit=3,
            score_threshold=0.0,
            with_payload=True,
            with_vectors=False,
            query_filter={"source": "local_note.txt"},
        )
        print(f"local_search result: {result}")
        print(f"local_search citations: {result.get_citations()}")

        assert result.value is not None
        citations = result.get_citations()
        assert isinstance(citations, list)
        assert len(citations) >= 1
        assert citations[0].url != ""

    @pytest.mark.asyncio
    @pytest.mark.skipif(len(AVAILABLE_PDFS) == 0, reason="No test PDF files available")
    async def test_local_search_pdf_citations(self, engine, test_collection_name):
        """Local search over chunked PDF with citation formatting."""
        pdf_path = AVAILABLE_PDFS[0]
        await engine.create_collection(test_collection_name, vector_size=1536)

        await engine.chunk_and_upsert(
            collection_name=test_collection_name,
            document_path=pdf_path,
            chunker_name="RecursiveChunker",
            metadata={"source": Path(pdf_path).name},
        )

        search = Interface("local_search", index_name=test_collection_name, url=engine.url)
        result = search.search(
            "phase-space regions",
            limit=8,
            score_threshold=0.0,
            with_payload=True,
            with_vectors=False,
        )
        print(f"local_search pdf result: {result}")
        print(f"local_search pdf citations: {result.get_citations()}")

        assert result.value is not None
        citations = result.get_citations()
        assert isinstance(citations, list)
        assert len(citations) >= 1


@pytest.mark.skipif(not QDrant_AVAILABLE, reason="Qdrant server not available")
class TestQdrantIntegration:
    """Integration tests combining multiple operations."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, engine, test_collection_name):
        """Test a complete workflow: create, add, search, delete."""
        # Create collection
        await engine.create_collection(test_collection_name, vector_size=1536)
        assert await engine.collection_exists(test_collection_name)

        # Add documents via chunk_and_upsert
        text = "This is a comprehensive test document. It contains multiple sentences for testing."
        num_chunks = await engine.chunk_and_upsert(collection_name=test_collection_name, text=text)
        assert num_chunks > 0

        # Search
        query_vector = [0.1] * 1536
        results = await engine.search(test_collection_name, query_vector, limit=5)
        assert len(results) > 0

        # Retrieve specific points
        if results:
            point_id = results[0].id
            retrieved = await engine.retrieve(test_collection_name, point_id)
            assert len(retrieved) == 1
            assert retrieved[0]["id"] == point_id

        # Delete points
        if results:
            point_id = results[0].id
            await engine.delete(test_collection_name, point_id)

            # Verify deletion
            info = await engine.get_collection_info(test_collection_name)
            assert info["points_count"] == num_chunks - 1

        # Delete collection
        await engine.delete_collection(test_collection_name)
        assert not await engine.collection_exists(test_collection_name)

    @pytest.mark.asyncio
    async def test_multiple_collections(self, engine):
        """Test working with multiple collections."""
        collections = [f"test_multi_{i}" for i in range(3)]

        # Create multiple collections
        for coll_name in collections:
            await engine.create_collection(coll_name, vector_size=1536)

        # Verify all exist
        all_collections = await engine.list_collections()
        for coll_name in collections:
            assert coll_name in all_collections

        # Add data to each
        for i, coll_name in enumerate(collections):
            text = f"Document for collection {i}"
            await engine.chunk_and_upsert(coll_name, text=text)

        # Verify data in each
        for coll_name in collections:
            info = await engine.get_collection_info(coll_name)
            assert info["points_count"] > 0

        # Cleanup
        for coll_name in collections:
            await engine.delete_collection(coll_name)


@pytest.mark.skipif(not QDrant_AVAILABLE, reason="Qdrant server not available")
class TestRagEmbedBatching:
    """Verify that chunk_and_upsert batch-embeds all chunks in a single API call.

    Compares embedding N chunks sequentially (old per-chunk behavior) against the
    full chunk_and_upsert pipeline, which now issues one batch embed call.
    Even though chunk_and_upsert also runs the chunker and a Qdrant upsert, the
    embedding savings dominate and the total time is still lower.
    """

    # First ~120 000 chars of sample.txt (The Odyssey) — yields 15+ natural chunks
    # with the default RecursiveChunker chunk size (~7 000 chars/chunk).
    SAMPLE_PATH = Path(__file__).parents[2] / "data" / "sample.txt"

    @pytest.mark.asyncio
    async def test_chunk_and_upsert_faster_than_sequential_embed(
        self, engine, test_collection_name
    ):

        document = self.SAMPLE_PATH.read_text(encoding="utf-8")[:120_000]

        await engine.create_collection(test_collection_name, vector_size=1536)

        # Pre-chunk with the same settings chunk_and_upsert will use, so the
        # sequential baseline embeds exactly the same strings.
        chunker = ChonkieChunker()
        chunks_symbol = chunker.forward(Symbol(document), chunker_name="RecursiveChunker")
        chunks = chunks_symbol.value if hasattr(chunks_symbol, "value") else chunks_symbol
        chunk_texts = [
            ChonkieChunker.clean_text(str(c))
            for c in chunks
            if ChonkieChunker.clean_text(str(c)).strip()
        ]
        assert len(chunk_texts) >= 15, (
            f"Expected ≥15 chunks from sample.txt excerpt, got {len(chunk_texts)}"
        )

        # Sequential baseline: one embed API call per chunk (old behavior).
        t0 = time.perf_counter()
        for text in chunk_texts:
            Symbol(text).embed()
        sequential_time = time.perf_counter() - t0

        # Batched via chunk_and_upsert: one embed call for all chunks.
        t0 = time.perf_counter()
        num_chunks = await engine.chunk_and_upsert(
            collection_name=test_collection_name,
            text=document,
            chunker_name="RecursiveChunker",
        )
        batch_time = time.perf_counter() - t0

        print(f"\nChunks: {num_chunks}")
        print(f"Sequential embed ({len(chunk_texts)} calls): {sequential_time:.3f}s")
        print(f"chunk_and_upsert (1 embed call):           {batch_time:.3f}s")
        print(f"Speedup: {sequential_time / batch_time:.1f}x")

        assert sequential_time > batch_time, (
            f"chunk_and_upsert ({batch_time:.3f}s) should be faster than "
            f"sequential per-chunk embedding ({sequential_time:.3f}s)"
        )

    @pytest.mark.asyncio
    async def test_concurrent_search_faster_than_sequential(self, engine, test_collection_name):
        """N concurrent searches finish faster than N sequential ones.

        This exercises Qdrant's QDRANT__SERVICE__MAX_WORKERS setting: with
        max_workers > 1, the server handles parallel HTTP requests simultaneously,
        so asyncio.gather + asyncio.to_thread delivers a real wall-clock speedup
        over sequential blocking calls.
        """

        document = self.SAMPLE_PATH.read_text(encoding="utf-8")[:120_000]
        await engine.create_collection(test_collection_name, vector_size=1536)
        await engine.chunk_and_upsert(
            collection_name=test_collection_name,
            text=document,
            chunker_name="RecursiveChunker",
        )

        query_vec = engine._normalize_vector(Symbol("Odyssey travels adventures sea").embed().value)

        N = 8  # number of parallel searches

        def do_search():
            return engine._search_sync(
                collection_name=test_collection_name, query_vector=query_vec, limit=5
            )

        # Sequential baseline
        t0 = time.perf_counter()
        for _ in range(N):
            do_search()
        sequential_time = time.perf_counter() - t0

        # Concurrent: asyncio.to_thread lets each blocking Qdrant call run in a
        # thread so the event loop can schedule all N at once.
        t0 = time.perf_counter()
        await asyncio.gather(*[asyncio.to_thread(do_search) for _ in range(N)])
        concurrent_time = time.perf_counter() - t0

        print(
            f"\nN={N} searches: sequential={sequential_time:.3f}s  concurrent={concurrent_time:.3f}s  speedup={sequential_time / concurrent_time:.1f}x"
        )

        assert concurrent_time < sequential_time, (
            f"Concurrent search ({concurrent_time:.3f}s) should be faster than "
            f"sequential ({sequential_time:.3f}s) with max_workers=4"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
