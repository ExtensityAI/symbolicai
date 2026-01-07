# Indexing Engine

SymbolicAI supports multiple indexing engines for vector search and RAG (Retrieval-Augmented Generation) operations. This document covers both the default naive vector engine and the production-ready Qdrant engine.

## Naive Vector Engine (Default)

By default, text indexing and retrieval is performed with the local naive vector engine using the `Interface` abstraction:

```python
from symai.interfaces import Interface

db = Interface('naive_vectordb', index_name="my_index")
db("Hello world", operation="add")
result = db("Hello", operation="search", top_k=1)
print(result.value)  # most relevant match
```

You can also add or search multiple documents at once, and perform save/load/purge operations:

```python
docs = ["Alpha document", "Beta entry", "Gamma text"]
db = Interface('naive_vectordb', index_name="my_index")
db(docs, operation="add")
db("save", operation="config")
# Load or purge as needed
```

## Qdrant RAG Engine

The Qdrant engine provides a production-ready vector database for scalable RAG applications. It supports both local and cloud deployments, advanced document chunking, and comprehensive collection management.

### Setup

#### Option 1: Local Qdrant Server (via symserver)

Start Qdrant using the `symserver` CLI (Docker by default).

```bash
# Pull the image once (recommended)
docker pull qdrant/qdrant:latest

# Docker (default): set INDEXING_ENGINE so symserver selects Qdrant
INDEXING_ENGINE=qdrant symserver --host 0.0.0.0 --port 6333 --storage-path ./qdrant_storage

# Use native binary
INDEXING_ENGINE=qdrant symserver --env binary --binary-path /path/to/qdrant --port 6333 --storage-path ./qdrant_storage

# Detach Docker if desired
INDEXING_ENGINE=qdrant symserver --docker-detach
```

#### Option 2: Cloud Qdrant

Configure your cloud Qdrant instance:

```python
import os
os.environ["INDEXING_ENGINE_URL"] = "https://your-cluster.qdrant.io"
os.environ["INDEXING_ENGINE_API_KEY"] = "your-api-key"
```

### Basic Usage

The Qdrant engine is used directly via the `QdrantIndexEngine` class:

```python
import asyncio
from symai.backend.engines.index.engine_qdrant import QdrantIndexEngine
from symai import Symbol

# Initialize engine
engine = QdrantIndexEngine(
    url="http://localhost:6333",  # or your cloud URL
    api_key=None,  # optional, for cloud instances
    index_name="my_collection",
    index_dims=1536,  # embedding dimension
    index_top_k=5,  # default top-k for searches
    index_metric="Cosine"  # Cosine, Dot, or Euclidean
)

async def basic_usage():
    # Create a collection
    await engine.create_collection("my_collection", vector_size=1536)

    # Add documents using chunk_and_upsert
    num_chunks = await engine.chunk_and_upsert(
        collection_name="my_collection",
        text="Hello world, this is a test document.",
        metadata={"source": "example"}
    )

    # Search for similar documents
    query = Symbol("Hello")
    query_embedding = query.embedding
    results = await engine.search(
        collection_name="my_collection",
        query_vector=query_embedding,
        limit=5
    )

    # Print results
    for result in results:
        print(f"Score: {result.score}")
        print(f"Text: {result.payload.get('text', '')}")

asyncio.run(basic_usage())
```

### Local Search with citations

If you need citation-formatted results compatible with `parallel.search`, use the `local_search` interface. It embeds the query locally, queries Qdrant, and returns a `SearchResult` (with `value` and `citations`) instead of raw `ScoredPoint` objects:

Local search accepts the same args as passed to Qdrant directly: `collection_name`/`index_name`, `limit`/`top_k`/`index_top_k`, `score_threshold`, `query_filter` (dict or Qdrant `Filter`), and any extra Qdrant search kwargs. Citation fields are derived from Qdrant payloads: the excerpt uses `payload["text"]` (or `content`), the URL is resolved from `payload["source"]`/`url`/`file_path`/`path` and is always returned as an absolute `file://` URI (relative inputs resolve against the current working directory), and the title is the stem of that path (PDF pages append `#p{page}` when provided). Each matching chunk yields its own citation; multiple citations can point to the same file.

If you want a stable source header for each chunk, store a `source_id` or `chunk_id` in the payload (otherwise the Qdrant point id is used).

Example:

```python
from symai.interfaces import Interface
from qdrant_client.http import models

search = Interface("local_search", index_name="my_collection")

qdrant_filter = models.Filter(
    must=[
        models.FieldCondition(key="category", match=models.MatchValue(value="AI"))
    ]
)

result = search.search(
    "neural networks and transformers",
    collection_name="my_collection",   # alias: index_name
    limit=5,                           # aliases: top_k, index_top_k
    score_threshold=0.35,
    query_filter=qdrant_filter,        # or a simple dict like {"category": "AI"}
    with_payload=True,                 # passed through to Qdrant query_points
    with_vectors=False,                # optional; defaults follow engine config
    # any other Qdrant query_points kwargs can be added here
)

print(result.value)          # formatted text with [1], [2] markers
print(result.get_citations())  # list of Citation objects
```

### Collection Management

Create and manage collections programmatically:

```python
import asyncio

async def manage_collections():
    # Create a new collection
    await engine.create_collection(
        collection_name="documents",
        vector_size=1536,
        distance="Cosine"
    )

    # Check if collection exists
    exists = await engine.collection_exists("documents")

    # List all collections
    collections = await engine.list_collections()

    # Get collection info
    info = await engine.get_collection_info("documents")
    print(f"Points: {info['points_count']}")

    # Delete collection
    await engine.delete_collection("documents")

asyncio.run(manage_collections())
```

### Document Chunking and RAG

The Qdrant engine includes built-in document chunking for RAG workflows:

```python
import asyncio

async def add_documents():
    # Chunk and index text directly
    num_chunks = await engine.chunk_and_upsert(
        collection_name="documents",
        text="Your long document text here...",
        metadata={"source": "manual_input", "author": "John Doe"}
    )
    print(f"Indexed {num_chunks} chunks")

    # Chunk and index from a file (PDF, DOCX, etc.)
    num_chunks = await engine.chunk_and_upsert(
        collection_name="documents",
        document_path="/path/to/document.pdf",
        metadata={"source": "document.pdf"}
    )
    # Note: document_path indexing stores the absolute file path in payload["source"]
    # so local_search citations resolve to file:// URIs.

    # Chunk and index from a URL
    num_chunks = await engine.chunk_and_upsert(
        collection_name="documents",
        document_url="https://example.com/document.pdf",
        metadata={"source": "url"}
    )

    # Custom chunker configuration
    num_chunks = await engine.chunk_and_upsert(
        collection_name="documents",
        text="Your text...",
        chunker_name="RecursiveChunker",
        chunker_kwargs={"chunk_size": 512, "chunk_overlap": 50}
    )

asyncio.run(add_documents())
```

### Point Operations

For fine-grained control over individual vectors:

```python
import asyncio
import numpy as np

async def point_operations():
    # Upsert points with embeddings
    points = [
        {
            "id": 1,
            "vector": [0.1] * 1536,  # your embedding vector
            "payload": {"text": "Document 1", "category": "tech"}
        },
        {
            "id": 2,
            "vector": [0.2] * 1536,
            "payload": {"text": "Document 2", "category": "science"}
        }
    ]
    await engine.upsert(collection_name="documents", points=points)

    # Retrieve points by ID
    retrieved = await engine.retrieve(
        collection_name="documents",
        ids=[1, 2],
        with_payload=True,
        with_vectors=False
    )

    # Search for similar vectors
    query_vector = [0.15] * 1536
    results = await engine.search(
        collection_name="documents",
        query_vector=query_vector,
        limit=10,
        score_threshold=0.7  # optional minimum similarity
    )

    # Delete points
    await engine.delete(collection_name="documents", points_selector=[1, 2])

asyncio.run(point_operations())
```

### Configuration Options

The Qdrant engine supports extensive configuration:

```python
engine = QdrantIndexEngine(
    # Connection settings
    url="http://localhost:6333",
    api_key="your-api-key",  # for cloud instances

    # Collection settings
    index_name="default_collection",
    index_dims=1536,  # must match your embedding model
    index_top_k=5,
    index_metric="Cosine",  # Cosine, Dot, or Euclidean

    # Chunking settings
    chunker_name="RecursiveChunker",
    tokenizer_name="gpt2",
    embedding_model_name="minishlab/potion-base-8M",

    # Retry settings
    tries=20,
    delay=0.5,
    max_delay=-1,
    backoff=1,
    jitter=0
)
```

### Environment Variables

Configure Qdrant via environment variables:

```bash
# Qdrant connection
export INDEXING_ENGINE_URL="http://localhost:6333"
export INDEXING_ENGINE_API_KEY="your-api-key"  # optional

# Embedding model (shared with other engines)
export EMBEDDING_ENGINE_API_KEY=""  # empty for local, or API key for cloud
export EMBEDDING_ENGINE_MODEL="all-mpnet-base-v2"  # or your preferred model
```

### Embedding Model & API Key Behavior

- **If `EMBEDDING_ENGINE_API_KEY` is empty (`""`, the default),** SymbolicAI will use a local, lightweight embedding engine based on SentenceTransformers. You can specify any supported model name via `EMBEDDING_ENGINE_MODEL` (e.g. `"all-mpnet-base-v2"`).
- **If you DO provide an `EMBEDDING_ENGINE_API_KEY`**, then the respective remote embedding engine will be used (e.g. OpenAI). The model is selected according to the `EMBEDDING_ENGINE_MODEL` key where applicable.

This allows you to easily experiment locally for free, and switch to more powerful cloud backends when ready.

### Installation

Install Qdrant support using the package extra (recommended):

```bash
pip install symai[qdrant]
```

This installs all required dependencies:
- `qdrant-client` - Qdrant Python client
- `chonkie[all]` - Document chunking library
- `tokenizers` - Tokenization support

Alternatively, install dependencies individually:

```bash
pip install qdrant-client chonkie tokenizers
```

### See Also

- See `tests/engines/index/test_qdrant_engine.py` for comprehensive usage examples
- Qdrant documentation: https://qdrant.tech/documentation/
