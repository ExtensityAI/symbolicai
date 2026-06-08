# Embedding Engine

The embedding engine generates high-dimensional vector representations of text (and, for supported models, images and other modalities). These embeddings enable semantic search, similarity comparisons, clustering, and retrieval-augmented generation (RAG).

SymbolicAI supports multiple embedding backends. The engine is selected automatically based on your `EMBEDDING_ENGINE_MODEL` configuration:

| Model | Backend | Dimensions | Modalities |
|-------|---------|------------|------------|
| `text-embedding-3-small` | OpenAI | 1536 | Text |
| `text-embedding-3-large` | OpenAI | 3072 | Text |
| `text-embedding-ada-002` | OpenAI | 1536 | Text |
| `gemini-embedding-001` | Google Gemini | 3072 | Text |
| `gemini-embedding-2` | Google Gemini | 3072 | Text, Image, Video, Audio, Document |
| `llamacpp` | llama.cpp (local) | Varies | Text |

## Configuration

Set the embedding engine in your `symai.config.json`:

```json
{
    "EMBEDDING_ENGINE_API_KEY": "<YOUR_API_KEY>",
    "EMBEDDING_ENGINE_MODEL": "gemini-embedding-001"
}
```

For OpenAI models, use your OpenAI API key. For Gemini models, use your Google AI API key (obtain one from [Google AI Studio](https://aistudio.google.com/)).

## Usage

### Text Embeddings

Generate embeddings for a single string or a batch of strings:

```python
from symai import Symbol

# Single text
result = Symbol("hello world").embed()
# result.value => [[0.012, -0.034, ...]]  (1 x dim)

# Batch
result = Symbol(["hello", "world"]).embed()
# result.value => [[0.012, ...], [-0.034, ...]]  (2 x dim)
```

### Dimension Reduction

Both OpenAI and Gemini embedding models support Matryoshka Representation Learning (MRL), which allows truncating embeddings to smaller dimensions with minimal quality loss. Use the `new_dim` parameter:

```python
result = Symbol(["hello"]).embed(new_dim=768)
# result.value => [[...]]  (1 x 768)
```

Google recommends using 768, 1536, or 3072 dimensions for Gemini models. Smaller dimensions reduce storage costs in vector databases while preserving most of the semantic quality.

### Multimodal Embeddings (Gemini Embedding 2)

The `gemini-embedding-2` model supports embedding images, video, audio, and documents alongside text. SymbolicAI now supports multimodal inputs directly through `Symbol.embed()`:

```python
from pathlib import Path
from google.genai.types import Content, Part
from symai import Symbol

# Image embedding from raw bytes
image_bytes = Path("photo.png").read_bytes()
result = Symbol(image_bytes).embed()

# Image embedding from Google Part object
image_part = Part.from_bytes(data=image_bytes, mime_type="image/png")
result = Symbol(image_part).embed()

# Combined text and image embedding
content = Content(parts=[
    Part.from_text(text="Describe this image"),
    Part.from_bytes(data=image_bytes, mime_type="image/png"),
])
result = Symbol(content).embed()

# Batch with mixed text and image inputs
result = Symbol(["hello world", image_part]).embed()
```

**Note:** When using `gemini-embedding-2` with multiple inputs in a single call, the model aggregates them into one embedding by default. For individual embeddings per input, wrap each in a `Content` object or use the Batch API.

**Supported input types:**
- `str` - Text input
- `bytes` - Raw file bytes (MIME type auto-detected internally via [filetype](https://github.com/h2non/filetype.py), supports 100+ formats including images, audio, video, and documents)
- `Part` - Google genai Part object (for explicit MIME type control)
- `Content` - Google genai Content object (for mixed text+image)

**Engine compatibility:** Each engine handles input types based on its capabilities. OpenAI engines only support text and will raise a clear error for non-text inputs.

## Supported Task Types

The Gemini engine supports task-specific embeddings via the `task_type` parameter:

| Task Type | Use Case |
|-----------|----------|
| `SEMANTIC_SIMILARITY` (default) | General-purpose similarity |
| `RETRIEVAL_QUERY` | Query side of document retrieval |
| `RETRIEVAL_DOCUMENT` | Document side of retrieval (pair with `title`) |
| `CLASSIFICATION` | Text classification |
| `CLUSTERING` | Clustering tasks |
| `QUESTION_ANSWERING` | QA retrieval |
| `FACT_VERIFICATION` | Fact-checking retrieval |

```python
result = Symbol(["What is machine learning?"]).embed(task_type="QUESTION_ANSWERING")
```

## Local Embeddings

For fully local, offline embedding, see the [Local Engine](local_engine.md#local-embedding-engine) documentation.
