# Indexing Engine

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

See code/tests for further advanced usage.

## Embedding Model & API Key Behavior

- **If `EMBEDDING_ENGINE_API_KEY` is empty (`""`, the default),** SymbolicAI will use a local, lightweight embedding engine based on SentenceTransformers. You can specify any supported model name via `EMBEDDING_ENGINE_MODEL` (e.g. `"all-mpnet-base-v2"`).
- **If you DO provide an `EMBEDDING_ENGINE_API_KEY`**, then the respective remote embedding engine will be used (e.g. OpenAI). The model is selected according to the `EMBEDDING_ENGINE_MODEL` key where applicable.

This allows you to easily experiment locally for free, and switch to more powerful cloud backends when ready.
