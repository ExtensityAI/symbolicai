# OpenAI Embeddings

The embedding capability supports OpenAI text embeddings through the same explicit runtime boundary used for language models.

## Model catalog

| Model | Default dimensions | `dimensions` request field |
| --- | ---: | --- |
| `text-embedding-ada-002` | 1536 | Not supported |
| `text-embedding-3-small` | 1536 | Supported, up to 1536 |
| `text-embedding-3-large` | 3072 | Supported, up to 3072 |

Use these unqualified model IDs with the `openai:embeddings` implementation. Cerebras and DeepSeek do not provide an embedding capability in this runtime.

## Configure and execute

The application obtains the OpenAI key and passes it into the engine's settings. The embedding request contains a non-empty tuple of text inputs:

```python
import os

from pydantic import SecretStr

from symai.loading import load_runtime
from symai.runtime.config import EngineConfig, RuntimeConfig
from symai.runtime.models import EmbeddingRequest

config = RuntimeConfig(
    embeddings={
        "vectors": EngineConfig(
            implementation="openai:embeddings",
            settings={
                "api_key": SecretStr(os.environ["OPENAI_API_KEY"]),
                "model": "text-embedding-3-small",
            },
        )
    },
)
request = EmbeddingRequest(inputs=("first", "second"), dimensions=512)


def embed() -> tuple[float, ...]:
    with load_runtime(config) as runtime:
        response = runtime.embedding("vectors").execute(request)
        return response.vectors[0].values
```

Call `embed()` when network execution is desired. `EmbeddingResponse.vectors` contains one normalized vector per input. Each vector has its source `index` and a non-empty tuple of floats in `values`; access the first as `response.vectors[0].values`. Vectors are returned in input order regardless of the order the provider sent them.

## Dimension validation

Leave `dimensions` unset to use the model default. A positive custom value is accepted only by `text-embedding-3-small` and `text-embedding-3-large`, and it cannot exceed that model's default size. The adapter raises `UnsupportedFeatureError` before transport when `dimensions` is supplied for `text-embedding-ada-002` or exceeds the selected model's maximum.

The runtime rejects unknown OpenAI embedding IDs with `UnsupportedModelError` before allocating an HTTP transport, and raises `UnsupportedCapabilityError` when no engine provides the embedding capability.

## Vector operations

`symai.ops.embed` wraps embedding execution and the deterministic vector math that usually follows it:

```python
from symai.ops import embed
from symai.symbol import Symbol

left = Symbol([1.0, 0.0])
right = Symbol([0.0, 1.0])
score = embed.similarity(left, right, metric="cosine")
```

`embed.embed` takes an embedding handle and performs I/O. The rest — `similarity` (`cosine`, `dot`), `distance` (`euclidean`, `manhattan`, `minkowski`), `mmd` (RBF), and `kernel` (`linear`, `rbf`, `polynomial`) — are local, take no handle, and perform no I/O.
