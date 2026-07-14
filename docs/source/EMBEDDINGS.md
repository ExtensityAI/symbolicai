# OpenAI Embeddings

The retained embedding capability supports OpenAI text embeddings through the same explicit runtime boundary used for language models.

## Model catalog

| Model | Default dimensions | `dimensions` request field |
| --- | ---: | --- |
| `text-embedding-ada-002` | 1536 | Not supported |
| `text-embedding-3-small` | 1536 | Supported, up to 1536 |
| `text-embedding-3-large` | 3072 | Supported, up to 3072 |

Use these unqualified model IDs with `Provider.OPENAI`. Cerebras and DeepSeek do not provide an embedding capability in this runtime.

## Configure and execute

The application obtains the OpenAI key and passes it into `ProviderEngineConfig`. The embedding request contains a non-empty tuple of text inputs:

```python
import os

from pydantic import SecretStr

from symai import (
    EmbeddingRequest,
    Provider,
    ProviderEngineConfig,
    RuntimeConfig,
    create_runtime,
)

config = RuntimeConfig(
    embedding=ProviderEngineConfig(
        provider=Provider.OPENAI,
        model="text-embedding-3-small",
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    ),
)
request = EmbeddingRequest(
    inputs=("first", "second"),
    dimensions=512,
)


def embed() -> tuple[float, ...]:
    with create_runtime(config) as runtime:
        response = runtime.execute(request)
        return response.vectors[0].values
```

Call `embed()` when network execution is desired. `EmbeddingResponse.vectors` contains one normalized vector per input. Each vector has its source `index` and a non-empty tuple of floats in `values`; access the first as `response.vectors[0].values`.

## Dimension validation

Leave `dimensions` unset to use the model default. A positive custom value is accepted only by `text-embedding-3-small` and `text-embedding-3-large`, and it cannot exceed that model's default size. The adapter raises `UnsupportedFeatureError` before transport when `dimensions` is supplied for `text-embedding-ada-002` or exceeds the selected model's maximum.

The runtime also rejects non-OpenAI embedding configuration with `UnsupportedCapabilityError` and unknown OpenAI embedding IDs with `UnsupportedModelError` before allocating an HTTP transport.
