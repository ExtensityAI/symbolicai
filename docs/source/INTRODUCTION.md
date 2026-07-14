# Introduction

SymbolicAI provides explicit, typed runtimes for three language-model providers and one embedding provider:

- OpenAI language models
- Cerebras language models
- DeepSeek language models
- OpenAI text embeddings

The public contract is provider-neutral. Applications build a `RuntimeConfig`, create a runtime with `create_runtime`, and submit normalized request models. Runtime responses use the same normalized models regardless of the selected provider.

## Explicit ownership

Configuration belongs to the calling application. An application may read a key from its environment, secret manager, or another source, but it passes the resulting value directly to `ProviderEngineConfig`. SymbolicAI has no global configuration source and does not choose credentials or models implicitly.

```python
import os

from pydantic import SecretStr

from symai import Provider, ProviderEngineConfig, RuntimeConfig

config = RuntimeConfig(
    language_model=ProviderEngineConfig(
        provider=Provider.CEREBRAS,
        model="gpt-oss-120b",
        api_key=SecretStr(os.environ["CEREBRAS_API_KEY"]),
    ),
)
```

Provider and model are separate fields. Model IDs are catalog IDs such as `gpt-oss-120b`, not provider-qualified strings.

## Normalized data

Language requests contain typed message and content objects. Embedding requests contain a non-empty tuple of strings. Provider adapters translate these normalized objects to HTTP payloads and translate successful responses back to `LanguageModelResponse` or `EmbeddingResponse`.

This boundary keeps provider-specific payloads out of application code. Typed runtime errors distinguish unsupported capability/model/feature combinations, authentication failures, rate limits, transport failures, and invalid responses.

## Deterministic lifetime

A runtime owns the HTTP transports created for its configured capabilities. Use it as a context manager:

```python
from symai import RuntimeConfig, create_runtime


def use_runtime(config: RuntimeConfig) -> None:
    with create_runtime(config):
        # Submit normalized requests inside this context.
        pass
```

Entering activates the runtime for the current context. Exiting waits for in-flight calls, closes owned transports, and deactivates the context. Each runtime has one lifecycle and cannot be entered again after it closes.

Continue with the [Quickstart](QUICKSTART.md), then consult [Runtime and Providers](RUNTIME.md) or [OpenAI Embeddings](EMBEDDINGS.md).
