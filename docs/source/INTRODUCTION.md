# Introduction

SymbolicAI provides explicit, typed runtimes for three language-model providers and one embedding provider:

- OpenAI language models
- Cerebras language models
- DeepSeek language models
- OpenAI text embeddings

The public contract is provider-neutral. Applications build a `RuntimeConfig`, load a runtime with `load_runtime`, and submit normalized request models. Runtime responses use the same normalized models regardless of the selected provider.

## Explicit ownership

Configuration belongs to the calling application. An application may read a key from its environment, secret manager, or another source, but it passes the resulting value directly into the engine's settings. SymbolicAI has no global configuration source and does not choose credentials or models implicitly.

```python
import os

from pydantic import SecretStr

from symai.runtime.config import EngineConfig, RuntimeConfig

config = RuntimeConfig(
    language_models={
        "chat": EngineConfig(
            implementation="cerebras:chat-completions",
            settings={
                "api_key": SecretStr(os.environ["CEREBRAS_API_KEY"]),
                "model": "gpt-oss-120b",
            },
        )
    },
)
```

The implementation ID selects the provider adapter; the model is a catalog ID such as `gpt-oss-120b`, not a provider-qualified string. The mapping key — `"chat"` above — is a name you choose, and it is how you select this engine later.

## Named engines

An engine is identified by its capability and its name. The same provider and model may be configured more than once under different names, with different credentials and transports, and each configured engine owns its own HTTP client. Nothing is pooled or shared between them.

## Normalized data

Language requests contain typed message and content objects. Embedding requests contain a non-empty tuple of strings. Provider adapters translate these normalized objects to HTTP payloads and translate successful responses back to `LanguageModelResponse` or `EmbeddingResponse`.

This boundary keeps provider-specific payloads out of application code. Typed runtime errors distinguish unsupported capability/model/feature combinations, authentication failures, permission failures, invalid requests, rate limits, provider failures, transport failures, and invalid responses. Their metadata is safe and bounded: prompts, credentials, and raw provider bodies never appear in exception messages or logs.

## Deterministic lifetime

A runtime owns the HTTP transports created for its configured engines. Use it as a context manager:

```python
from symai.loading import load_runtime
from symai.runtime.config import RuntimeConfig


def use_runtime(config: RuntimeConfig) -> None:
    with load_runtime(config) as runtime:
        # Submit normalized requests inside this context.
        del runtime
```

Entering records the owner thread. A runtime may only execute on the thread that entered it; a future async API is designed separately rather than weakening this contract. Exiting closes every owned engine exactly once, in reverse construction order, and reports all cleanup failures together. Each runtime has one lifecycle and cannot be entered again after it closes.

Continue with the [Quickstart](QUICKSTART.md), then consult [Runtime and Providers](RUNTIME.md) or [OpenAI Embeddings](EMBEDDINGS.md).
