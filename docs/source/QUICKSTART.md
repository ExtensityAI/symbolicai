# Quickstart

A SymbolicAI application chooses a provider and unqualified model ID explicitly, builds normalized requests, and owns the runtime lifetime.

## Configure OpenAI

The application may obtain credentials from any source. This example reads its own environment and passes the value into `ProviderEngineConfig`:

```python
import os

from pydantic import SecretStr

from symai import (
    LanguageModelRequest,
    Provider,
    ProviderEngineConfig,
    RuntimeConfig,
    TextContent,
    UserMessage,
    create_runtime,
)

config = RuntimeConfig(
    language_model=ProviderEngineConfig(
        provider=Provider.OPENAI,
        model="gpt-5.4",
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    ),
)
request = LanguageModelRequest(
    messages=(UserMessage(content=(TextContent(text="Say hello in German."),)),),
)


def generate() -> str:
    with create_runtime(config) as runtime:
        response = runtime.execute(request)
        return response.outputs[0].text
```

Call `generate()` when the application is ready to make the HTTP request. `runtime.execute` accepts the normalized `LanguageModelRequest` and returns a normalized `LanguageModelResponse`; the text accessor is `response.outputs[0].text`.

## Choose another language provider

Only the provider configuration changes. Cerebras and DeepSeek use the same request and response models:

```python
import os

from pydantic import SecretStr

from symai import Provider, ProviderEngineConfig, RuntimeConfig

cerebras_config = RuntimeConfig(
    language_model=ProviderEngineConfig(
        provider=Provider.CEREBRAS,
        model="gpt-oss-120b",
        api_key=SecretStr(os.environ["CEREBRAS_API_KEY"]),
    ),
)
deepseek_config = RuntimeConfig(
    language_model=ProviderEngineConfig(
        provider=Provider.DEEPSEEK,
        model="deepseek-v4-flash",
        api_key=SecretStr(os.environ["DEEPSEEK_API_KEY"]),
    ),
)
```

Model IDs are unqualified because the `provider` field already selects the adapter.

## Add OpenAI embeddings

A runtime may own both a language model and an embedding engine. The embedding request is normalized in the same way:

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

embedding_config = RuntimeConfig(
    embedding=ProviderEngineConfig(
        provider=Provider.OPENAI,
        model="text-embedding-3-small",
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    ),
)
embedding_request = EmbeddingRequest(
    inputs=("first", "second"),
    dimensions=512,
)


def embed() -> tuple[float, ...]:
    with create_runtime(embedding_config) as runtime:
        response = runtime.execute(embedding_request)
        return response.vectors[0].values
```

Call `embed()` to execute the request. The runtime closes its owned HTTP transport when the context exits, including when execution raises an exception.

See [Runtime and Providers](RUNTIME.md) for catalogs and request validation, and [OpenAI Embeddings](EMBEDDINGS.md) for dimension rules.
