# Quickstart

A SymbolicAI application chooses an implementation and unqualified model ID explicitly, builds normalized requests, and owns the runtime lifetime.

## Configure OpenAI

The application may obtain credentials from any source. This example reads its own environment and passes the value into the engine's settings:

```python
import os

from pydantic import SecretStr

from symai.loading import load_runtime
from symai.runtime.config import EngineConfig, RuntimeConfig
from symai.runtime.models import LanguageModelRequest, TextContent, UserMessage

config = RuntimeConfig(
    language_models={
        "chat": EngineConfig(
            implementation="openai:responses",
            settings={
                "api_key": SecretStr(os.environ["OPENAI_API_KEY"]),
                "model": "gpt-5.4",
            },
        )
    },
)
request = LanguageModelRequest(
    messages=(UserMessage(content=(TextContent(text="Say hello in German."),)),),
)


def generate() -> str:
    with load_runtime(config) as runtime:
        response = runtime.language_model("chat").execute(request)
        return response.text
```

Call `generate()` when the application is ready to make the HTTP request. `runtime.language_model("chat")` returns a bound handle; its `execute` accepts the normalized `LanguageModelRequest` and returns a normalized `LanguageModelResponse`. `response.text` is the first output's text, and `response.outputs` exposes every normalized output.

The name may be omitted when exactly one engine provides the capability. With more than one, omitting it raises an ambiguity error listing the configured names rather than guessing.

## Choose another language provider

Only the engine configuration changes. Cerebras and DeepSeek use the same request and response models:

```python
import os

from pydantic import SecretStr

from symai.runtime.config import EngineConfig, RuntimeConfig

cerebras_config = RuntimeConfig(
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
deepseek_config = RuntimeConfig(
    language_models={
        "chat": EngineConfig(
            implementation="deepseek:chat-completions",
            settings={
                "api_key": SecretStr(os.environ["DEEPSEEK_API_KEY"]),
                "model": "deepseek-v4-flash",
            },
        )
    },
)
```

Model IDs are unqualified because the implementation ID already selects the adapter.

## Configure the same model twice

Names, not provider/model pairs, identify engines. Two tenants can share a model while keeping separate credentials and separate HTTP clients:

```python
import os

from pydantic import SecretStr

from symai.runtime.config import EngineConfig, RuntimeConfig

tenants = RuntimeConfig(
    language_models={
        "tenant-a": EngineConfig(
            implementation="openai:responses",
            settings={"api_key": SecretStr(os.environ["TENANT_A_KEY"]), "model": "gpt-5.4"},
        ),
        "tenant-b": EngineConfig(
            implementation="openai:responses",
            settings={"api_key": SecretStr(os.environ["TENANT_B_KEY"]), "model": "gpt-5.4"},
        ),
    },
)
```

## Add OpenAI embeddings

A runtime may own both language models and embedding engines. The embedding request is normalized in the same way:

```python
import os

from pydantic import SecretStr

from symai.loading import load_runtime
from symai.runtime.config import EngineConfig, RuntimeConfig
from symai.runtime.models import EmbeddingRequest

embedding_config = RuntimeConfig(
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
embedding_request = EmbeddingRequest(inputs=("first", "second"), dimensions=512)


def embed() -> tuple[float, ...]:
    with load_runtime(embedding_config) as runtime:
        response = runtime.embedding("vectors").execute(embedding_request)
        return response.vectors[0].values
```

Call `embed()` to execute the request. The runtime closes every owned HTTP transport when the context exits, including when execution raises an exception.

See [Runtime and Providers](RUNTIME.md) for catalogs and request validation, and [OpenAI Embeddings](EMBEDDINGS.md) for dimension rules.
