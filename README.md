# SymbolicAI

Explicit typed runtimes for OpenAI, Cerebras, and DeepSeek language models and OpenAI embeddings.

SymbolicAI gives applications one normalized request/response contract across the retained providers. The application owns configuration and credentials, creates a `Runtime`, enters it as a context manager, and receives normalized responses rather than provider SDK objects.

## Supported capabilities

| Capability | Provider | Example model |
| --- | --- | --- |
| Language model | OpenAI | `gpt-5.4` |
| Language model | Cerebras | `gpt-oss-120b` |
| Language model | DeepSeek | `deepseek-v4-flash` |
| Text embeddings | OpenAI | `text-embedding-3-small` |

Model IDs are unqualified. The `provider` field selects the adapter; do not prefix a model ID with a provider name.

## Installation

SymbolicAI requires Python 3.11 or newer.

```bash
pip install symbolicai
```

For a repository checkout:

```bash
git clone https://github.com/ExtensityAI/symbolicai.git
cd symbolicai
uv sync --frozen
```

The package has no capability extras. Applications provide API keys directly in `ProviderEngineConfig`; SymbolicAI does not read application environment variables or configuration files.

## Quickstart

This example defines both retained OpenAI capabilities. Network execution occurs only when the application calls `run()`.

```python
import os

from pydantic import SecretStr

from symai import (
    EmbeddingRequest,
    LanguageModelRequest,
    Provider,
    ProviderEngineConfig,
    RuntimeConfig,
    TextContent,
    UserMessage,
    create_runtime,
)

api_key = SecretStr(os.environ["OPENAI_API_KEY"])
config = RuntimeConfig(
    language_model=ProviderEngineConfig(
        provider=Provider.OPENAI,
        model="gpt-5.4",
        api_key=api_key,
    ),
    embedding=ProviderEngineConfig(
        provider=Provider.OPENAI,
        model="text-embedding-3-small",
        api_key=api_key,
    ),
)
language_request = LanguageModelRequest(
    messages=(UserMessage(content=(TextContent(text="Say hello in German."),)),),
)
embedding_request = EmbeddingRequest(
    inputs=("first", "second"),
    dimensions=512,
)


def run() -> tuple[str, tuple[float, ...]]:
    with create_runtime(config) as runtime:
        language_response = runtime.execute(language_request)
        embedding_response = runtime.execute(embedding_request)
        return (
            language_response.outputs[0].text,
            embedding_response.vectors[0].values,
        )
```

`RuntimeConfig` is immutable and requires at least one capability. `create_runtime` validates provider/capability and model combinations before constructing the runtime. The context manager activates the runtime, waits for in-flight work during shutdown, and closes every owned HTTP transport exactly once. A runtime has a single lifecycle and cannot be re-entered.

Requests and responses are provider-neutral:

- `LanguageModelRequest` contains typed messages, response format, reasoning, sampling, user, and metadata fields.
- `LanguageModelResponse.outputs` contains normalized outputs; use `response.outputs[0].text` for text.
- `EmbeddingRequest` contains one or more text inputs plus an optional dimension count.
- `EmbeddingResponse.vectors` contains normalized vectors; use `response.vectors[0].values` for values.

Adapters validate normalized request features against the selected model catalog and raise a typed runtime error before transport when a model cannot satisfy the request.

## Documentation

- [Introduction](docs/source/INTRODUCTION.md)
- [Installation](docs/source/INSTALLATION.md)
- [Quickstart](docs/source/QUICKSTART.md)
- [Runtime and providers](docs/source/RUNTIME.md)
- [OpenAI embeddings](docs/source/EMBEDDINGS.md)

## Development

```bash
uv sync --frozen
uv run pytest
uv run ruff format --check symai tests
uv run ruff check symai tests
```

## License

SymbolicAI is distributed under the BSD 3-Clause license. See [LICENSE](LICENSE).
