# Installation

SymbolicAI requires Python 3.11 or newer.

## Install the package

```bash
pip install symbolicai
```

The published package has one mandatory dependency set and no capability extras. OpenAI, Cerebras, and DeepSeek access is implemented by SymbolicAI's retained HTTP adapters; no provider SDK installation is required.

## Install a repository checkout

Use [uv](https://docs.astral.sh/uv/) 0.9.17 or newer to reproduce the development environment:

```bash
git clone https://github.com/ExtensityAI/symbolicai.git
cd symbolicai
uv sync --frozen
```

The `dev` dependency group contains pytest, pytest-xdist, and Ruff. Pyright may be run from an available project-local or system installation.

## Configure an application

The calling application owns its credential source. For example, it can read an environment variable and pass the value into a typed configuration:

```python
import os

from pydantic import SecretStr

from symai import Provider, ProviderEngineConfig, RuntimeConfig, TransportConfig

openai = ProviderEngineConfig(
    provider=Provider.OPENAI,
    model="gpt-5.4",
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    transport=TransportConfig(
        request_timeout=60.0,
        connect_timeout=10.0,
        connect_retries=1,
    ),
)
config = RuntimeConfig(language_model=openai)
```

`ProviderEngineConfig` requires an explicit provider, an unqualified catalog model ID, and a non-empty API key. `TransportConfig` controls request timeout, connect timeout, and connection retries. It defaults to a 600-second request timeout, a 10-second connect timeout, and zero retries.

SymbolicAI does not load credentials, models, or transport settings from environment variables on its own. It does not create or inspect configuration files. Reading `OPENAI_API_KEY` above is application code, not package behavior.

## Verify the checkout

```bash
uv lock --check
uv build
```

See the [Quickstart](QUICKSTART.md) for normalized language execution and [OpenAI Embeddings](EMBEDDINGS.md) for embedding configuration.
