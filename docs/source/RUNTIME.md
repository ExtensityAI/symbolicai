# Runtime and Providers

`Runtime` is the explicit execution boundary for SymbolicAI. The calling application supplies a frozen `RuntimeConfig`; `create_runtime` resolves all configured capabilities against local provider/model catalogs and returns a single-owner runtime.

## Language model catalogs

Model IDs are unqualified catalog values. Select the adapter with `ProviderEngineConfig.provider` and pass one of these IDs as `model`.

### OpenAI

- `gpt-5.5`
- `gpt-5.5-pro`
- `gpt-5.4`
- `gpt-5.4-pro`
- `gpt-5.4-mini`
- `gpt-5.4-nano`
- `o3-pro`
- `o3`
- `gpt-4.1`
- `gpt-4.1-mini`

### Cerebras

- `gpt-oss-120b`
- `gemma-4-31b`
- `zai-glm-4.7`

### DeepSeek

- `deepseek-v4-flash`
- `deepseek-v4-pro`

## Caller-owned configuration

Applications choose how to acquire credentials. SymbolicAI receives the resulting strings only through explicit configuration:

```python
import os

from pydantic import SecretStr

from symai import Provider, ProviderEngineConfig, RuntimeConfig

openai_config = RuntimeConfig(
    language_model=ProviderEngineConfig(
        provider=Provider.OPENAI,
        model="gpt-5.4",
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    ),
)
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

`RuntimeConfig` requires at least one configured capability. `create_runtime` rejects unsupported provider/capability pairs with `UnsupportedCapabilityError` and unknown catalog IDs with `UnsupportedModelError` before constructing transports.

## Normalized language requests

Every language adapter accepts `LanguageModelRequest` rather than a provider payload:

```python
from symai import (
    LanguageModelRequest,
    RuntimeConfig,
    TextContent,
    UserMessage,
    create_runtime,
)

request = LanguageModelRequest(
    messages=(UserMessage(content=(TextContent(text="Summarize explicit ownership."),)),),
)


def generate(config: RuntimeConfig) -> str:
    with create_runtime(config) as runtime:
        response = runtime.execute(request)
        return response.outputs[0].text
```

A request has a non-empty message tuple and optional normalized response format, reasoning, sampling, user, and metadata fields. Each adapter validates requested fields against the selected model's catalog. If the normalized request asks for a feature that model cannot satisfy, execution raises `UnsupportedFeatureError` before sending an HTTP request.

Successful execution returns `LanguageModelResponse`. Its `outputs` tuple is non-empty. Each output contains a normalized assistant message, finish reason, optional refusal, and a `text` property; access the first text as `response.outputs[0].text`. Response metadata identifies the provider and model and may include request identifiers, token usage, and rate-limit data.

## Runtime lifetime

A runtime is created inactive and has one context-managed lifecycle:

1. `with create_runtime(config) as runtime` activates it for the current context.
2. `runtime.execute(request)` is accepted only while that context is active.
3. Context exit stops accepting work, waits for in-flight calls, closes each owned transport, and deactivates the runtime.
4. A closed runtime cannot be re-entered or execute another request.

When both capabilities use OpenAI, each configured capability has its own owned transport and both are closed on exit. Calling `close()` more than once is safe, but context management is the normal ownership pattern.

Runtime failures use the exported error hierarchy: authentication, rate limit, transport, invalid response, unsupported capability/model/feature, missing active runtime, and closed runtime errors remain distinct.
