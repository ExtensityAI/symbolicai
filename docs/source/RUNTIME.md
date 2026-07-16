# Runtime and Providers

`Runtime` is the explicit execution boundary for SymbolicAI. The calling application supplies a frozen `RuntimeConfig`; `load_runtime` resolves every configured engine against local provider/model catalogs and returns a single-owner runtime.

## Language model catalogs

Model IDs are unqualified catalog values. Select the adapter with `EngineConfig.implementation` and pass one of these IDs as the `model` setting.

### OpenAI — `openai:responses`

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

### Cerebras — `cerebras:chat-completions`

- `gpt-oss-120b`
- `gemma-4-31b`
- `zai-glm-4.7`

### DeepSeek — `deepseek:chat-completions`

- `deepseek-v4-flash`
- `deepseek-v4-pro`

Capabilities differ per model, not just per provider. Reasoning effort, vision, and reasoning controls are validated against the specific model's catalog entry, so a model that does not accept a control rejects it locally rather than failing at the API.

## Caller-owned configuration

Applications choose how to acquire credentials. SymbolicAI receives the resulting values only through explicit configuration:

```python
import os

from pydantic import SecretStr

from symai.runtime.config import EngineConfig, RuntimeConfig

openai_config = RuntimeConfig(
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
```

`RuntimeConfig` requires at least one configured engine. `load_runtime` rejects an unknown implementation ID and unknown catalog IDs with `UnsupportedModelError`, and validates every engine's settings before allocating any transport.

## Named engines and selection

The mapping key is the engine's name. Names are unique within each capability, so the same name may identify one language model and one embedding engine; every selection path already fixes the capability.

```python
from symai.loading import load_runtime
from symai.runtime.config import RuntimeConfig


def select(config: RuntimeConfig) -> None:
    with load_runtime(config) as runtime:
        chat = runtime.language_model("chat")
        del chat
```

Resolution rules:

1. A supplied name must exist within the requested capability, or `UnknownEngineError` is raised.
2. An omitted name resolves only when exactly one engine provides that capability.
3. Several matching engines raise `AmbiguousEngineError` listing the configured names.
4. No matching engine raises `UnsupportedCapabilityError`.

`runtime.execute(request, engine="chat")` remains as a low-level escape hatch for dynamic routing; the request type selects the capability.

## Normalized language requests

Every language adapter accepts `LanguageModelRequest` rather than a provider payload:

```python
from symai.loading import load_runtime
from symai.runtime.config import RuntimeConfig
from symai.runtime.models import LanguageModelRequest, TextContent, UserMessage

request = LanguageModelRequest(
    messages=(UserMessage(content=(TextContent(text="Summarize explicit ownership."),)),),
)


def generate(config: RuntimeConfig) -> str:
    with load_runtime(config) as runtime:
        response = runtime.language_model("chat").execute(request)
        return response.text
```

A request has a non-empty message tuple and optional normalized response format, reasoning, sampling, user, and metadata fields. Each adapter validates requested fields against the selected model's catalog. If the normalized request asks for a feature that model cannot satisfy, execution raises `UnsupportedFeatureError` before sending an HTTP request.

Successful execution returns `LanguageModelResponse`. Its `outputs` tuple is non-empty. Each output contains a normalized assistant message, finish reason, optional refusal, and a `text` property; `response.text` is the first output's text. Response metadata identifies the provider, the requested model, and the model the provider actually served, and may include request identifiers, token usage, and rate-limit data.

A terminal response is representable without inventing content: a content-filtered result, or one truncated while the model was still reasoning, normalizes with its finish reason, reasoning, and usage intact rather than raising.

## Observing execution

A runtime and a bound handle can both report executions for the duration of a scope:

```python
from symai.loading import load_runtime
from symai.runtime.config import RuntimeConfig
from symai.runtime.observability import ExecutionRecord


def observe(config: RuntimeConfig) -> list[ExecutionRecord]:
    records: list[ExecutionRecord] = []
    with load_runtime(config) as runtime, runtime.observe(records.append):
        del runtime
    return records
```

`runtime.observe` reports every engine; a handle's `observe` reports only that engine. Records carry safe fields: engine name, capability, provider, requested and served model, usage, rate limit, request ID, status, duration, and any error.

## Runtime lifetime

A runtime is created inactive and has one context-managed lifecycle:

1. `with load_runtime(config) as runtime` activates it and records the owner thread.
2. Execution is accepted only while that context is active, and only on the owner thread.
3. Context exit closes each owned engine exactly once, in reverse construction order.
4. A closed runtime cannot be re-entered or execute another request.

Each configured engine owns its own HTTP client, and every one is closed on exit. Cleanup attempts every engine even if one fails, and reports the failures together. Calling `close()` more than once is safe, but context management is the normal ownership pattern.

## Errors

Runtime failures use a typed hierarchy so callers can react differently: `AuthenticationError`, `PermissionDeniedError`, `InvalidRequestError`, `RateLimitError`, `ProviderError`, `TransportError`, and `InvalidResponseError`, alongside `UnsupportedCapabilityError`, `UnsupportedModelError`, `UnsupportedFeatureError`, `UnknownEngineError`, `EngineCapabilityError`, `AmbiguousEngineError`, `RuntimeClosedError`, and `RuntimeOwnershipError`.

Execution errors carry `ErrorMetadata`: provider, model, request ID, retry-after, HTTP status, the provider's error code, type, and parameter, a bounded provider message, and whether the failure class is retryable. The library classifies retryability but never retries on its own — retrying a non-idempotent request risks duplicate execution and billing, so the decision stays with the application.
