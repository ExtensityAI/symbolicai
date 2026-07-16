# SymbolicAI

Explicit typed runtimes for OpenAI, Cerebras, and DeepSeek language models and OpenAI embeddings.

SymbolicAI gives applications one normalized request/response contract across the supported providers. The application owns configuration and credentials, loads a `Runtime`, enters it as a context manager, and receives normalized responses rather than provider SDK objects. Every model call is explicit in code: nothing is discovered from the environment, and no operation reaches a provider unless you pass it an engine handle.

## Supported capabilities

| Capability | Implementation ID | Example model |
| --- | --- | --- |
| Language model | `openai:responses` | `gpt-5.4` |
| Language model | `cerebras:chat-completions` | `gpt-oss-120b` |
| Language model | `deepseek:chat-completions` | `deepseek-v4-flash` |
| Text embeddings | `openai:embeddings` | `text-embedding-3-small` |

Model IDs are unqualified. The implementation ID selects the provider adapter; do not prefix a model ID with a provider name.

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

Applications provide API keys directly in engine settings. SymbolicAI does not read environment variables or configuration files on your behalf.

## Quickstart

The package root is empty: import from the module that owns each name.

```python
import os

from pydantic import SecretStr

from symai.loading import load_runtime
from symai.runtime.config import EngineConfig, RuntimeConfig
from symai.runtime.models import EmbeddingRequest, LanguageModelRequest, TextContent, UserMessage

api_key = SecretStr(os.environ["OPENAI_API_KEY"])
config = RuntimeConfig(
    language_models={
        "chat": EngineConfig(
            implementation="openai:responses",
            settings={"api_key": api_key, "model": "gpt-5.4"},
        )
    },
    embeddings={
        "vectors": EngineConfig(
            implementation="openai:embeddings",
            settings={"api_key": api_key, "model": "text-embedding-3-small"},
        )
    },
)


def run() -> tuple[str, tuple[float, ...]]:
    with load_runtime(config) as runtime:
        language_response = runtime.language_model("chat").execute(
            LanguageModelRequest(
                messages=(UserMessage(content=(TextContent(text="Say hello in German."),)),)
            )
        )
        embedding_response = runtime.embedding("vectors").execute(
            EmbeddingRequest(inputs=("first", "second"), dimensions=512)
        )
        return language_response.text, embedding_response.vectors[0].values
```

`RuntimeConfig` is immutable and requires at least one engine. Names are yours to choose and identify an engine within its capability; the same provider and model may be configured twice under different names with different credentials, and each configured engine owns its own HTTP client. `load_runtime` resolves every configuration before allocating any transport, and closes what it built in reverse order if construction fails partway.

A Runtime has a single lifecycle, cannot be re-entered, and may only be used from the thread that entered it. Closing it closes every engine exactly once and reports all cleanup failures together.

### Engine handles

`runtime.language_model(name)` and `runtime.embedding(name)` return a bound handle. Handles are what you pass to functions and operations, so a wrong-capability mistake is a type error rather than a runtime one. The name may be omitted when exactly one engine provides that capability; with more than one, omitting it raises rather than guessing.

`runtime.execute(request, engine=name)` remains available as a low-level escape hatch for dynamic routing.

## Normalized contracts

Requests and responses are provider-neutral:

- `LanguageModelRequest` contains typed messages, response format, reasoning, sampling, user, and metadata fields.
- `LanguageModelResponse.text` is the first output's text; `response.outputs` exposes every normalized output.
- `EmbeddingRequest` contains one or more text inputs plus an optional dimension count.
- `EmbeddingResponse.vectors` contains normalized vectors; use `response.vectors[0].values` for values.

Adapters validate a normalized request against the selected model's catalog entry and raise a typed error before transport when the model cannot satisfy it. Provider failures arrive as typed errors — `AuthenticationError`, `PermissionDeniedError`, `InvalidRequestError`, `RateLimitError`, `ProviderError`, `TransportError`, `InvalidResponseError` — carrying safe, bounded metadata. Prompts, credentials, and raw provider bodies never appear in exception messages or logs.

## Functions and decoding

`Function` builds and executes one request. It is not generic and does not decode:

```python
from symai.decoding import decode_output
from symai.function import Function

classify = Function("Classify the sentiment as positive or negative.")

with load_runtime(config) as runtime:
    chat = runtime.language_model("chat")
    request = classify.request("The result was excellent.")  # no I/O
    response = classify(chat, "The result was excellent.")
    label = response.text
    score = decode_output(classify(chat, "Rate this 1-10: great"), int)
```

Decoding is a separate stage, and a decoder is any `Callable[[str], T]` — `int`, `decode_text`, `decode_bool`, `scalar_decoder(int)`, or `TypeAdapter(list[User]).validate_json`. `decode_output` selects an output by index, decodes it, and applies an optional limit. A `default` replaces a decode failure only; selection errors and decoder bugs always propagate.

## Symbols and operations

`Symbol` is a shallow-immutable value wrapper. Native operators are deterministic and never call a model:

```python
from symai.ops import text
from symai.symbol import Symbol

source = Symbol("A long passage.")
doubled = Symbol(2) * 3          # Symbol(6), no I/O

with load_runtime(config) as runtime:
    summary = text.summarize(runtime.language_model("chat"), source)
```

Semantic operations are explicit free functions taking an engine handle and a Symbol, returning a new Symbol. Namespaces: `ops.text`, `ops.reason`, `ops.compare`, `ops.rank`, `ops.embed`. Deterministic operations such as `ops.text.template` and the `ops.embed` similarity, distance, MMD, and kernel functions take no handle and perform no I/O.

## Contracts

The `@contract` decorator enforces typed input and output with pre/post conditions, natural-language semantic conditions, and a self-healing remediation loop:

```python
from symai.contract.decorator import contract
from symai.contract.models import LLMDataModel


class Review(LLMDataModel):
    text: str


class Verdict(LLMDataModel):
    label: str


@contract(post_remedy=True)
class Classify:
    prompt = "Classify the sentiment as positive or negative."
    semantic_conditions = ("The label must be positive or negative.",)

    def pre(self, review: Review) -> None:
        if not review.text.strip():
            msg = "review text must not be empty"
            raise ValueError(msg)

    def forward(self, review: Review) -> Verdict:
        return self.contract_result


with load_runtime(config) as runtime:
    classify = Classify(runtime.language_model("chat"))
    verdict = classify(Review(text="The result was excellent."))
```

A failed validation does not prevent `forward` from running, so you can return a fallback; `self.contract_successful`, `self.contract_result`, and `self.contract_exception` report what happened, and `contract_perf_stats()` reports timings, token usage, and per-provider totals.

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
uv run pyright
```

## License

SymbolicAI is distributed under the BSD 3-Clause license. See [LICENSE](LICENSE).
