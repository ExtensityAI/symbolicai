# Migrating from 1.x to 2.0

Version 2.0 replaces the public API. There are no deprecation shims, no forwarding
fallbacks, and no compatibility aliases: 1.x code does not run on 2.0, and the failure is
an immediate `ImportError` or `AttributeError` rather than a silent behaviour change.

Three ideas explain nearly every rename below.

1. **Execution is explicit.** Nothing reaches a provider unless you passed it something
   that can. There is no ambient runtime, no global registry, and no config file read at
   import time.
2. **`Symbol` is a value, not an engine.** It holds a Python object immutably and does no
   I/O. The semantic operations that used to be its methods are now free functions that
   take an engine handle.
3. **Typing is a decoding stage.** A call returns a `LanguageModelResponse`. Turning that
   into an `int`, a `bool`, or a Pydantic model is a separate, explicit step.

## Configuration

1.x resolved engines from a `symai.config.json` discovered in the current directory, the
Python environment, or your home directory, and constructed engines from an ambient
singleton. That mechanism is gone, along with `symai.config_manager`, `__root_dir__`, and
the `.symai/` directories.

2.0 takes configuration as an argument. Each engine gets a name you choose, and you select
it by that name:

```python
from pydantic import SecretStr

from symai.loading import load_runtime
from symai.runtime.config import EngineConfig, RuntimeConfig

config = RuntimeConfig(
    language_models={
        "chat": EngineConfig(
            implementation="openai:responses",
            settings={"api_key": SecretStr("..."), "model": "gpt-5.4"},
        ),
    },
    embeddings={
        "vectors": EngineConfig(
            implementation="openai:embeddings",
            settings={"api_key": SecretStr("..."), "model": "text-embedding-3-small"},
        ),
    },
)

with load_runtime(config) as runtime:
    model = runtime.language_model("chat")
    vectors = runtime.embedding("vectors")
```

`runtime.language_model(name)` returns a bound handle. Pass that handle to operations; they
never learn which provider or model is behind it. The runtime owns every HTTP transport and
closes them when the `with` block exits, including on exception.

To keep configuration in a file, parse it yourself and validate it — this replaces the
implicit discovery, and it means the file's location and precedence are yours to decide:

```python
import tomllib
from pathlib import Path

from symai.runtime.config import RuntimeConfig

config = RuntimeConfig.model_validate(tomllib.loads(Path("engines.toml").read_text()))
```

## `Symbol`

`Symbol` is now an immutable wrapper with no semantic surface. It is unhashable, its state
cannot be reassigned, and it imports nothing from the runtime. Native Python operators work
on the held value and never perform I/O:

```python
from symai.symbol import Symbol

total = Symbol(2) + Symbol(3)
assert total.value == 5
```

The syntactic/semantic distinction is gone. There is no `.sem`, no `.syn`, and no
`.value = ...` assignment. Operators such as `&`, `|`, and `^` are plain Python operators
again; the semantic versions became `reason.logic(...)`.

## Semantic operations

Every semantic method moved to a free function that takes an engine handle first and
returns a **new** `Symbol`. The input is never mutated.

```python
from symai.ops import text

summary = text.summarize(model, Symbol("a long document"))
```

| 1.x `Symbol` method | 2.0 function | Notes |
| --- | --- | --- |
| `.summarize(context=...)` | `text.summarize(model, source)` | `context` removed |
| `.translate(language=...)` | `text.translate(model, source, language)` | `language` now required |
| `.modify(changes)` | `text.modify(model, source, changes)` | |
| `.filter(criteria, include=...)` | `text.filter(model, source, criteria)` | `include` removed |
| `.map(instruction)` | `text.map(model, source, instruction)` | |
| `.convert(format)` | `text.convert(model, source, format)` | |
| `.style(description, libraries=...)` | `text.style(model, source, description)` | `libraries` removed |
| `.replace(old, new)` | `text.replace(model, source, old, new)` | |
| `.include(information)` | `text.include(model, source, information)` | |
| `.combine(information)` | `text.combine(model, left, right)` | takes a `Symbol`, not a `str` |
| `.extract(pattern)` | `text.extract(model, source, pattern)` | |
| `.template(template, placeholder=...)` | `text.template(source, template, placeholder=...)` | local; takes no handle |
| `.query(context, prompt=..., examples=...)` | `reason.query(model, source, question)` | `prompt`/`examples` removed |
| `.interpret(prompt=..., accumulate=...)` | `reason.interpret(model, source)` | both options removed |
| `&` / `\|` / `^` (semantic) | `reason.logic(model, left, operator, right)` | operators are syntactic again |
| `.equals(string, context=...)` | `compare.equals(model, left, right)` | returns `Symbol[bool]` |
| `.contains(element)` | `compare.contains(model, container, element)` | returns `Symbol[bool]` |
| `.isinstanceof(query)` | `compare.is_instance_of(model, source, type_description)` | renamed |
| `.rank(measure=..., order=...)` | `rank.rank(model, source, measure, order="desc")` | `measure` now required |
| `.embed()` / `.embedding` | `embed.embed(model, source)` | the property is removed |
| `.similarity(other, metric=...)` | `embed.similarity(left, right, metric=...)` | local; `cosine` or `dot` only |
| `.distance(other, kernel=...)` | `embed.distance(left, right, metric=...)` | local; euclidean, manhattan, minkowski |
| `.distance(other, kernel=<kernel>)` | `embed.kernel(left, right, kind=...)` | split out; linear, rbf, polynomial |
| `.distance(other, kernel="mmd")` | `embed.mmd(left, right, gamma=...)` | split out; RBF only, bounded |

`similarity`, `distance`, `kernel`, `mmd`, and `template` are deterministic and local: they
take Symbols and no handle.

### Removed with no replacement

These 1.x methods have no 2.0 successor. Most were thin prompt wrappers, engine-coupled
state, or I/O on the value; rebuild them as your own `Function` call plus a decoder if you
need them.

`clean`, `outline`, `remove`, `unique`, `compose`, `correct`, `choice`, `transcribe`,
`analyze`, `execute`, `fexecute`, `simulate`, `sufficient`, `list`, `foreach`, `stream`,
`ftry`, `dict`, `cluster`, `zip`, `input`, `open`, `expand`, `save`, `load`, `output`,
`tune`, `data`, `cast`, `to`, `ast`, `str`, `int`, `float`, `bool`, `size`, `tokens`,
`tokenizer`, `type`, `value_type`, `index`, `split`, `join`, `startswith`, `endswith`,
`init_results`, `get_results`, `clear_results`, `syn`, `sem`.

`Expression`, `Result`, the symbol graph and linker, `Symbol` persistence, and
`GlobalSymbolPrimitive` are removed outright. So are the console entry points and the
`.symai/` configuration tree.

The `static_context` and `dynamic_context` class attributes are gone with `Expression`.
They injected prompt text ambiently, from the class body of whatever `Expression` you had
subclassed, which made a request's real content impossible to read off the call site. Put
that text where it now belongs — in the instruction you pass to `Function`, or in the
prompt an `ops.*` function builds — and use `Function.request()` to inspect exactly what
would be sent.

## `Function` and typed decoding

1.x combined execution and output typing through `sym_return_type` and `Expression`
subclasses. 2.0 splits them. `Function` performs execution and returns a
`LanguageModelResponse` — always, with no mode flag and no metadata toggle:

```python
from symai.decoding import decode_output
from symai.function import Function

answer = Function(model, "Answer with a single integer.")
response = answer("How many moons does Mars have?")
count = decode_output(response, int)
```

`request()` builds the request for inspection and performs no I/O. `execute_many` runs
inputs sequentially and preserves their order.

A decoder is any `Callable[[str], T]`, so `int`, `float`, and your own functions work
directly. There is no decoder class hierarchy — `TextDecoder`, `ConstructorDecoder`, and
`TypeAdapterDecoder` are gone. `symai.decoding` provides `decode_text`, `decode_bool`, and
`scalar_decoder`; `decode_output` adds output selection, an optional `default`, and a
`limit`. Static type checkers infer `T` from the decoder, so `decode_output(response, int)`
is an `int`.

## Contracts

`@contract` remains, and `LLMDataModel` is still the base for its input and output models.
Import them from their canonical modules rather than the package root:

```python
from symai.contract.decorator import contract
from symai.contract.models import LLMDataModel
```

## Errors

Errors are typed and carry structured metadata instead of provider text. `ErrorMetadata`
exposes `status_code`, `error_code`, `error_type`, `param`, `provider_message`, and
`retryable`. Provider responses are never interpolated raw into exception messages, and
credential validation failures are deliberately indistinguishable from one another.
