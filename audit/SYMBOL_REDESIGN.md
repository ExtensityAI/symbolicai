# Symbol / Function Redesign

Status: **approved design** · 2026-07-15

This document defines the user-facing value and operation layer that should replace the current `Symbol` / `Expression` / `Function` surface. Runtime instance ownership and engine selection are specified in [`FIXPLAN.md`](./FIXPLAN.md); observed defects remain in [`FINDINGS.md`](./FINDINGS.md).

---

## 1. Goals

1. Preserve the recognizable experience of composing operations with Symbols.
2. Make every model call explicit in code.
3. Keep values immutable and free of runtime, provider, prompt-state, persistence, and graph concerns.
4. Separate request execution from typed output decoding.
5. Make the low-level `Function` / Runtime path usable without Symbol.
6. Remove abstractions that no longer own a distinct responsibility.

## 2. Non-goals

- Tool or function calling is not supported and is not part of this redesign or roadmap.
- Symbol does not discover a Runtime, select a provider, own an engine, or perform I/O.
- Symbol does not store prompt context, execution history, embeddings, graph links, or persistence state.
- This design does not add async execution. A future async API is a separate design.
- This design does not add a generic capability/plugin system.

---

## 3. Architecture

```text
Symbol[T]                         shallow-immutable ergonomic value
    │
    ├── native Python operators  deterministic, no Runtime
    │
    └── passed explicitly to ops.*
             │
             ▼
         Function                request construction + execution
             │
             ▼
    LanguageModelResponse        normalized provider-neutral result
             │
             ▼
         decode_output          explicit typed conversion
             │
             ▼
          Symbol[T]              ergonomic operation result

Runtime                          explicit lifecycle and named-engine owner
```

Dependency direction:

```text
Symbol ← ops.* → Function → Runtime → engines → clients
                 ↓
              decoding
```

`Function`, Runtime, and decoders do not import or return Symbol. The `ops.*` layer is the only layer that wraps decoded values back into Symbols.

### 3.1 Module ownership and canonical imports

```text
symai/symbol.py       Symbol
symai/function.py     Function
symai/decoding.py     decode_output and callable decoders
symai/ops/*.py        ergonomic semantic operations
symai/runtime/*.py    Runtime lifecycle, configuration, normalized contracts
symai/contract/*.py   Design-by-Contract: Contract[In, Out], @contract, LLMDataModel
```

Canonical imports come from the owning modules:

```python
from symai.decoding import decode_output
from symai.function import Function
from symai.ops import text
from symai.symbol import Symbol
```

The package root is empty rather than a compatibility facade. Canonical imports come from owning modules, and the major-version migration guide documents them together with the Runtime model/configuration imports used by applications.

---

## 4. `Symbol[T]`: immutable value DSL

### 4.1 Purpose

Symbol is the ergonomic value layer: it keeps native and semantic results composable and visibly distinct from ordinary Python values. It is not the engine API and does not hide model execution.

If Symbol did not provide this compositional layer, a transparent wrapper around `T` would not justify reimplementing Python's data model. The contract below is therefore intentionally narrow.

### 4.2 State

A Symbol contains one held value:

```python
T = TypeVar("T")


class Symbol(Generic[T]):
    @property
    def value(self) -> T: ...
```

The wrapper is **shallow immutable**:

- the held reference cannot be reassigned through Symbol;
- Symbol exposes no item assignment/deletion or mutating methods;
- Symbol creates no mutable internal state;
- a caller-provided list, dictionary, or custom object retains its own mutability outside Symbol;
- Symbol is unhashable (`__hash__ = None`) because wrapped mutable values remain legal.

No value setter, `__setitem__`, `__delitem__`, in-place mutation implementation, or dynamic attribute assignment survives.

### 4.3 Native operator contract

Without an explicit `ops.*` call, **no operation calls a model**.

| Operation family | Result |
|---|---|
| Equality / ordering | `bool` using held-value semantics |
| Membership | `bool` using held-value semantics |
| Arithmetic / bitwise / unary value operations | new `Symbol[U]` |
| Indexing / slicing | new `Symbol[U]` |
| Iteration | iterator of Symbols |
| `len()` / truth testing | native `int` / `bool` |
| Explicit casts (`str`, `int`, `float`, `bool`) | native Python value |

Unsupported native operations propagate Python's original `TypeError`, `KeyError`, or `IndexError`. Symbol does not catch them to attempt a semantic fallback and does not replace them with generic exceptions.

There are no semantic flags, `_semantic`, `.sem`, `.syn`, or implicit native-to-model fallback. There are no mutating in-place implementations; normal Python rebinding may use the corresponding non-mutating binary operator.

### 4.4 No wrapped-value forwarding

Symbol does not implement a broad `__getattr__` that delegates arbitrary attributes to its held value. Such forwarding hides the boundary between the wrapper and `T`, creates name collisions, and makes the public API depend on every possible wrapped type. Callers explicitly unwrap with `.value` when they need the underlying object's API.

---

## 5. Semantic operations are explicit free functions

Named semantic operations live in focused namespaces. Operations that perform I/O take a bound engine handle; deterministic local operations accept neither a handle nor a Runtime:

```python
from symai.ops import reason, text
from symai.symbol import Symbol

source = Symbol("A long passage")
summary = text.summarize(runtime.language_model("tenant-a"), source)
answer = reason.query(runtime.language_model("tenant-b"), source, "What is the thesis?")
```

Rules:

- public ergonomic operations accept Symbols, not an ambiguous `Symbol | object` union;
- auxiliary instructions and configuration remain ordinary typed values rather than meaningless wrapped Symbols; for example `is_instance_of` takes a `type_description: str`, while equality and containment take their compared value as a Symbol;
- operations never mutate their input;
- operations return a new `Symbol[T]` with the documented result type;
- operations that perform I/O take a bound engine handle (`runtime.language_model(name)` / `runtime.embedding(name)`) and never receive provider or model details;
- language operations use Function plus an appropriate decoder internally;
- `ops.embed.embed` executes a normalized embedding request directly because Function is language-only;
- deterministic `ops.text.template` and `ops.embed` similarity, distance, MMD, and kernel functions are local, accept no Runtime or engine, and perform no I/O;
- provider/model selection never appears in operation-specific options;
- operations with no semantic or focused deterministic behavior stay ordinary Python operations rather than being mirrored in `ops.*`.

Initial namespaces:

```text
ops.text       summarize, translate, modify, filter, map, convert, style,
               template, replace, include, combine, extract
ops.reason     query, interpret, logic
ops.compare    equals, contains, is_instance_of
ops.rank       rank
ops.embed      embed; similarity (cosine, dot); distance (euclidean,
               manhattan, minkowski); MMD (RBF); kernel (linear, RBF,
               polynomial)
```

Persistence is not an operation namespace. Built-in Symbol persistence is removed.

---

## 6. `Function`: execution without output typing

### 6.1 Contract

Function constructs and executes a language request. It is deliberately **not generic** and has no `return_type`, `sym_return_type`, `default`, `limit`, `preview`, or `return_metadata` mode flags.

Conceptual API:

```python
class Function:
    def request(self, *values: object) -> LanguageModelRequest: ...

    def __call__(
        self,
        engine: LanguageModel,
        *values: object,
    ) -> LanguageModelResponse: ...

    def execute_many(
        self,
        engine: LanguageModel,
        inputs: Sequence[object],
    ) -> tuple[LanguageModelResponse, ...]: ...
```

Usage:

```python
sentiment = Function("Classify the sentiment.")
request = sentiment.request("The preview path performs no I/O.")
response = sentiment(runtime.language_model("tenant-a"), "The result was excellent.")
request_id = response.metadata.request_id
```

`request()` replaces `preview=True`. Metadata is always present on the normalized response. `execute_many` is documented as stable-order sequential execution, not a provider batch API.

Function owns immutable request instructions and normalized request options. There is no framework-level static/dynamic context concept. Per-call information is supplied as values or normalized messages.

### 6.2 Structured response requests

Request response format remains explicit normalized request configuration. Asking a model for JSON or JSON Schema output and validating the returned text are separate concerns. Function does not infer request behavior from a Python return annotation.

---

## 7. Typed decoding is a separate stage

Typing belongs to a decoder that carries real parsing behavior. A decoder is any
`Callable[[str], T]`; there is no decoder class hierarchy, and a plain string result needs no
decoder at all (`response.text`):

```python
T = TypeVar("T")


def decode_output(
    response: LanguageModelResponse,
    decoder: Callable[[str], T],
    *,
    output_index: int = 0,
    default: T | Missing = MISSING,
    limit: int | None = None,
) -> T: ...
```

Examples:

```python
answer: str = response.text
score: int = decode_output(response, int)
users: list[User] = decode_output(response, TypeAdapter(list[User]).validate_json)
```

Decoder rules:

- the decoder, not Function, determines `T`;
- decoder failures use one explicit `DecodeError` family;
- `default` catches only the documented decode failure, never transport, selection, or programming errors;
- output index selection is deterministic and raises `IndexError` when absent;
- collection limiting is post-decode and deterministic;
- sets pass through because deterministic truncation is undefined;
- nested/container typing uses `TypeAdapter`, not bare runtime classes;
- Function and Runtime errors remain distinguishable from decode errors.

A typed `DecodedFunction[T]` convenience wrapper is not part of the initial surface. It may be added only after repeated callers demonstrate that the explicit two-stage form is burdensome.

---

## 8. Removed concepts

| Removed | Reason |
|---|---|
| `Expression` | No distinct role remains after graph and prompt helpers are removed; ordinary callables compose operations. |
| `Result` | No production consumer and no state beyond the obsolete Expression wrapper. |
| Graph/linker/root/nodes/edges/results | Unused retained machinery with unbounded result retention. |
| `Expression.prompt` | Function / Runtime already own explicit request execution. |
| `sym_return_type` | Function returns a normalized response; decoding returns raw `T`; ops wrap explicitly. |
| `.sem`, `.syn`, `_semantic` | Model execution is an explicit `ops.*` call. |
| `static_context`, `dynamic_context`, `global_context` | Hidden prompt state on values is removed. |
| `adapt()` / `clear()` | Mutable process-wide prompt state is removed. |
| Symbol embedding cache | Derived execution state does not belong on an immutable value. |
| `save()` / `load()` | Persistence is not intrinsic to Symbol; current pickle loading is executable and current save collision handling can overwrite data. |
| broad `__getattr__` forwarding | Makes the wrapper API depend on arbitrary held types. |
| operation mixin inheritance | The value type should not inherit unrelated execution, embedding, and persistence surfaces. |
| `Prompt` / `PromptRegistry` hierarchy | Function accepts immutable string examples directly; focused operations own private immutable example tuples, so a mutable public prompt registry and class-per-example hierarchy add no domain capability. |

If durable Symbol artifacts are required later, they need a separate versioned, validated, non-executable codec design. Generic pickle loading is not retained as compatibility API.

---

## 9. Runtime boundary

Function and semantic operations receive a bound engine handle obtained from the Runtime. `current_runtime()` and ambient `ContextVar` discovery are removed.

Runtime owns named configured engine instances. Two instances may use the same provider/model with different credentials and transports. `runtime.language_model(name)` and `runtime.embedding(name)` return a bound handle (`LanguageModel` / `EmbeddingModel`) that callers pass to operations and Function without knowing provider details. The low-level `runtime.execute(request, engine=<name>)` path remains as an escape hatch.

The synchronous Runtime has one owner thread. Async execution, if needed, will use a separately designed `AsyncRuntime` rather than weakening the synchronous lifecycle contract.

---

## 10. Compatibility

This is a clean major-version cutover:

- no `Symbol.__getattr__` forwarding shim for moved semantic operations;
- no compatibility aliases for Expression, Result, graph state, context state, persistence, or `sym_return_type`;
- no implicit semantic fallback period;
- migration documentation maps each retained semantic method to its `ops.*` function;
- callers that need raw results use Function + decoder; callers that want the value DSL use `ops.*` and Symbols.

A forwarding shim would conflict with explicit wrapped-value access, hide raw-versus-Symbol return changes, and keep the god-object surface alive during the exact release intended to remove it.

---

## 11. Acceptance criteria

### Symbol

- state cannot be reassigned through Symbol;
- Symbol is unhashable;
- no Runtime/client/provider import exists below Symbol;
- no context, graph, persistence, embedding, or semantic-mode state remains;
- native operators never perform I/O;
- operator result types match the table in §4.3;
- original Python exceptions propagate from invalid native operations.

### Function and decoding

- Function calls return `LanguageModelResponse` only;
- request preview is `request()` and performs no I/O;
- response metadata needs no mode flag;
- scalar, boolean, Pydantic model, nested container, default, limit, and output-index decoding are independently covered;
- static checking proves the `Callable[[str], T]` decoder flows to `decode_output(...) -> T`;
- sequential multi-execution has stable ordering and honest naming.

### Operations

- each operation that performs I/O takes a bound engine handle and Symbol explicitly;
- deterministic local operations take Symbols without a handle or Runtime;
- each operation returns a new Symbol with no input mutation;
- operations forward the bound engine handle without knowing provider or model;
- there is no provider/model option at the operation layer;
- no tool-calling request or output type exists.

### Removal

- Expression, Result, graph/linker state, ambient contexts, Symbol persistence, semantic flags, and forwarding mixins are absent from production and tests;
- no deprecated aliases or forwarding shims remain;
- documentation contains no examples using removed concepts.

---

## 12. Rejected alternatives

### Remove Symbol entirely

This is the smallest technical API, but it loses the intentional ergonomic value layer. Symbol remains only because composing operations with immutable Symbols is an explicit product goal.

### Keep a generic `Function[T]`

Python still requires a runtime decoder, making `T` and the decoder redundant sources of truth. It also leaves execution, parsing, defaults, metadata, preview, and batching coupled in one class.

### Put a decoder on Function

`Function(prompt, decoder=...) -> T` is statically workable but still conflates remote execution with local conversion. It may be introduced later as a small composition wrapper if repetition proves the need.

### Retain ambient Runtime discovery

Ambient discovery hides ownership, prevents two same-context runtimes from being equally addressable, and preserves cross-context lifecycle failure modes. Explicit Runtime passing is simpler and supports named engines directly.

### Deep-freeze every Symbol value

Recursively converting lists, dictionaries, sets, and custom objects changes `T` and creates a second object model. The chosen contract freezes the wrapper, disallows Symbol mutation, and remains unhashable.
