# r1-08 · Public API surface, naming, ergonomics

> **Historical snapshot terminology.** References and examples below using `EngineSpec` or
> configured `default_*` fields describe the audited code, not the final API. The target uses
> `EngineConfig`, bound engine handles, and sole-engine-only unnamed selection.

Lens: least-surprise for a new user. Scope: `symai/__init__.py`, `ops.*` naming,
`ImplementationId`, `load_runtime` layering, decoders, error/sentinel surface.
Snapshot moving target — anchored by symbol + snippet, line numbers approximate.

## Executive summary

1. **The root `__init__.py` is a 68-name re-export facade that directly contradicts
   the approved design AND fails the project's own test suite.** Design §3.2 and §10
   say "the package root is empty rather than a compatibility facade"; the user's
   Python style guide says "empty `__init__.py` — no re-exports, no `__all__`";
   `test_public_cutover.py` asserts `not hasattr(symai, "__all__")` and that
   `Function`/`Runtime`/`load_runtime`/`current_runtime`/… are absent. All of that is
   violated by the live file. **6 of 10 cutover tests fail today** because of this and
   the un-deleted `prompts.py`.
2. The facade is not just wrong-per-spec, it is **internally incoherent**: it exports
   `Function`, decoders, runtime models, and errors, but omits the two most
   user-facing names — `Symbol` and the `ops.*` namespaces. A partial facade that
   drops the headline type is the worst of both worlds.
3. Real ergonomic friction confirmed: `cast("ImplementationId", ...)` in `loading.py`
   is a **redundant no-op cast** (pyright: 0 errors without it) that also falsely
   implies static safety the branded alias does not provide.
4. Genuinely good and worth keeping: op signature uniformity
   (`runtime, source, …, *, engine`), the `*Decoder` family + `decode_output`, the
   split of decode errors (`DecodeError(ValueError)`) from the runtime error tree, and
   the deterministic-vs-I/O op split.
5. Overall read: the *intended* surface (owning-module imports, empty root) is small,
   obvious, and consistent. The blocker is that production has not finished cutting
   over to it — the root facade and its dead exports (`current_runtime`,
   `NoActiveRuntimeError`) must go.

## Findings

| ID | Finding | Conf | Impact | Effort |
|----|---------|------|--------|--------|
| API-01 | Root `__init__.py` is a facade; design + test + style-guide all mandate an empty root | high | high | S |
| API-02 | Facade is incoherent: omits `Symbol` and `ops.*` while exporting everything else | high | high | S |
| API-03 | Facade forces eager import of the whole stack (incl. legacy `prompts.py`), defeating inert-import goal | high | med | S |
| API-04 | `cast("ImplementationId", …)` is a redundant no-op cast; branded alias gives zero static safety | high | med | S |
| API-05 | Dead exports for a removed feature: `current_runtime` + `NoActiveRuntimeError` in `__all__` | high | med | S |
| API-06 | Two same-named `load_runtime` in sibling modules; public entry only obvious once documented | med | low | S |
| API-07 | `ops.text.filter`/`map` shadow builtins — acceptable DSL, contained, but a footgun on direct import | med | low | S |
| API-08 | Overlapping "capability/unsupported" error names need a disambiguation table | low | low | S |

---

### API-01 — The root is a compatibility facade the design explicitly forbids

**What.** `symai/__init__.py` imports ~74 names and publishes an `__all__` of 68.
This is exactly the "compatibility facade" the approved design rejects.

**Where.** `symai/__init__.py`:

```python
from symai.function import Function
from symai.loading import load_runtime
from symai.runtime.runtime import Runtime, current_runtime
...
__all__ = [
    "AmbiguousEngineError", "AssistantMessage", ..., "current_runtime",
    "decode_output", "load_runtime",
]   # 68 entries
```

Against `SYMBOL_REDESIGN.md` §3.2:

> The package root is empty rather than a compatibility facade. Canonical imports come
> from owning modules …

…and §10 ("clean major-version cutover"). The user's Python style guide is even
blunter: *"Empty `__init__.py` files — no re-exports, no `__all__`."*

**Verified failing.** `test_public_cutover.py::test_old_root_names_are_absent_after_canonical_imports`:

```python
import symai
assert not hasattr(symai, "__all__")
for name in OLD_ROOT_NAMES | FORBIDDEN_PUBLIC_NAMES:
    assert not hasattr(symai, name)
```

Live behaviour (measured):

```
has __all__: True   len __all__: 68
Function: True   Runtime: True   load_runtime: True   current_runtime: True   NoActiveRuntimeError: True
```

`pytest tests/test_public_cutover.py` → **6 failed, 4 passed** (this test plus the
inert-import, ambient-registry, deleted-tree, and AST-guard tests; the last two are
driven by the un-deleted `prompts.py`, out of this lens but same cutover).

**Why it matters.** The single most important API-surface decision of the release is
undone by a leftover file. It also creates drift: every name added to a module must be
mirrored here or the facade rots.

**Proposed change.** Empty `symai/__init__.py` (or leave only a package docstring).
Publish nothing at root. Move the entire public surface to owning-module imports
(see "Proposed minimal public surface" below). `keeps-all` — no capability is lost;
callers change import lines only, which is expected in a breaking major.

**Feature impact:** keeps-all · **Confidence:** high · **Impact:** high · **Effort:** S.

---

### API-02 — If a facade is kept at all, this one is incoherent

**What.** The facade exports `Function`, the four decoders, `decode_output`, ~40
runtime model types, and 15 error types — but **not** `Symbol` and **not** the
`ops.*` namespaces, which are the two things a new user reaches for first.

**Where.** `symai/__init__.py` has no `from symai.symbol import Symbol` and no
`from symai import ops`. Measured: `from symai import Symbol` → `AttributeError`
(`hasattr(symai, "Symbol") == False`), while `from symai import Function` works.

The "hello world" a user must currently write is therefore split across two idioms:

```python
from symai import Function, Runtime, RuntimeConfig, EngineSpec, load_runtime  # root
from symai.symbol import Symbol      # NOT at root
from symai.ops import text           # NOT at root
```

**Why it matters.** Least-surprise is violated in both directions: a user who learns
"import from `symai`" hits a wall on `Symbol`; a user who learns "import from owning
modules" finds half the API redundantly also at root. `test_public_cutover.py` lists
`Symbol` in *both* `OLD_ROOT_NAMES` and `FORBIDDEN_PUBLIC_NAMES`, confirming the
intended answer is "not at root" — which only makes sense if *nothing* is at root.

**Proposed change.** Same as API-01: empty the root so every name has exactly one
canonical home. Do **not** "fix" the incoherence by adding `Symbol`/`ops` to the
facade — that would re-introduce the god-facade the design deletes.

**Feature impact:** keeps-all · **Confidence:** high · **Impact:** high · **Effort:** S.

---

### API-03 — The facade forces eager import of the whole stack, including legacy `prompts.py`

**What.** Because the root imports `function`, `decoding`, `loading`, and all of
`runtime.*`, a bare `import symai` transitively loads 13 submodules and binds 74
public names. `function.py` imports `symai.prompts` (the 1042-LOC legacy module with
jinja2 / tomllib / python-box deps), so that loads too.

**Where.** Measured on `import symai`:

```
symai submodules loaded on import: 13
   symai.decoding  symai.function  symai.loading  symai.operations
   symai.prompts   symai.runtime.config  symai.runtime.loading
   symai.runtime.models  symai.runtime.runtime  ...
public names at root: 74
```

The intended contract is `test_public_cutover.py::test_import_symai_is_subprocess_isolated_and_inert`:

```python
assert observed == {
    ...,
    "public_names": [],
    "symai_modules": ["symai"],
    ...
}
```

i.e. `import symai` should load only `symai` itself and expose nothing.

**Why it matters.** This is the API-surface half of the standing "eager engine loading"
concern. An empty root makes `import symai` inert and lets `prompts.py`/heavy deps load
only when actually reached, aligning the surface with lazy load-on-demand.

**Proposed change.** Empty root (API-01) achieves this directly. Separately, decouple
`Function` from `prompts.Prompt` so importing `symai.function` does not drag in
jinja2/box (tracked by other lenses).

**Feature impact:** keeps-all · **Confidence:** high · **Impact:** med · **Effort:** S
(root) — the prompts decoupling is a separate M.

---

### API-04 — `cast("ImplementationId", …)` is redundant friction with no payoff

**What.** Every builtin loader entry casts a string literal to `ImplementationId`.
The cast is a **no-op**: `ImplementationId = Annotated[str, BeforeValidator(...)]`
is statically just `str` (PEP 593), so a bare literal already satisfies the type.

**Where.** `symai/loading.py`:

```python
BUILTIN_LANGUAGE_MODEL_LOADERS: tuple[LanguageModelLoaderEntry, ...] = (
    (cast("ImplementationId", "openai:responses"), _load_openai_responses),
    (cast("ImplementationId", "cerebras:chat-completions"), _load_cerebras_chat_completions),
    (cast("ImplementationId", "deepseek:chat-completions"), _load_deepseek_chat_completions),
)
```

`symai/runtime/config.py`:

```python
ImplementationId = Annotated[str, BeforeValidator(_normalize_implementation_id)]
```

**Verified.** A scratch module type-checked against the worktree:

```python
entry: LanguageModelLoaderEntry = ("openai:responses", loader)   # no cast
plain: ImplementationId = "openai:responses"                     # no cast
```

→ `pyright: 0 errors, 0 warnings`. The casts add noise and, worse, imply
`ImplementationId` is a distinct type guarding call sites — it is not. Any extension
author registering a custom loader will cargo-cult the same useless `cast`.

**Why it matters.** This is the most concrete ergonomic wart in the loader surface —
the first thing a plugin author copies. Runtime validation already happens in
`runtime/loading.py::_index_entries` via `_IMPLEMENTATION_ID_ADAPTER.validate_python`,
so the static brand buys nothing.

**Proposed change** (pick one, in order of preference):
- **(a) Drop the brand.** Type loader-entry keys as plain `str`; keep runtime
  validation in `_index_entries`. Call sites become `("openai:responses", loader)`.
- **(b) A discoverable constant set.** The *builtin* implementations are a known set,
  so a `StrEnum` fits the project's "known-set str → StrEnum" rule and is str-compatible
  (extensions still pass raw `str`):

  ```python
  class BuiltinImplementation(StrEnum):
      OPENAI_RESPONSES = "openai:responses"
      OPENAI_EMBEDDINGS = "openai:embeddings"
      CEREBRAS_CHAT = "cerebras:chat-completions"
      DEEPSEEK_CHAT = "deepseek:chat-completions"
  ```
  Then `(BuiltinImplementation.OPENAI_RESPONSES, _load_openai_responses)` — no cast,
  autocompletion, typo-safe for builtins.

Avoid `NewType` — it needs `ImplementationId("openai:responses")` at every site and
skips the `BeforeValidator`, so it is friction without the normalization.

**Feature impact:** keeps-all · **Confidence:** high · **Impact:** med · **Effort:** S.

---

### API-05 — Dead exports: `current_runtime` and `NoActiveRuntimeError`

**What.** Both are in the root `__all__`, but the design removed ambient runtime
discovery (§9: "`current_runtime()` and ambient `ContextVar` discovery are removed").
`NoActiveRuntimeError` is *only* raised by the to-be-removed `current_runtime()`; no
op, `Function`, or decoder references it.

**Where.** `symai/__init__.py` `__all__` contains `"current_runtime"` and
`"NoActiveRuntimeError"`. `grep` confirms `NoActiveRuntimeError` is raised solely in
`runtime/runtime.py::current_runtime`, which the design deletes; ops/function/decoding
never touch either. `test_public_cutover.py` asserts the runtime module exposes
neither (currently failing on `_CURRENT_RUNTIME`).

**Why it matters.** Shipping a public error/function for a deleted feature is exactly
the "keep the god-object surface alive during the release meant to remove it" trap the
design warns against (§10). New users will discover and depend on an API that is slated
to vanish.

**Proposed change.** Remove `current_runtime`, `_CURRENT_RUNTIME`, and
`NoActiveRuntimeError` from production (runtime lens owns the ContextVar removal); they
must not appear in any public surface. `drops-minimal` — only the already-rejected
ambient-discovery convenience.

**Feature impact:** drops-minimal (ambient discovery, already rejected by design) ·
**Confidence:** high · **Impact:** med · **Effort:** S.

---

### API-06 — Two `load_runtime` functions with the same name

**What.** `symai.loading.load_runtime` (public: composes builtins, takes just a
`RuntimeConfig`) and `symai.runtime.loading.load_runtime` (generic: requires explicit
loader lists). The public module imports the generic as `_load_runtime`.

**Where.** `symai/loading.py`:

```python
from symai.runtime.loading import (... load_runtime as _load_runtime)

def load_runtime(config: RuntimeConfig, *, language_model_loaders=(), embedding_loaders=()) -> Runtime:
    """Compose immutable built-ins with explicit extension entries and load a Runtime."""
    return _load_runtime(config, language_model_loaders=(*BUILTIN_..., *language_model_loaders), ...)
```

**Why it matters.** With an empty root (API-01), the user must *know* the entry lives
in `symai.loading`. Two identically-named functions in sibling modules make grep/goto
ambiguous and blur which is the supported entry point.

**Proposed change.** Keep `symai.loading.load_runtime` as the one public entry; rename
the generic to `compose_runtime` (or `load_runtime_from_loaders`) so the names read as
"public entry" vs "low-level composer." Document `from symai.loading import load_runtime`
as the canonical entry in the migration guide.

**Feature impact:** keeps-all · **Confidence:** med · **Impact:** low · **Effort:** S.

---

### API-07 — `ops.text.filter` / `map` shadow builtins

**What.** `ops/text.py` defines and exports `filter`, `map`, `convert`, `template`;
`convert` takes a `format` parameter. `filter`/`map`/`format` are Python builtins.

**Where.** `symai/ops/text.py`:

```python
__all__ = ("summarize", "translate", "modify", "filter", "map", "convert", ...)
def filter[T](runtime, source, criteria, *, engine=None) -> Symbol[str]: ...
def map[T](runtime, source, instruction, *, engine=None) -> Symbol[str]: ...
def convert[T](runtime, source, format, *, engine=None) -> Symbol[str]: ...
```

**Assessment — mostly a nicety, not a footgun.** Verified the shadowing is *contained*:
`text.py` never calls builtin `map`/`filter`/`format` after redefining them (grep
returns nothing). At the call site the DSL reads well and is unambiguous because it is
always namespace-qualified: `text.filter(...)`, `text.map(...)` (cf. pandas `.filter`).
The only real risk is a user doing `from symai.ops.text import filter, map`, which
silently rebinds the builtins in their module.

**Why it matters.** Low, but worth a documented convention: these are meant to be used
as `text.filter`, never imported bare.

**Proposed change.** Keep the names (the DSL value is real), but document "import the
namespace, not the functions" and add a lint note. No rename needed.

**Feature impact:** keeps-all · **Confidence:** med · **Impact:** low · **Effort:** S.

---

### API-08 — Overlapping "capability/unsupported" error names

**What.** The runtime error tree carries `UnsupportedCapabilityError`,
`EngineCapabilityError`, `UnsupportedFeatureError`, `UnsupportedModelError` — four
names a new user must disambiguate, plus `AmbiguousEngineError`/`UnknownEngineError`
for selection. The base is the verbose `SymbolicAIRuntimeError` while the package is
`symai`.

**Where.** `symai/runtime/errors.py` (whole hierarchy) + the 15 error names in the
root `__all__`.

**Why it matters.** The hierarchy itself is sound (single base, `ExecutionError`
subtree for transport-time failures, decode errors deliberately separate as
`DecodeError(ValueError)`). The issue is purely discoverability of near-synonyms.

**Proposed change.** No structural change. Add a one-screen error table to the
migration/reference docs mapping each error to *when* it fires
(selection-time vs preflight vs transport-time). Optionally alias the base's public
name in docs as `SymaiRuntimeError` for brevity — low priority.

**Feature impact:** keeps-all · **Confidence:** low · **Impact:** low · **Effort:** S.

---

## What is already good (keep it)

- **Op signature uniformity.** Every I/O op is `(runtime, source, …extra…, *, engine=None)`
  and every deterministic op drops both `runtime` and `engine` (`text.template`,
  `embed.similarity/distance/mmd/kernel`). This is exactly the design's §5 contract and
  it makes the surface predictable. The `engine=<name>` keyword is consistent across
  ops, `Function.__call__`, and `Runtime.execute`.
- **Decoder family.** `TextDecoder` / `ConstructorDecoder` / `TypeAdapterDecoder` /
  `PydanticDecoder` share a `*Decoder` suffix, all implement the same `Decoder` protocol,
  and `decode_output(response, decoder, *, output_index, default, limit)` is a single,
  well-named free function. `PydanticDecoder(User)` as sugar over
  `TypeAdapterDecoder(TypeAdapter(User))` is a reasonable convenience.
- **Error-class separation.** `DecodeError` subclasses `ValueError` and lives in
  `decoding.py`, cleanly distinct from the `SymbolicAIRuntimeError` tree — the design's
  "Function and Runtime errors remain distinguishable from decode errors" holds.
- **`MISSING`/`Missing` sentinel.** Exporting both the instance (for the default) and
  the class (for annotations, `default: T | Missing`) is the standard, correct pattern.
- **`ops.*` namespace packaging.** `symai/ops/__init__.py` re-exports the five
  namespaces (`compare, embed, rank, reason, text`) so `from symai.ops import text`
  works — clean and this is the canonical import the design documents.

---

## Proposed minimal public surface (clean major release)

Root `symai/__init__.py`: **empty** (package docstring only). Canonical imports, one
owner per name:

```python
# value + operations
from symai.symbol   import Symbol
from symai.function import Function
from symai.ops      import text, reason, compare, rank, embed

# decoding
from symai.decoding import (
    Decoder, decode_output,
    TextDecoder, ConstructorDecoder, TypeAdapterDecoder, PydanticDecoder,
    DecodeError, MISSING, Missing,
)

# runtime construction / config
from symai.loading         import load_runtime
from symai.runtime.runtime import Runtime
from symai.runtime.config  import RuntimeConfig, EngineSpec   # + BuiltinImplementation (API-04)

# normalized contracts (apps that build requests / read responses)
from symai.runtime.models  import (
    LanguageModelRequest, LanguageModelResponse, EmbeddingRequest, EmbeddingResponse,
    Message, SystemMessage, DeveloperMessage, UserMessage, AssistantMessage,
    TextContent, ImageContent, ResponseFormat, SamplingConfig, ReasoningConfig,
    TokenUsage, ResponseMetadata, FinishReason,   # etc.
)

# error handling
from symai.runtime.errors  import (
    SymbolicAIRuntimeError, RuntimeClosedError, RuntimeOwnershipError,
    UnknownEngineError, AmbiguousEngineError, EngineCapabilityError,
    UnsupportedCapabilityError, UnsupportedModelError, UnsupportedFeatureError,
    ExecutionError, AuthenticationError, RateLimitError, TransportError,
    InvalidResponseError, ErrorMetadata,
)
```

**Removed from any public surface:** `current_runtime`, `NoActiveRuntimeError`
(API-05); the `cast(...)`-requiring bare `ImplementationId` at call sites (API-04).

**The one-import "hello world" a new user writes:**

```python
from symai.symbol import Symbol
from symai.ops import text
from symai.loading import load_runtime
from symai.runtime.config import RuntimeConfig, EngineSpec

config = RuntimeConfig(
    language_models={"main": EngineSpec(
        implementation="openai:responses",
        settings={"model": "gpt-...", "api_key": "..."},
    )},
    default_language_model="main",
)
with load_runtime(config) as runtime:
    summary = text.summarize(runtime, Symbol("A long passage ..."))
    print(summary.value)
```

This is clean once the root facade is removed: each import states its owner, `import
symai` is inert, and there is no split between "root names" and "submodule names."
The blocker is not the design — it is that production has not finished the cutover the
test suite already encodes.
