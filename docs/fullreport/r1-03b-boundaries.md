# r1-03b — Layering & Module Boundaries

**Lens:** import graph, layer direction, module placement, client↔engine seam.
**Scope:** `symai/**` in the `engine-redesign` worktree only. Read-only audit; findings
anchored by symbol + snippet (line numbers approximate — moving target).

---

## Executive summary

1. **The intended clean boundary is already written down as a test — and the live code
   fails 6 of its assertions.** `tests/test_public_cutover.py` encodes the target module
   map (inert root, no `backend/`, no `prompts.py`, no ambient `current_runtime`, no
   `static_context/dynamic_context`). Running it now: **6 failed, 8 passed.** The cutover
   is real but **incomplete**; most boundary debt below is "the last mile of that cutover."
2. **Root `symai/__init__.py` is a 68-name facade** the design explicitly rejects ("root is
   not a compatibility facade"), and it inverts curation — it re-exports wire-contract
   minutiae (`JsonEntry`, `LogitBias`, `MetadataLabel`) while **omitting the two most
   user-facing symbols, `Symbol` and the `ops` namespace.**
3. **`prompts.py` (slated for deletion) is still a live dependency of `Function` and every
   `ops` module** — the single largest remaining boundary knot. `Function` imports it only
   for an `isinstance(Prompt)` branch that internal callers never hit (ops pre-convert via
   `.value`).
4. **The good news is structural and verified:** the graph is **acyclic**, `Symbol` is
   **contained** (referenced only in `symbol.py` + `ops/*`), and the **client↔engine seam
   is clean** — clients import only `_client`, never `runtime`; engines are the sole
   provider→runtime crossing; runtime stays provider-agnostic.
5. **Naming collisions add friction:** `symai/operations.py` vs `symai/ops/` (both "operations",
   both used) and two `loading.py` modules that both export `load_runtime`. The layer *splits*
   are right; only the *names* mislead.

**Overall read:** the boundary *design* is sound and mostly honored; the *implementation* is
mid-flight. Finish the cutover the test already specifies, fix two naming collisions, and add
positive-layering enforcement (only the lazy-loading rule is guarded today).

---

## Findings table

| ID | What | Where | Feature impact | Conf | Impact | Effort |
|----|------|-------|----------------|------|--------|--------|
| B1 | Root `__init__` is a 68-name facade; omits `Symbol`+`ops`; breaks cutover tests | `symai/__init__.py` | keeps-all | high | high | S |
| B2 | `backend/` empty tree — dead vestige, slated for deletion | `symai/backend/__init__.py` | keeps-all | high | med | S |
| B3 | `prompts.py` (to be deleted) still imported by `Function` + all `ops` | `function.py`, `ops/*` | drops-minimal | high | high | M |
| B4 | Name collision: `operations.py` (builders) vs `ops/` (semantic ops) | `symai/operations.py` | keeps-all | high | med | S |
| B5 | Two `loading.py` both export `load_runtime` (generic vs builtin) | `symai/loading.py`, `runtime/loading.py` | keeps-all | high | low | S |
| B6 | Only the lazy-load rule is enforced; positive layering rules are convention-only | `tests/test_import_boundaries.py` | keeps-all | high | med | M |

---

## Detailed findings

### B1 — Root `symai/__init__.py` is a 68-name facade the design rejects, and it omits the highest-level symbols

**What.** `symai/__init__.py` eagerly imports and re-exports **68** names via `__all__`
(decoders, `Function`, `load_runtime`, every error, every `runtime.models` contract, `Runtime`,
`current_runtime`). The ratified design (`audit/SYMBOL_REDESIGN.md`, and enforced by
`test_public_cutover.py`) says the root is **inert** — canonical imports come from owning
modules. Two things compound the smell:
- The facade **inverts curation**: it exports deep wire-contract types (`JsonArray`, `JsonEntry`,
  `JsonObject`, `LogitBias`, `MetadataLabel`, `ReasoningSummary`) but **not** `Symbol` and **not**
  the `ops` namespace — precisely the two surfaces a user reaches for first. Tests already import
  those from their owning modules (`from symai.symbol import Symbol`, `import symai.ops as ops`),
  confirming the "owning module" intent.
- It makes `import symai` **non-inert**: pulling the root eagerly imports `decoding`, `function`,
  `loading`, and all of `runtime.*`.

**Where.** `symai/__init__.py`:
```python
from symai.decoding import (MISSING, ConstructorDecoder, ...)
from symai.function import Function
from symai.loading import load_runtime
from symai.runtime.models import (AssistantMessage, ..., JsonEntry, ..., LogitBias, ...)
from symai.runtime.runtime import Runtime, current_runtime
__all__ = [ ... 68 names ... ]   # Symbol and ops are NOT here
```
The design contract, from `tests/test_public_cutover.py`:
```python
def test_old_root_names_are_absent_after_canonical_imports():
    import symai
    assert not hasattr(symai, "__all__")
    for name in OLD_ROOT_NAMES | FORBIDDEN_PUBLIC_NAMES:   # includes Function, Runtime, Symbol, load_runtime …
        assert not hasattr(symai, name)
# and test_import_symai_is_subprocess_isolated_and_inert expects:
#   "public_names": []   and   "symai_modules": ["symai"]
```
**Verified:** both tests **fail** now (`import symai` exposes 68 names and drags in submodules).

**Why it matters.** This is the top-level module boundary. A partial facade is the worst of both
worlds: users can't rely on the root (Symbol/ops missing) yet the root isn't inert either. It also
defeats the lazy-loading goal — the whole point of the deferred provider imports (B6, and the
`__getattr__` provider facades) is undercut if `import symai` eagerly wires the exec/runtime layers.

**Proposed change (do not apply).** Reconcile toward the ratified design: make
`symai/__init__.py` **empty** (`__init__.py` carries no re-exports, matching the repo's own
"empty `__init__.py`" convention). Consumers import from owning modules
(`from symai.function import Function`, `from symai.symbol import Symbol`, `import symai.ops as ops`,
`from symai.loading import load_runtime`). If a curated facade is genuinely wanted instead, that is a
*design reversal* that must be ratified against `SYMBOL_REDESIGN.md` and the cutover test — and even
then it should surface `Symbol`+`ops` and hide the JSON-AST/contract minutiae, i.e. the opposite of
today's list.

**Feature impact:** keeps-all (import paths move to owning modules; nothing lost).
**Confidence:** high · **Impact:** high · **Effort:** S (delete re-exports; ripple to any root-import callsites).

---

### B2 — `symai/backend/` is an empty vestige already marked for deletion

**What.** `symai/backend/__init__.py` is a 0-byte file and `symai/backend/` is otherwise empty. No
code, test, or config imports `symai.backend` (the only hits are stale references in
`audit/FINDINGS.md`/`CMDS.md` describing the *old* pre-cutover tree). The cutover test lists
`backend` in both `DELETED_FILES` and `DELETED_TREES`.

**Where.** `symai/backend/__init__.py` (empty). Contract:
```python
DELETED_TREES = {"backend", "extended", "models", "server"}
def test_deleted_modules_have_no_import_spec():
    ... assert find_spec("symai.backend") is None ...
```
**Verified:** `test_deleted_modules_have_no_import_spec` and
`test_deleted_production_tree_and_adapter_inventory` **fail** solely because `symai/backend/` still exists.

**Why it matters.** A phantom top-level package invites accidental re-use and contradicts the
"providers, not backend" module map. Pure dead weight on the boundary.

**Proposed change (do not apply).** Delete the `symai/backend/` directory.

**Feature impact:** keeps-all. **Confidence:** high · **Impact:** med · **Effort:** S.

---

### B3 — `prompts.py` (slated for deletion) is still a live dependency of `Function` and every `ops` module

**What.** `prompts.py` (1042 LOC; `Prompt`, `PromptRegistry`, jinja2/tomllib/python-box) is in the
cutover's `DELETED_FILES`, and `Prompt`/`PromptRegistry` are in `FORBIDDEN_IDENTIFIERS`. Yet it is
still imported in two directions:
- **`function.py` (exec layer) → `prompts.Prompt`**, used *only* to special-case an example
  container:
  ```python
  from symai.prompts import Prompt
  ...
  def _normalize_examples(examples):
      if isinstance(examples, Prompt):
          return tuple(examples.value)
  ```
  Internal callers never exercise this branch — `ops/text.py` already pre-converts:
  `_MODIFY_EXAMPLES = tuple(Modify().value)` then `Function(..., examples=_MODIFY_EXAMPLES)`. So the
  exec layer imports a 1042-LOC content module (and its jinja2/box/tomllib deps) for a branch that
  fires only for an external caller passing a raw `Prompt`.
- **`ops/{text,reason,compare,rank}.py` → `prompts`** for few-shot example classes (`Modify`,
  `MapExpression`, `RankList`, `ContainsValue`, `IsInstanceOf`, `LogicExpression`, …).

**Where.** `function.py` head `from symai.prompts import Prompt`; `ops/text.py`
`from symai.prompts import (Format, MapExpression, Modify, ...)`. Contract:
```python
DELETED_FILES = { ... "prompts.py", ... }
FORBIDDEN_IDENTIFIERS = { ... "Prompt", "PromptRegistry", ... }
```
**Verified:** `test_production_ast_has_no_legacy_graph_references` **fails with 75 violations**, the
first being `prompts.py:15: definition Prompt`.

**Why it matters.** This is the largest single boundary knot blocking the cutover. `prompts.py`
cannot be deleted while `Function` and `ops` depend on it, and the exec→content coupling
(`Function` → `prompts`) is a layering inversion regardless of the cutover: request execution should
not depend on a few-shot-example library.

**Proposed change (do not apply).**
1. Drop `Prompt` from `Function`'s accepted example types → signature becomes
   `examples: Sequence[str] | str | None`; `Function` stops importing `prompts` entirely. Callers
   holding a `Prompt` pass `prompt.value` (ops already do the equivalent).
2. Relocate the *surviving* few-shot example content (the `Modify`/`MapExpression`/`RankList`/…
   strings the ops need) into a lightweight home owned by ops — e.g. `symai/ops/_examples.py` as
   plain `tuple[str, ...]` constants — and drop the `Prompt`/`PromptRegistry`/jinja2/tomllib/box
   machinery with `prompts.py`. (Depth of the content slim-down belongs to the prompts-lens; the
   *boundary* requirement is only "give the example content a home that isn't the doomed module,
   and sever `Function`→`prompts`.")

**Feature impact:** drops-minimal — loses the convenience of passing a `Prompt` object directly to
`Function` (replaced by `.value`) and the unused `PromptRegistry`/jinja/box machinery. Named
few-shot content is preserved.
**Confidence:** high · **Impact:** high · **Effort:** M.

---

### B4 — Naming collision: `symai/operations.py` (request builders) vs `symai/ops/` (semantic operations)

**What.** Two sibling surfaces are both "operations": `symai/operations.py` (functions
`language_request`, `image_request`, `embedding_request`, `data_uri`, `parse_embedding_response`)
and the `symai/ops/` package (semantic operations `text`, `embed`, `reason`, `compare`, `rank`).
Both are imported in the codebase and tests (`from symai.operations import language_request` in
`function.py`; `import symai.operations as operations` in `tests/test_operations.py`;
`import symai.ops as ops`). A reader must constantly disambiguate `operations` from `ops`.

**Where.** `symai/operations.py` imports **only** `runtime.models` and **builds** `runtime.models`
request types:
```python
from symai.runtime.models import (EmbeddingRequest, ..., LanguageModelRequest, SamplingConfig, ...)
def language_request(system_prompt, user_prompt, *, examples=(), ...): -> LanguageModelRequest
```

**Why it matters.** The module is a pure, provider-neutral *constructor* for the runtime request
contract, living at top level under a name that shadows the `ops` package. Its dependency footprint
(only `runtime.models`) says it belongs next to the models it builds, not floating at the root with a
colliding name.

**Proposed change (do not apply).** Move/rename `symai/operations.py` →
`symai/runtime/requests.py` (co-locate request *builders* with the request *models*). Consumers
become `from symai.runtime.requests import language_request` (`function.py`) and
`from symai.runtime.requests import embedding_request, parse_embedding_response` (`ops/embed.py`).
This removes the `operations`/`ops` ambiguity and tightens runtime cohesion. (If keeping it at root
is preferred, at minimum rename to `symai/requests.py`.)

**Feature impact:** keeps-all. **Confidence:** high (collision is real) / med (on destination) ·
**Impact:** med · **Effort:** S.

---

### B5 — Two `loading.py` modules both export `load_runtime`

**What.** `symai/runtime/loading.py` defines the *generic* `load_runtime(config, *, language_model_loaders, embedding_loaders)`
(preflight, allocation-free validation, failure cleanup). `symai/loading.py` defines the *public*
`load_runtime(config, ...)` that composes `BUILTIN_*_LOADERS` and delegates to the generic one —
imported as `_load_runtime` to dodge the name clash:
```python
# symai/loading.py
from symai.runtime.loading import (..., load_runtime as _load_runtime)
def load_runtime(config, *, language_model_loaders=(), embedding_loaders=()):
    return _load_runtime(config, language_model_loaders=(*BUILTIN..., *language_model_loaders), ...)
```

**Where.** `symai/loading.py` + `symai/runtime/loading.py`.

**Why it matters.** The **layer split is correct and worth keeping** — the generic mechanism stays
in `runtime/` and knows nothing about providers; the provider-aware builtin registry lives *above*
runtime in `symai/loading.py` (which is exactly why runtime remains provider-agnostic, verified: no
`runtime/*` module imports `providers`). The problem is only nominal: same function name at two
layers, and two modules both literally named `loading.py`, forcing an `as _load_runtime` alias and
extra reader effort.

**Proposed change (do not apply).** Keep the split; disambiguate the names. Rename the generic
mechanism to convey "compose from an explicit registry" — e.g.
`runtime/loading.py::load_runtime` → `build_runtime` (or `load_runtime_from_registry`). Optionally
rename `symai/loading.py` → `symai/builtins.py` to signal "the built-in provider registry +
public entry." The public `load_runtime` name (the documented entry point) stays.

**Feature impact:** keeps-all. **Confidence:** high · **Impact:** low/med · **Effort:** S.

---

### B6 — Only the lazy-loading rule is enforced; the positive layering invariants are convention-only

**What.** The single dedicated boundary test, `tests/test_import_boundaries.py`, checks exactly one
thing: the four `loading` modules do not *eagerly* import heavy provider `client`/`engines` modules.
The architecturally important invariants are unguarded by any positive rule:
- `Symbol` containment (`Function`/`Runtime`/`decoding`/`operations` must not import `symbol`),
- clients never import `runtime`/`engines`,
- no cross-provider imports (`deepseek` ↮ `cerebras` ↮ `openai`),
- acyclicity.

`test_public_cutover.py`'s AST guard covers the *negative* legacy-identifier bans (`Prompt`,
`current_runtime`, `static_context`, …) but not the *positive* direction of the dependency arrows.

**Where.** `tests/test_import_boundaries.py` (only asserts absence of heavy provider modules after
importing a `loading` module).

**Why it matters.** All four invariants above currently hold (see "What's already good"), but they
hold *by discipline*. In a greenfield rebuild that is explicitly re-drawing these seams, the seams
should be executable. This is forward-looking infrastructure, not speculative — the layering is the
product's core claim.

**Proposed change (do not apply).** Add an `import-linter` (or `tach`) contract, or extend the
existing AST test, to encode the layered graph:
`symbol` (no symai deps) ← `ops.*` → `function`/`decoding`/`operations` → `runtime.*` → `providers.*.engines`
→ `providers.*.client` → `providers._client`; with rules "nothing below `ops` imports `symbol`",
"`providers.*.client` imports only `providers._client`", "no `providers.<a>` imports
`providers.<b>`", and "`runtime.*` never imports `providers`".

**Feature impact:** keeps-all. **Confidence:** high · **Impact:** med · **Effort:** M.

---

## What's already good — keep

- **`Symbol` is cleanly contained.** `Symbol` is referenced only in `symai/symbol.py` and
  `symai/ops/*`. `function.py`, `runtime/*`, `decoding.py`, and `operations.py` do **not** import it.
  (The lone `prompts.py` hit is a literal string inside a few-shot example docstring, not an import.)
  This is the load-bearing invariant of the whole design and it holds.
- **`symbol.py` and `runtime/models.py` are true leaves** — they import only stdlib/pydantic, so the
  value DSL and the contract layer never reach upward.
- **The graph is acyclic.** Tarjan SCC over module-level imports across `symai/**` finds **no cycles**
  (no `TYPE_CHECKING`-guarded circular-import papering-over in the core, either — the three
  `TYPE_CHECKING` uses are the provider `__getattr__` lazy facades, the legitimate lazy-API case).
- **Client↔engine seam is honored (the memory's key concern).** `providers/_client/*` and every
  `providers/*/client/*` import only pydantic and `_client` — **never** `symai.runtime`. Engines
  (`providers/*/engines/*`) are the **sole** crossing: they import `runtime.models`/`runtime.errors`
  downward *and* their own `client`. No `client` module imports `runtime`; no cross-provider imports
  exist. Faithful-binding-client / crossing-point-engine is real.
- **Runtime stays provider-agnostic.** No module under `runtime/` imports `providers`. The
  provider-aware builtin registry deliberately lives *above* runtime in `symai/loading.py`, and the
  generic loader in `runtime/loading.py` takes loaders as explicit parameters. Good inversion.
- **Lazy loading is real and enforced.** `symai/loading.py` defers provider imports to function
  bodies; provider `__init__` files expose `Client`/`*Engine` via `__getattr__`; and
  `test_import_boundaries.py` guards that `import symai.loading` does not drag in heavy engines.
- **`providers/_client` as a shared client toolkit** (`StrictModel`/`TolerantModel`/`ModelId`,
  `errors`, `headers`) with per-provider `client/` on top is a clean seam — shared HTTP/model
  plumbing with zero symai-runtime knowledge.

---

## Verified import / dependency map (arrows point in dependency direction, downward)

```
LEAVES (stdlib + pydantic only)
  symbol.py        runtime/models.py        prompts.py(†)        providers/_client/{models,errors,headers}
     │                   │                       │                          │
     │        ┌──────────┼───────────┬───────────┼──────────┐               │  (shared client toolkit,
     │        ▼          ▼           ▼            │          ▼               │   no symai.runtime deps)
     │   decoding.py  operations.py  runtime/errors  runtime/config  runtime/engines
     │        │          │  (‡ name        │            │               │
     │        │          │   collides      └──────┬─────┴───────────────┘
     │        │          │   with ops/)           ▼
     │        │          │                  runtime/runtime.py
     │        │          │                    │            │
     │        │          │                    ▼            ▼
     │        │          │            runtime/loading.py  function.py ◄─── prompts.py(†,exec→content)
     │        │          │             (generic           (imports operations, runtime.models)
     │        │          │              load_runtime ‡‡)
     │        │          │                    │
     │        │          │                    ▼
     │        │          └──────────► symai/loading.py  ──deferred──►  providers/*/loading.py
     │        │            (builtin registry;            (+settings.py → runtime.models)
     │        │             public load_runtime ‡‡)               │
     │        ▼                                                   ▼ deferred
     └────► ops/*  ──► function, decoding, operations,      providers/*/engines/*   ← the ONLY
            (Symbol wrapper layer;   symbol, prompts(†)      │   provider→runtime crossing
             the sole re-wrap point)                         ▼   (imports runtime.models/errors)
                                                       providers/*/client/*  ──►  providers/_client/*
                                                       (faithful API binding; NEVER imports runtime)

  symai/__init__.py  ── eagerly re-exports 68 names from decoding/function/loading/runtime.*
                        (B1: design wants this INERT; Symbol + ops are notably absent)
  symai/backend/     ── empty vestige, no importers (B2)

  (†)  prompts.py + Function→prompts coupling are slated for deletion by test_public_cutover (B3)
  (‡)  operations.py name collides with the ops/ package (B4)
  (‡‡) two load_runtime at two layers — split is good, names collide (B5)
```

No cycles. All non-annotated edges already point strictly downward; the annotated ones (†/‡/‡‡)
are the boundary debts above. Verified live: `pytest tests/test_public_cutover.py
tests/test_import_boundaries.py` → **6 failed, 8 passed** (all 6 failures are B1/B2/B3 plus the
sibling ambient-registry assertion), and Tarjan SCC over module-level imports → **no cycles**.
