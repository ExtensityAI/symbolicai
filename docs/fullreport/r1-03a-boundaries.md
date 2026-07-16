# r1-03a — Layering & Module Boundaries

## Executive summary

1. The **core dependency DAG is exactly right**: `symbol ← ops.* → function → operations/decoding → runtime → providers.engines → providers.client → providers._client`. No import cycles. Nothing below `Symbol` imports `Symbol`; `Function`/`Runtime`/decoders neither import nor return `Symbol`; `ops.*` is the sole layer that wraps decoded values back into `Symbol`. The intended direction **holds in live code.**
2. The **module *inventory* has not converged to the target the test-suite already encodes.** `tests/test_public_cutover.py` is the spec for the end-state module map (root `__init__` empty, `prompts.py` + `backend/` deleted, `current_runtime`/`static_context`/`Prompt` forbidden). Live code violates all of these — several cutover tests are currently red. This is the dominant boundary gap.
3. Two **naming collisions** blur otherwise-clean seams: `symai/operations.py` (request builders) vs `symai/ops/` (Symbol operations); and `symai/loading.py` (builtin registry) vs `symai/runtime/loading.py` (generic loader) — both files named `loading.py`, both exposing `load_runtime`.
4. Small **placement gaps**: shared ops helpers (`_symbol_value`, `_require_text`) are copy-pasted per module instead of living in `ops/primitives.py`; `runtime/models.py` doubles as normalized contracts *and* a base-pydantic-helper grab-bag (`FrozenModel`, `*FiniteFloat`).
5. The provider `_client/` shared seam is genuinely clean (provider-neutral primitives, its own client-error hierarchy, no upward imports) and lazy provider loading keeps `import symai` inert — both worth preserving.

## Findings table

| # | Finding | Where | Feature impact | Conf | Impact | Effort |
|---|---------|-------|----------------|------|--------|--------|
| B1 | Legacy modules still present that the target deletes: `prompts.py` (71 KB), `backend/` (empty vestige) | `symai/prompts.py`, `symai/backend/__init__.py` | keeps-all (target drops them) | high | high | S (backend) / M (prompts) |
| B2 | Root `symai/__init__.py` is a ~90-name re-export facade with `__all__`; target is an empty root | `symai/__init__.py` | keeps-all | high | med | S |
| B3 | Ambient `current_runtime()` / `_CURRENT_RUNTIME` ContextVar reintroduces the implicit global the redesign removes | `runtime/runtime.py`, re-exported by root | drops-real-feature:`ambient runtime discovery` (intended) | high | med | M |
| B4 | Naming collision: `operations.py` (request builders) vs `ops/` (Symbol ops) — opposite ends of the stack, near-identical names | `symai/operations.py` vs `symai/ops/` | keeps-all | high | med | S |
| B5 | Duplicate `loading.py` filename + duplicate `load_runtime` symbol across the mechanism/policy split | `symai/loading.py`, `symai/runtime/loading.py` | keeps-all | high | low | S |
| B6 | Shared ops helpers copy-pasted instead of hoisted into `ops/primitives.py` | `ops/text.py`, `ops/reason.py`, `ops/compare.py` | keeps-all | high | low | S |
| B7 | `runtime/models.py` mixes normalized contracts with base pydantic helpers | `runtime/models.py` (`FrozenModel`, `*FiniteFloat`) | keeps-all | med | low | S |
| B8 | Provider `_client` seam is clean, but the shared/per-provider line is drawn low (transport/headers/_client largely provider-neutral yet duplicated ×3) | `providers/*/client/` vs `providers/_client/` | keeps-all | med | med | M (defer to dup lens) |

---

## Import / dependency map (live code, verified)

```
LAYER (top → bottom)     MODULE                                 → imports (internal)
────────────────────────────────────────────────────────────────────────────────────────
value DSL          symbol.py .................................. (none)          [base]
                        ▲  (imported ONLY by ops.*)
                        │
ergonomic ops      ops/__init__ → {text, reason, rank, compare, embed}
   (owns the       ops/primitives.py → decoding, function, runtime.runtime, symbol
    Symbol-wrap)   ops/{text,reason,rank,compare} → decoding, function, ops.primitives,
                                                     prompts(!), runtime.runtime, symbol
                   ops/embed.py → operations, runtime.runtime, symbol
                        │
exec unit          function.py → operations, prompts(!), runtime.models, runtime.runtime
request builders   operations.py → runtime.models            [misnamed vs ops/]
decoders           decoding.py  → runtime.models
                        │
runtime            runtime/loading.py → config, engines, runtime.runtime
                   runtime/runtime.py → engines, errors, models
                   runtime/config.py  → models
                   runtime/errors.py  → models
                   runtime/engines.py → models
                   runtime/models.py  → (none)   [contracts + base pydantic helpers]  [base]
                        │
composition        loading.py (public) → runtime.{config,engines,loading,runtime}
                                          ⇢ lazy: providers/*/loading
                        │
providers          providers/*/loading.py  → settings, runtime.{engines,errors}
                                              ⇢ lazy: client, engines
                   providers/*/engines/*    → runtime.{errors,models}, own client
                   providers/*/settings.py  → runtime.models
                   providers/*/client/*     → providers._client.{models,errors,headers}
                        │
shared client      providers/_client/{models,errors,headers}.py → (pydantic/stdlib only)  [base]
────────────────────────────────────────────────────────────────────────────────────────
LEGACY / VESTIGE   prompts.py → (none)          [71 KB; DELETED_FILES → target: gone]
                   backend/__init__.py          [empty; DELETED_TREES → target: gone]

(!) = import of a module the cutover spec marks for deletion
```

**Cycle check:** none. Every edge points down or sideways-within-a-layer; `runtime.models` and `symbol.py` and `providers/_client/*` are pure sinks. **Layer invariants all hold:** the only `Symbol` reference outside `symbol.py`/`ops/` is a *string literal* inside a `prompts.py` few-shot example, not an import. `runtime.execute(...) -> LanguageModelResponse | EmbeddingResponse`; `Function.__call__ -> LanguageModelResponse`; `decode_output -> T` (caller's type, never `Symbol`).

---

## Detailed findings

### B1 — Legacy modules still in the tree that the target explicitly deletes

**What.** `tests/test_public_cutover.py` encodes the target module inventory. `prompts.py` is in `DELETED_FILES`; `backend` is in `DELETED_TREES`:

```python
DELETED_FILES = { ... "prompts.py", "backend/async_bridge.py", ... }
DELETED_TREES = { "backend", "extended", "models", "server" }
```

and the tests assert they are gone:

```python
def test_deleted_modules_have_no_import_spec():
    for module_name in ("symai.backend", ..., "symai.prompts", ...):
        assert find_spec(module_name) is None, module_name

def test_deleted_production_tree_and_adapter_inventory():
    for path in DELETED_FILES | DELETED_TREES:
        assert not (PACKAGE / path).exists(), path
```

**Where.** Live tree still has both:
- `symai/prompts.py` — 71 017 bytes; imported by `function.py` (`from symai.prompts import Prompt`) and `ops/{text,reason,rank,compare}.py`.
- `symai/backend/__init__.py` — 0 bytes, empty vestige, imported by nothing in `symai/`.

So `test_deleted_modules_have_no_import_spec`, `test_deleted_production_tree_and_adapter_inventory`, and `test_production_ast_has_no_legacy_graph_references` (which walks the AST and flags any `import symai.prompts` / `symai.backend`) are **currently failing**.

**Why it matters.** `backend/` is pure dead weight — an empty package that only exists to satisfy nothing. `prompts.py` is a live boundary leak: the `ops.*`/`function.py` layer reaches into a 71 KB legacy module that the design has already condemned. As long as ops depends on `prompts`, the "leaf" that the DAG shows as terminal is actually a large legacy blob carrying `PromptRegistry`, jinja2/tomllib/box machinery, and dozens of example classes (content covered by the prompts lens).

**Proposed change (do NOT apply).**
- Delete `symai/backend/` outright — zero consumers, zero cost.
- Extract the handful of few-shot example classes that `ops.*` actually use (`Modify`, `MapExpression`, `Format`, `ReplaceText`, `IncludeText`, `CombineText`, `ExtractPattern`, `ContainsValue`, `FuzzyEquals`, `IsInstanceOf`, `LogicExpression`, `SimpleSymbolicExpression`, `RankList`, `Prompt`) into a small owned module (e.g. `ops/examples.py` or `symai/examples.py`), drop the rest, and delete `prompts.py`. This is what makes the DAG's "leaf" honest.

**Feature impact.** `backend/`: keeps-all. `prompts.py`: keeps-all for the classes ops uses; the legacy `PromptRegistry`/registry machinery is intended to drop (drops-real-feature:`legacy Prompt registry`, already ratified).
**Confidence** high · **Impact** high · **Effort** S (backend) / M (prompts extraction).

---

### B2 — Root `symai/__init__.py` is a re-export facade; the target is an empty root

**What.** The root package re-exports ~90 names and declares `__all__` (142 lines). The cutover test asserts the opposite:

```python
def test_old_root_names_are_absent_after_canonical_imports():
    import symai
    assert not hasattr(symai, "__all__")
    for name in OLD_ROOT_NAMES | FORBIDDEN_PUBLIC_NAMES:   # Function, Runtime, Symbol, current_runtime, ...
        assert not hasattr(symai, name)
```

and `test_import_symai_is_subprocess_isolated_and_inert` asserts `public_names == []` after `import symai`.

**Where.** `symai/__init__.py` — `from symai.decoding import (...)`, `from symai.function import Function`, ... , closing with a 70-entry `__all__ = [...]`.

**Why it matters.** The design states the root "is not a compatibility facade"; canonical imports come from owning modules (`from symai.function import Function`, `from symai.symbol import Symbol`). A fat root re-export (a) contradicts the ratified design and its test, (b) forces the whole public surface to load on `import symai` — the current lazy/inert-import guarantee is only preserved because the re-exported modules themselves avoid heavy deps, a fragile property to lean on, and (c) also violates the project style rule "empty `__init__.py` — no re-exports, no `__all__`" (which the sibling `runtime/`, `providers/`, `_client/` `__init__.py` already follow — they are 0 bytes).

**Proposed change (do NOT apply).** Empty `symai/__init__.py` (or reduce to a version string). Update any docs/examples to import from owning modules. This is a mechanical cutover already fully specified by the test.

**Feature impact.** keeps-all (imports move to canonical modules). · **Confidence** high · **Impact** med · **Effort** S.

---

### B3 — Ambient `current_runtime()` / `_CURRENT_RUNTIME` ContextVar reintroduces the implicit global

**What.** `runtime/runtime.py` still defines a module-global ContextVar, sets it in `__enter__`, resets it in `__exit__`, and exposes `current_runtime()`; the root re-exports both `current_runtime` and `NoActiveRuntimeError`.

**Where.**
```python
# runtime/runtime.py
_CURRENT_RUNTIME: ContextVar[Runtime | None] = ContextVar(...)
def __enter__(self): token = _CURRENT_RUNTIME.set(self); ...
def current_runtime() -> Runtime:
    runtime = _CURRENT_RUNTIME.get()
    ...
    raise NoActiveRuntimeError(msg)
```
The cutover test forbids exactly these:
```python
FORBIDDEN_IDENTIFIERS = { "_CURRENT_RUNTIME", "current_runtime", "NoActiveRuntimeError", ... }
def test_runtime_module_exposes_no_ambient_registry_or_provider_clients():
    for name in ("_CURRENT_RUNTIME", "current_runtime", "NoActiveRuntimeError", "Client", "EngineHandle"):
        assert not hasattr(runtime, name)
```

**Why it matters.** This is a *boundary/coupling* smell, not just an API one: an ambient ContextVar lets any code discover a `Runtime` without being handed one, which is precisely the hidden global coupling the explicit-runtime redesign exists to remove. Its continued presence keeps a back-channel open around the explicit `runtime` parameter threaded through `Function.__call__` / `ops.*`. `SYMBOL_REDESIGN.md` says this was removed; it was not.

**Proposed change (do NOT apply).** Remove `_CURRENT_RUNTIME`, `current_runtime()`, the `__enter__`/`__exit__` set/reset, and `NoActiveRuntimeError`; drop them from the root re-export. `Runtime` stays a plain explicit object passed to `ops`/`Function`. (Coordinate with the runtime-lens agent — this overlaps their surface.)

**Feature impact.** drops-real-feature:`ambient runtime discovery` — but the drop is the ratified intent. · **Confidence** high · **Impact** med · **Effort** M.

*(Related, same root cause: `Function.static_context`/`dynamic_context` — `static_context` is a `FORBIDDEN_IDENTIFIER`, and the design says `Function` carries no static/dynamic context. `function.py` still declares both fields and composes them in `_system_prompt()`. Flagged here for completeness; content owned by the function/ops lens.)*

---

### B4 — Naming collision: `operations.py` (request builders) vs `ops/` (Symbol operations)

**What.** Two top-level names one keystroke apart denote things at *opposite* ends of the stack:
- `symai/operations.py` — low-level, provider-neutral request builders (`language_request`, `image_request`, `embedding_request`, `parse_embedding_response`, `data_uri`); imports only `runtime.models`; consumed by `function.py` and `ops/embed.py`.
- `symai/ops/` — high-level ergonomic Symbol operations (`text`, `reason`, `rank`, `compare`, `embed`) that wrap results back into `Symbol`.

**Where.** `symai/operations.py` header: `from symai.runtime.models import (EmbeddingRequest, LanguageModelRequest, SamplingConfig, ...)`. `symai/ops/__init__.py`: `from symai.ops import compare, embed, rank, reason, text`.

**Why it matters.** "operations" and "ops" are synonyms; a reader cannot infer from the name that one is request plumbing and the other is the Symbol DSL. The *layering* is fine — `operations.py` sits correctly below `Function`, imports only `runtime.models`, and never touches `Symbol` — it is purely a **naming** hazard.

**Proposed change (do NOT apply).** Rename `operations.py` → `requests.py` (it builds `*Request`/response objects), or move it to `runtime/requests.py` since it constructs only `runtime.models` types and is provider-neutral request authoring. Given its consumers are `Function`/`ops` (not `runtime` internals), a top-level `requests.py` is the lighter-touch option; the disambiguating rename matters more than the directory. Prefer `pyseam` for the module rename.

**Feature impact.** keeps-all. · **Confidence** high · **Impact** med · **Effort** S.

---

### B5 — Duplicate `loading.py` filename and duplicate `load_runtime` symbol

**What.** The mechanism/policy split is *correct*, but both halves are named `loading.py` and both export `load_runtime`:
- `runtime/loading.py::load_runtime(config, *, language_model_loaders, embedding_loaders)` — generic, provider-agnostic; preflight + allocation-free validation + failure cleanup.
- `symai/loading.py::load_runtime(config, *, language_model_loaders=(), embedding_loaders=())` — the builtin registry (`BUILTIN_LANGUAGE_MODEL_LOADERS`, `BUILTIN_EMBEDDING_LOADERS`) that prepends built-ins and delegates to the generic one imported `as _load_runtime`.

**Where.** `symai/loading.py`: `from symai.runtime.loading import (..., load_runtime as _load_runtime)`; provider loaders imported **lazily inside** `_load_openai_responses` etc.

**Why it matters.** The split is a genuine strength (see "keep" below) — generic mechanism in the runtime layer with zero provider knowledge, vs. builtin *policy* at the top that knows providers but imports them lazily so `import symai` stays inert (enforced by `tests/test_import_boundaries.py`). The only cost is cognitive: two `loading.py` and two `load_runtime` make stack traces and imports ambiguous at a glance.

**Proposed change (do NOT apply).** Keep the split; disambiguate names. Rename the generic entrypoint `runtime/loading.py::compose_runtime` (or `build_runtime`), and/or rename the public registry file to `symai/registry.py` / `symai/builtins.py`. Low priority polish.

**Feature impact.** keeps-all. · **Confidence** high · **Impact** low · **Effort** S.

---

### B6 — Shared ops helpers copy-pasted instead of hoisted into `ops/primitives.py`

**What.** `ops/primitives.py` is the designated shared-helper home for the ops layer but holds only `_execute_language`. Two helpers are duplicated across siblings instead:
- `_symbol_value[T](symbol, field)` — **3 copies**: `ops/text.py`, `ops/reason.py`, `ops/compare.py`.
- `_require_text(value, field)` — **2 copies**: `ops/reason.py`, `ops/text.py`.

**Where.** e.g. `ops/compare.py::_symbol_value`, `ops/text.py::_symbol_value`, `ops/reason.py::_symbol_value` — identical `def _symbol_value[T](symbol: Symbol[T], field: str) -> T`.

**Why it matters.** Pure placement gap: the shared module exists, the helpers are shared, but they were pasted per file. Three copies drift independently. (Also seed #7.)

**Proposed change (do NOT apply).** Move `_symbol_value` and `_require_text` into `ops/primitives.py` and import them. (Note: `operations.py::_string_tuple` and `function.py::_normalize_string_sequence` are a *similar* pair but live in different layers with slightly different contracts — consolidating those is a smaller, separate call; don't force it.)

**Feature impact.** keeps-all. · **Confidence** high · **Impact** low · **Effort** S.

---

### B7 — `runtime/models.py` mixes normalized contracts with base pydantic helpers

**What.** `runtime/models.py` is the layer's pure sink (imports nothing internal) and holds the normalized message/request/response contracts — but it also hosts generic base helpers used package-wide: `FrozenModel`, `FiniteFloat`, `NonNegativeFiniteFloat`, `PositiveFiniteFloat`, `ProviderId`.

**Where.** These base helpers are imported *as foundation* by `runtime/config.py`, `runtime/errors.py`, and all three `providers/*/settings.py` (`from symai.runtime.models import FrozenModel, PositiveFiniteFloat`). So `runtime.models` doubles as both "the wire contracts" and "the base-model toolkit."

**Why it matters.** Minor grab-bag: provider settings pulling `FrozenModel` from a module named "models" (implying message contracts) is a slightly surprising dependency. It also means anything wanting the base `FrozenModel` transitively depends on the full contract module.

**Proposed change (do NOT apply).** Optionally split the base helpers (`FrozenModel`, the `*FiniteFloat` aliases) into a tiny `runtime/base.py` (or `runtime/pydantic.py`) that `models.py`, `config.py`, `errors.py`, and provider settings import. Low priority — the current arrangement is a DAG sink and causes no cycle.

**Feature impact.** keeps-all. · **Confidence** med · **Impact** low · **Effort** S.

---

### B8 — Provider `_client` seam is clean, but drawn low (scaffolding duplicated ×3)

**What.** The shared/per-provider seam itself is correct (see "keep"). The observation is *where the line sits*: each provider's `client/` re-implements `transport.py`, `headers.py`, `_client.py`, and `errors.py`, and the three look near-identical (seed #5). Only genuinely provider-specific shapes are the endpoint bindings (`responses.py` / `chat.py` / `embeddings.py`).

**Where.** `providers/{openai,cerebras,deepseek}/client/{transport,headers,_client,errors}.py`. Their only shared dependency today is `providers/_client/{models,errors,headers}.py`.

**Why it matters.** From a boundary standpoint the seam is sound and one-directional; the question is whether more provider-neutral scaffolding belongs *above* the seam in `_client/`. Pulling the common transport/`_client` skeleton up would shrink each provider to its endpoint bindings + settings.

**Proposed change (do NOT apply).** Evaluate hoisting the common `transport`/`_client` skeleton into `providers/_client/` (parameterized by provider specifics). **Defer the depth of this to the duplication lens** — flagged here only as a seam-placement note.

**Feature impact.** keeps-all. · **Confidence** med · **Impact** med · **Effort** M.

---

## What's already good — keep

- **The intended layering holds in live code.** Verified by full internal-import grep: the DAG is `symbol ← ops.* → function → operations/decoding → runtime.* → providers.engines → providers.client → providers._client`, acyclic. Nothing below `Symbol` imports `Symbol`; `Function`, `Runtime`, and every decoder neither import nor return `Symbol` (`runtime.execute -> LanguageModelResponse | EmbeddingResponse`; `decode_output -> T`). **`ops.*` is the single Symbol-wrapping layer** — every `Symbol(...)` construction lives in `ops/primitives.py`, `ops/embed.py`, `ops/text.py`. This is the redesign's central invariant and it is intact.
- **Provider engines reach only downward.** `providers/*/engines/*` import exactly `runtime.errors`, `runtime.models`, and their own provider `client` — never `Function`/`ops`/`decoding`/`Symbol`. Clean adapter boundary.
- **The `providers/_client/` seam is genuinely clean.** `_client/models.py` (StrictModel/TolerantModel/ModelId) and `_client/errors.py` (the `ClientError` → `TransportError`/`APIError`/`AuthError`/`RateLimitError` hierarchy) import only pydantic/stdlib and know nothing of `runtime`. The client layer having its **own** error hierarchy, distinct from `runtime/errors.py`, is correct: engines translate raw client errors into runtime errors at the boundary, and the client never learns about the runtime above it.
- **Lazy provider loading keeps `import symai` inert.** `symai/loading.py` imports provider loaders *inside* functions, not at module top; `tests/test_import_boundaries.py` and `test_import_symai_is_subprocess_isolated_and_inert` enforce that loading modules pull in zero heavy provider clients/engines and that `import symai` touches no filesystem/env/network. Preserve this discipline.
- **The mechanism/policy loader split is right** (its only flaw is naming, B5): generic `load_runtime` in the runtime layer with no provider knowledge; builtin registry as top-level policy.
- **Narrow engine protocols.** `runtime/engines.py` `LanguageModelEngine`/`EmbeddingEngine` expose exactly `execute` + `close`, provider-neutral, enforced by `test_runtime_operation_protocols_are_narrow_and_provider_neutral`.
- **Sibling `__init__.py` hygiene.** `runtime/`, `providers/`, `providers/_client/` `__init__.py` are all 0 bytes (per the empty-`__init__` rule) — the root `__init__` (B2) is the sole exception to fix.
- **The test-suite is an executable module-boundary spec.** `test_public_cutover.py` (deleted-file inventory, forbidden-identifier AST walk, forbidden-import-prefix reconstruction) + `test_import_boundaries.py` together nail down the target map. Keep them; drive the code to green against them rather than the reverse.
