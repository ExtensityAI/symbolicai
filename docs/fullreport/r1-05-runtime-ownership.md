# r1-05 — Runtime lifecycle, ownership, and selection

> **Historical snapshot terminology.** References below to `EngineSpec` and configured
> `default_*` fields describe the audited code. The final target uses `EngineConfig`, has no
> configured defaults, and permits unnamed selection only for a sole engine of that capability.

Lens: `runtime/runtime.py`, `runtime/loading.py`, `runtime/engines.py`, `runtime/config.py`, `runtime/errors.py`.
Regime: greenfield / pre-release / breaking cutover — optimize for best end-state, not minimal diff.
Snapshot is a **moving target**: `git status` at audit time showed the live agent editing
`tests/runtime/test_errors.py`, `test_runtime.py`, `test_public_cutover.py`,
`test_symbol_runtime_cutover.py`, and `ops/{compare,rank,reason,text}.py`, but **not** the three
production files this lens flags. Findings re-verified against the live tree; line numbers approximate.

---

## Executive summary

1. **The ambient `ContextVar` is fully vestigial and the migration is half-done.** Production
   `runtime.py` still defines `_CURRENT_RUNTIME`, `current_runtime()`, the `_token` slot, and
   sets/resets the ContextVar in `__enter__`/`__exit__`; `errors.py` still defines
   `NoActiveRuntimeError`; `__init__.py` still has `__all__` and re-exports both. **Zero production
   callers** read any of it. The already-updated test suite *fails* against this code
   (4 cutover tests red, verified live). Removing it is pure simplification, `keeps-all`.
2. **Selection (`_resolve_engine`) and named-instance ownership are the strong core** — the
   named → default → sole → ambiguous → unsupported ladder, the cross-capability
   `EngineCapabilityError`, id()-based instance dedup, allocation-free preflight, and reverse-order
   partial-construction cleanup all match the FIXPLAN contract cleanly. Keep them.
3. **`execute()` releasing the lock before `selected.execute(request)` is correct and good** —
   no lock is held across provider I/O, and ownership enforcement makes concurrent teardown of the
   selected engine unreachable.
4. **The lock/ownership/state triad is right-sized, but the lock is over-*applied*.** Its only
   irreplaceable job is at-most-once teardown during the *pre-entry, no-owner* window; post-entry the
   owner-thread check already guarantees single-threaded access, so the lock in `execute`/`__exit__`
   is redundant (harmless). This should be documented, not expanded.
5. **Validation is duplicated between `Runtime.__init__` and `RuntimeConfig` with *divergent* rules**
   (whitespace/str-type checks differ), and engine names are only *per-capability* unique, contradicting
   the FIXPLAN "globally unique within one Runtime" invariant. Both are real, low-effort consistency fixes.

Overall read: the selection + ownership + loading design is genuinely well-built and matches the spec.
The one load-bearing defect is that the ambient-discovery removal (SYMBOL_REDESIGN §9, FIXPLAN §5)
was applied to the design and tests but **not** to production runtime/errors/root modules.

---

## Findings table

| ID | Finding | Feature impact | Conf | Impact | Effort |
|----|---------|----------------|------|--------|--------|
| R5-1 | Ambient `ContextVar`/`current_runtime`/`NoActiveRuntimeError` vestigial; cutover incomplete in production | keeps-all | high | high | S |
| R5-2 | `_lifecycle_lock` over-applied post-entry (redundant in `execute`/`__exit__`); document its true scope | keeps-all | high | low | S |
| R5-3 | Divergent duplicate validation across `Runtime.__init__` vs `RuntimeConfig` | keeps-all | med | med | S |
| R5-4 | Engine names only per-capability unique, contradicting "globally unique within one Runtime" | drops-minimal | med | low | S |
| R5-5 | (positive) `execute()` drops lock before provider I/O; selection ladder; loading preflight/cleanup | keeps-all | high | — | — |

---

## R5-1 — Ambient `ContextVar` machinery is vestigial; the removal is applied to tests but not production

**What.** SYMBOL_REDESIGN §9 ("`current_runtime()` and ambient `ContextVar` discovery are removed")
and FIXPLAN §5 ("Remove `_CURRENT_RUNTIME` and `current_runtime()`. `with Runtime` controls lifecycle
only") both mandate removal. The test suite already encodes the target state. Production does not.

**Where.** `runtime.py` still carries the whole apparatus:

```python
from contextvars import ContextVar, Token
...
    __slots__ = ( ... "_token")
...
    def __enter__(self) -> Runtime:
        with self._lifecycle_lock:
            ...
            token = _CURRENT_RUNTIME.set(self)
            self._token = token
            self._owner_thread_id = get_ident()
            self._state = _RuntimeState.ACTIVE
            return self

    def __exit__(self, ...):
        with self._lifecycle_lock:
            ...
            _CURRENT_RUNTIME.reset(token)
            self._token = None
        try:
            self.close()
        ...

_CURRENT_RUNTIME: ContextVar[Runtime | None] = ContextVar("symai_active_runtime", default=None)

def current_runtime() -> Runtime:
    runtime = _CURRENT_RUNTIME.get()
    if runtime is None:
        raise NoActiveRuntimeError(msg)
    return runtime
```

`errors.py` still defines `class NoActiveRuntimeError(SymbolicAIRuntimeError)`, and `__init__.py`
still imports/exports `current_runtime` + `NoActiveRuntimeError` and defines `__all__`.

**No production consumer reads any of it.** Every executor takes Runtime explicitly and calls
`runtime.execute(...)` / `function(runtime, ...)` — verified across `ops/{text,reason,compare,rank,embed}.py`,
`ops/primitives.py`, and `function.py`. The only references to `current_runtime`/`_CURRENT_RUNTIME`
outside `runtime.py`/`__init__.py`/`errors.py` are in tests. The set/reset in `__enter__`/`__exit__`
maintains a stack that nothing ever `.get()`s in production. It is dead machinery.

**The already-updated tests fail against this code** (run live at audit time):

```
tests/test_public_cutover.py::test_runtime_module_exposes_no_ambient_registry_or_provider_clients
  AssertionError: assert not True  where True = hasattr(runtime, '_CURRENT_RUNTIME')
tests/test_public_cutover.py::test_old_root_names_are_absent_after_canonical_imports
  AssertionError: assert not True  where True = hasattr(<module 'symai'...>, '__all__')
tests/runtime/test_runtime.py::test_runtime_has_no_ambient_registry_slot_or_module_state
  AssertionError: assert '_token' not in ('_acceptance_order', ... '_lifecycle_lock', ...)
```

The production AST guard (`_production_ast_violations`) flags exactly this lens's identifiers:

```
runtime/runtime.py:299: definition current_runtime
runtime/runtime.py:293: name _CURRENT_RUNTIME
runtime/runtime.py:303: name NoActiveRuntimeError
runtime/errors.py:22:  definition NoActiveRuntimeError
```

Cross-test contradiction confirms the migration is mid-flight: `test_errors.py` was *just* edited to
assert `not hasattr(errors_module, "NoActiveRuntimeError")` (it previously imported the symbol), while
production `errors.py` still defines it. Same for `__all__` (root defines it; `test_old_root_names_...`
asserts its absence).

**Why it matters.** This is the single load-bearing defect in the lens. It is not a style nit: it is
an *incomplete cutover* that leaves a rejected feature (SYMBOL_REDESIGN §12 "Retain ambient Runtime
discovery" — rejected) alive in the shipped surface, keeps a ContextVar stack pushed/popped on every
`with Runtime` for no reader, and keeps `NoActiveRuntimeError` exported as public API. It also blocks
the release gate (public-cutover + AST guard tests are red).

**Proposed change.** Delete the apparatus end-to-end:

- `runtime.py`: drop `from contextvars import ContextVar, Token`, the `_token` slot, the
  `_CURRENT_RUNTIME` global, `current_runtime()`, and the `NoActiveRuntimeError` import. `__enter__`
  collapses to state-check + `_owner_thread_id`/`_state` set; `__exit__` collapses to owner-check +
  `self.close()` (the entire first locked block that resets the token disappears):

  ```python
  def __enter__(self) -> Runtime:
      with self._lifecycle_lock:
          self._require_owner_thread("enter")
          if self._state is not _RuntimeState.CREATED:
              raise RuntimeClosedError("Runtime contexts have a single lifecycle ...")
          self._owner_thread_id = get_ident()
          self._state = _RuntimeState.ACTIVE
          return self

  def __exit__(self, _exc_type, exc_value, _traceback) -> Literal[False]:
      self._require_owner_thread("exit")
      try:
          self.close()
      except BaseExceptionGroup as cleanup_failures:
          if exc_value is None:
              raise
          for failure in cleanup_failures.exceptions:
              exc_value.add_note(f"Runtime cleanup failed: {failure!r}")
      return False
  ```

  Note `__exit__` no longer needs to touch shared state before `close()`, so it needs no lock of its
  own — `close()` already takes the lock for its transition (see R5-2).
- `errors.py`: delete `NoActiveRuntimeError`.
- `__init__.py`: remove both imports/exports (and, per the public-surface lens + `test_old_root_names_...`,
  drop `__all__` entirely).

**Feature impact:** `keeps-all` — ambient discovery is an explicitly rejected feature with no production
consumer. **Confidence:** high. **Impact:** high (release-gate + public API). **Effort:** S.

---

## R5-2 — The lock/ownership/state triad is right-sized, but the lock is over-*applied* post-entry

**What.** The prompt asks whether `_lifecycle_lock` + `_owner_thread_id`/`_require_owner_thread` +
`_state` is over-built for a single-owner-thread contract. Answer: the three mechanisms own *distinct*
concerns and are not mutually redundant, so the triad is right-sized. But the lock is acquired in more
places than it is actually needed.

**Where.** Each mechanism has a separate job:

- `_state` (`CREATED`/`ACTIVE`/`CLOSED`) is *semantic* lifecycle, not concurrency: it enforces
  single-lifecycle (`"cannot be re-entered"`), execute-only-while-active, and idempotent close.
- `_owner_thread_id` + `_require_owner_thread` enforce thread-affinity — the actual contract.
- `_lifecycle_lock` serializes state transitions.

The lock's *only irreplaceable* job is at-most-once teardown during the **pre-entry, no-owner window**.
Before `__enter__`, `_owner_thread_id is None`, so `_require_owner_thread` returns early for *any* thread:

```python
def _require_owner_thread(self, operation) -> None:
    owner_thread_id = self._owner_thread_id
    if owner_thread_id is None or owner_thread_id == get_ident():
        return
    raise RuntimeOwnershipError(operation)
```

So a constructed-but-never-entered Runtime can be `close()`d from arbitrary threads (and FIXPLAN §5
requires close to "work for a constructed Runtime that was never entered"). Two concurrent `close()`
calls would both pass the `_state is CLOSED` guard and both snapshot the *same* `_acceptance_order`
before it is cleared, double-closing every engine — unless the lock serializes the transition:

```python
def close(self) -> None:
    with self._lifecycle_lock:
        self._require_owner_thread("close")
        if self._state is _RuntimeState.CLOSED:
            return
        self._state = _RuntimeState.CLOSED
        engines = tuple(reversed(self._acceptance_order))
        self._acceptance_order = ()
        ...
```

This is load-bearing and correct. But **post-entry**, `_owner_thread_id` is set, so `execute`, `close`,
and `__exit__` are reachable only on the owner thread; a non-owner is rejected *before* touching state.
The lock inside `execute()` and the first block of `__exit__()` therefore guards against nothing
reachable — the owner is single-threaded and sees its own writes by program order.

**Why it matters.** Not a bug — it is a clarity/right-sizing question the reviewer raised. The honest
description is: "the lock exists solely to make pre-entry (no-owner) `enter`/`close` races safe; once an
owner exists, thread-affinity alone provides mutual exclusion." Today that intent is invisible, and a
future reader may either (a) think the lock provides general thread-safety and weaken the ownership
check, or (b) think the lock is pure ceremony and delete it, breaking pre-entry at-most-once teardown.

**Proposed change.** Keep all three mechanisms; do **not** add anything. Two low-risk options:

- *Minimal:* add a one-line comment on `_lifecycle_lock` stating its scope ("serializes the CREATED→ACTIVE
  and →CLOSED transitions during the pre-entry window where no owner thread is established yet; post-entry
  exclusivity comes from `_require_owner_thread`"). Leave the uniform locking for simplicity.
- *Tighter:* stop taking the lock in `execute()` (ownership already guarantees exclusivity there), and
  drop the first locked block from `__exit__` once R5-1 removes the token reset. Retain the lock only in
  `__enter__` and `close()` — the two transitions that can race with no owner. This is the simplest form
  that still keeps single-owner-thread, explicit lifecycle, and at-most-once teardown.

**Feature impact:** `keeps-all`. **Confidence:** high. **Impact:** low. **Effort:** S.

---

## R5-3 — Duplicate validation across `Runtime.__init__` and `RuntimeConfig` uses *divergent* rules

**What.** `_validate_aliases` / `_validate_default` exist in both `Runtime` and `RuntimeConfig`, and the
double-validation is defensible (two independent entry points: direct `Runtime(...)` construction, used
throughout the tests, and `RuntimeConfig` → `load_runtime`). The problem is not that both validate — it
is that they validate *different rules* for what should be one "valid runtime" invariant.

**Where.**

```python
# Runtime._validate_aliases — checks str type + non-empty
if not isinstance(alias, str): raise TypeError(...)
if not alias: raise ValueError(...)

# RuntimeConfig._validate_aliases — checks non-empty + no OUTER WHITESPACE
if not alias: raise ValueError(...)
if alias != alias.strip(): raise ValueError("...must not contain outer whitespace")
```

```python
# Runtime._validate_default — str type + membership
# RuntimeConfig._validate_default — strip check + membership
if not default or default != default.strip(): raise ValueError("...is invalid")
```

Consequences: `Runtime(language_models={" chat ": engine})` is **accepted** (whitespace alias), while
the same alias via `RuntimeConfig` is **rejected**. `Runtime` alone type-checks keys are `str` (needed —
it accepts an arbitrary `Mapping`); `RuntimeConfig` relies on Pydantic for that but adds whitespace
rules `Runtime` lacks. So the two "valid alias" contracts disagree in both directions.

`_validate_engine_identities` (id()-based dedup) correctly lives *only* on `Runtime` — it operates on
live engine instances, which is meaningless for `EngineSpec` config (equal specs are legal, that is the
whole named-instance point). That one is not duplicated and should stay put.

**Why it matters.** Two construction paths that are supposed to yield the same guarantees enforce
different alias syntax. A caller who validates a `RuntimeConfig`, then (post-`load_runtime`) trusts the
resulting `Runtime`, is fine; but a caller who builds a `Runtime` directly gets weaker validation. That
is a silent contract seam.

**Proposed change.** Factor the *shared* alias-syntax + default-membership + at-least-one-engine rules
into module-level free functions in `runtime/config.py` (or a small shared module) keyed on
`Mapping[str, object]`, and call them from both `Runtime.__init__` and `RuntimeConfig`. Unify the rule
set (decide once whether outer whitespace is legal — recommend rejecting it everywhere) and keep the
`str`-type guard on the `Runtime` path (it alone accepts untyped mappings). Keep `_validate_engine_identities`
Runtime-only.

**Feature impact:** `keeps-all`. **Confidence:** med. **Impact:** med. **Effort:** S.

---

## R5-4 — Engine names are only *per-capability* unique, contradicting the "globally unique" invariant

**What.** FIXPLAN §2 states "Names are globally unique within one Runtime." The code enforces uniqueness
only *within* each capability map (dict keys), never *across* them. `_validate_engine_identities` dedups
by `id()` across both maps but not by name; there is no cross-map name check (verified — no intersection
logic in `runtime.py`).

**Where.** `Runtime.__init__` validates `language_snapshot` and `embedding_snapshot` independently, so
`language_models={"x": langA}` + `embeddings={"x": embB}` is accepted. Selection is capability-scoped, so
it resolves without ambiguity — `execute(LanguageModelRequest, engine="x")` → `langA`;
`execute(EmbeddingRequest, engine="x")` → `embB`:

```python
if isinstance(request, LanguageModelRequest):
    selected = self._resolve_engine("language_model", self._language_models, self._embeddings, ...)
elif isinstance(request, EmbeddingRequest):
    selected = self._resolve_engine("embedding", self._embeddings, self._language_models, ...)
```

**Why it matters.** It is functionally safe *today* (request type disambiguates), but it diverges from the
written invariant and makes `engine="x"` mean two different engines depending on request type — surprising,
and a latent hazard if a future request type is capability-ambiguous. It is a spec-vs-code gap that should
be resolved deliberately, in one of two directions.

**Proposed change.** Pick one and make code + FIXPLAN agree:

- *Enforce global uniqueness:* add a cross-map check in `Runtime.__init__` (and the shared validator from
  R5-3) rejecting any name present in both maps. Matches the stated invariant; costs the (rare) ability to
  reuse a name across capabilities.
- *Or* accept capability-scoped names as intended and amend FIXPLAN §2 to say "unique within each
  capability" — since ops already know whether they issue a language or embedding request, scoped names are
  arguably the cleaner model.

Recommend the first (enforce global) to keep the mental model "a name identifies one engine."

**Feature impact:** `drops-minimal` (only same-name-across-capabilities). **Confidence:** med. **Impact:**
low. **Effort:** S.

---

## R5-5 — What is already good and must be kept

- **`execute()` releases the lock before provider I/O.** The pattern selects under the lock, then calls
  `selected.execute(request)` *outside* it:

  ```python
  with self._lifecycle_lock:
      self._require_owner_thread("execute")
      if self._state is not _RuntimeState.ACTIVE: raise RuntimeClosedError(...)
      selected = self._resolve_engine(...)
  return selected.execute(request)
  ```

  Correct and clear: no lock is held across a blocking network call, and because `close()` requires the
  owner thread (which is currently *inside* `execute`), the selected engine cannot be torn down
  concurrently — so releasing early is provably safe. Keep this shape.

- **The selection ladder `_resolve_engine`** (named → default → sole → ambiguous → unsupported) matches
  FIXPLAN §5 exactly, and the cross-capability path raises the structured `EngineCapabilityError` carrying
  both `requested_capability` and `engine_capability` (no credential leakage). `AmbiguousEngineError` lists
  sorted safe names. `UnknownEngineError` carries only `engine_name`. Clean, distinct, safe errors.

- **Named-instance model is cleanly supported.** Two same-model engines with different keys/transports are
  first-class: `RuntimeConfig` holds `Mapping[str, EngineSpec]` (equal `implementation`, different
  `settings` allowed), `loading.py` constructs a fresh engine (hence a fresh client) per alias, and
  `_validate_engine_identities` rejects only accidental reuse of the *same* engine object under two names
  (which would break at-most-once teardown). This directly satisfies the FIXPLAN acceptance criterion.

- **Loading preflight + partial-construction cleanup are exactly right.** `_preflight` indexes loaders and
  validates every config reference resolves *before* any engine/transport is allocated ("allocation-free
  preflight"); the `except BaseException` path closes already-`loaded` engines in reverse order and reports
  cleanup failures via `BaseExceptionGroup` chained onto the original error. Matches "construction resolves
  all configurations before allocating any transport" and "partial construction closes completed resources
  in reverse order."

- **`close()` teardown semantics.** Marks `CLOSED` and detaches maps under the lock, then closes engines
  in reverse acceptance order, aggregating *all* failures into a `BaseExceptionGroup` — idempotent,
  at-most-once, work-for-never-entered. `__exit__` attaches cleanup failures as notes on an in-flight
  exception rather than masking it. This is a faithful implementation of the ownership invariants.

- **Protocols are minimal.** `LanguageModelEngine` / `EmbeddingEngine` expose exactly `execute` + `close`
  (enforced by `test_runtime_operation_protocols_are_narrow_and_provider_neutral`). No handle/client leaks.

---

## Simplest form that preserves the four invariants

Keeping single-owner-thread, explicit lifecycle, named engines, and at-most-once teardown, the minimal
end-state is:

1. **Delete the ambient layer entirely** (R5-1): no `ContextVar`, `_token`, `current_runtime`,
   `NoActiveRuntimeError`, or root `__all__`. `with Runtime` becomes lifecycle-only.
2. **Keep the state machine + owner check; scope the lock to `__enter__` + `close()`** (R5-2) — the two
   transitions that can race before an owner exists — and document that post-entry exclusivity comes from
   thread-affinity, not the lock.
3. **One shared alias/default validator** called from both `Runtime` and `RuntimeConfig`, with a single
   agreed rule set; identity-dedup stays Runtime-only (R5-3).
4. **Decide name scoping once** (R5-4) and make code + spec agree.
5. Keep `_resolve_engine`, the named-instance loading path, preflight, and reverse-order grouped teardown
   unchanged — they are the correct core.
