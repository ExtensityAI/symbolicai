# r4 — `Symbol[T]` variance & type-ergonomics

**Scope:** `Symbol` generic variance and the pyright errors it causes, plus the two
other known prod pyright errors. Read-only audit against live HEAD `84f703b`,
`pyright 1.1.411`, venv Python 3.14.

## Executive summary

- `Symbol` is declared `class Symbol(Generic[T])` with a module-level
  `T = TypeVar("T")` (legacy style, **invariant**). Because `Symbol[T]` is invariant,
  `Symbol[Sequence[float] | np.ndarray]` is not assignable to `Symbol[object]`, which
  produces **2** of the 7 `uv run pyright symai` errors (both in `ops/embed.py`).
- The invariance is **not required for safety**. `T` appears only in output position
  (`value` property getter) and in the constructor (`__init__(value: T)`, exempt from
  variance rules). There is **no contravariant use of `T`** anywhere — every operator
  takes `object` and returns `Symbol[Any]`; there is no setter and `__setattr__` raises.
  Therefore **covariance is sound.**
- **Recommended fix: migrate `Symbol` to PEP 695 `class Symbol[T]`** (drop the
  module-level `TypeVar` + `Generic`). Pyright then **infers covariance** from usage,
  which clears both `embed.py` errors *and* fixes real caller ergonomics
  (`similarity(Symbol([1.0, 0.0]), …)` currently would fail invariance). It also resolves
  the dormant ruff UP046 finding and the `python.md` "no module-level TypeVar" rule in one
  move. `keeps-all`, effort **S**.
- The union `Sequence[float] | np.ndarray` needs **no separate treatment** — covariance
  alone makes every `Symbol[X]` assignable to `Symbol[object]` and makes `Symbol[list[float]]`
  assignable to `Symbol[Sequence[float] | np.ndarray]`.
- The other 5 pyright errors are **unrelated to `Symbol` variance**: 2 are `Mapping`-key
  invariance in `runtime.py` (`_validate_aliases`), 2 are `execute()` union-dispatch, 1 is
  the intentional `__hash__ = None`. Fixing `Symbol` covariance does not touch them.

## Error inventory — `uv run pyright symai` @ `84f703b` (7 errors, 0 warnings)

| # | Location (symbol) | pyright message (abridged) | Category | `Symbol`-variance? |
|---|---|---|---|---|
| 1 | `ops/embed.py:206` — `_numeric_vector` → `_numeric_array(symbol, field)` | `Symbol[Sequence[float] \| ndarray]` not assignable to `Symbol[object]`; "Type parameter T@Symbol is invariant" | Symbol invariance | **YES** |
| 2 | `ops/embed.py:221` — `_numeric_matrix` → `_numeric_array(symbol, field)` | `Symbol[Sequence[Sequence[float]] \| ndarray]` not assignable to `Symbol[object]`; "T@Symbol is invariant" | Symbol invariance | **YES** |
| 3 | `runtime/runtime.py:60` — `_validate_aliases("language-model", language_snapshot)` | `dict[str, LanguageModelEngine]` not assignable to `Mapping[object, object]`; "_KT@Mapping is invariant" | `Mapping`-key invariance | Variance, **not Symbol** |
| 4 | `runtime/runtime.py:61` — `_validate_aliases("embedding", embedding_snapshot)` | `dict[str, EmbeddingEngine]` not assignable to `Mapping[object, object]` | `Mapping`-key invariance | Variance, **not Symbol** |
| 5 | `runtime/runtime.py:210` — `selected.execute(request)` | `LanguageModelRequest \| EmbeddingRequest` not assignable to param `EmbeddingRequest` | union-dispatch | Separate |
| 6 | `runtime/runtime.py:210` — same call, other overload | `LanguageModelRequest \| EmbeddingRequest` not assignable to param `LanguageModelRequest` | union-dispatch | Separate |
| 7 | `symbol.py:15` — `__hash__ = None` | `None` not assignable to declared type `(self) -> int` | assignment-type | Separate (intentional) |

Only **#1 and #2** are caused by `Symbol[T]` invariance. Every other `Symbol[<union>]`
op (`similarity`, `distance`, `mmd`, `kernel`, and their `_matching_vectors` helper) is
error-free *inside `symai`* only because the internal call chain keeps the element type
uniform — but the **public signatures are still invariant**, so external callers hit the
same wall (see Finding B).

## Detailed findings

### Finding A — `Symbol[T]` is invariant; it is safe to make it covariant

**What.** `Symbol` uses legacy generics:

```python
# symbol.py
from typing import Any, Generic, TypeVar
T = TypeVar("T")                 # invariant by default

class Symbol(Generic[T]):
    __slots__ = ("_value",)
    __hash__ = None
    def __init__(self, value: T) -> None: ...
    @property
    def value(self) -> T: ...
```

A bare `TypeVar` + `Generic[T]` is **invariant**, so pyright rejects
`Symbol[list[float]] → Symbol[object]` and `Symbol[X] → Symbol[Y]` for any `X != Y`.

**Soundness analysis (where does `T` appear?).** Exhaustive scan of `symbol.py`:

- `__init__(self, value: T)` — input, but the **constructor is exempt** from variance
  checks (both PEP 695 inference and explicit `covariant=True` allow `T` in
  `__init__`/`__new__`).
- `value` property getter `-> T` — **output** (covariant position). ✔
- **Every other method** (`__eq__`, `__add__`, `__getitem__`, `__contains__`, all ~40
  dunders) takes `other: object` / `item: object` / `key: object` and returns
  `Symbol[Any]`, `bool`, `int`, `Iterator[Symbol[Any]]`, etc. — **`T` does not appear**.
- No setter, no `__setitem__`/`__delitem__`; `__setattr__`/`__delattr__` raise;
  `__slots__ = ("_value",)`; the design (`SYMBOL_REDESIGN.md` §4.2) guarantees "no value
  setter … survives" and "Symbol creates no mutable internal state".

Because `T` never appears in a **contravariant (parameter, non-constructor)** position and
the wrapper is shallow-immutable, **covariance is type-safe** — the classic covariant-
mutable-container unsoundness (`list`-style) cannot arise here.

**Why it matters.** Invariance is a pure liability for this type: it blocks the natural
"a `Symbol` of a subtype is a `Symbol` of the supertype" substitution that every read-only
wrapper wants, forcing awkward `Symbol[object]` params or `# type: ignore` at every
callsite that mixes concrete element types.

**Confidence** high · **Impact** high · **Effort** S.

### Finding B — the embed public signatures already break real callers (not just the internal helper)

**What.** The reported errors #1/#2 are on the *internal* `_numeric_array(symbol: Symbol[object])`
call. But the **public** signatures are equally invariant:

```python
def similarity(left: Symbol[Sequence[float] | np.ndarray],
               right: Symbol[Sequence[float] | np.ndarray], ...) -> Symbol[float]: ...
```

**Where.** `tests/test_symbol_runtime_cutover.py::test_similarity_metrics`:

```python
left = Symbol([1.0, 0.0])          # inferred Symbol[list[float]]
right = Symbol([0.0, 2.0])
result = embed.similarity(left, right, metric=metric)   # list[float] ⊄ Sequence[float] | np.ndarray  (invariant)
```

`uv run pyright symai` does not flag this because it is scoped to the package and does not
check `tests/`, but the callsite is a genuine invariance failure that any real user writing
`similarity(Symbol([...]), Symbol([...]))` hits. This is the ergonomics wall the task is
about — it is wider than the two reported package errors.

**Why it matters.** The `Symbol[<union>]` element types on `embed` are documentation the
type checker cannot honor while `Symbol` is invariant. Covariance is what makes them usable.

**Confidence** high · **Impact** high · **Effort** — covered by Finding A's fix.

### Options & trade-offs

| Option | Effect | Fixes #1/#2? | Fixes caller ergonomics (Finding B)? | Repo-standard alignment | Feature impact |
|---|---|---|---|---|---|
| **(a) PEP 695 `class Symbol[T]`** (drop module TypeVar) | pyright **infers** covariance from output-only usage | ✔ | ✔ | Matches `python.md` ("PEP 695 generics; no module-level TypeVar") and clears the dormant ruff UP046; ends the `Generic[T]`-vs-`def f[T]` mixing | `keeps-all` |
| (b) Explicit legacy covariance: `T_co = TypeVar("T_co", covariant=True)`; `class Symbol(Generic[T_co])` | same type effect via explicit variance | ✔ | ✔ | Keeps legacy style (precedent exists — `decoding.py:11`), but **perpetuates** the mixing and would be flagged by UP046 once ruff `target-version` is corrected to `py312` | `keeps-all` |
| (c) Relax embed internal helpers to `Symbol[object]` | only the internal call chain becomes uniform | ✔ | ✘ (public sigs still invariant → callers still break) | neutral | `keeps-all` but treats the symptom |
| (d) `@overload` / accept `Symbol[object]` in public sigs + runtime-validate | drops static element typing; runtime check already exists (`_numeric_array` validates dtype/shape) | ✔ | ✔ (by widening) | loses the `Sequence[float] | np.ndarray` documentation the sigs express | `drops-minimal` (static element hints) |
| (e) Leave invariant + document | no code change | ✘ | ✘ | — | rejected |

**Recommended: (a).** Replace

```python
from typing import Any, Generic, TypeVar
T = TypeVar("T")
class Symbol(Generic[T]):
```

with

```python
from typing import Any
class Symbol[T]:
```

(no other body change). PEP 695 variance is **inferred**: since `T` is used only in an
output position (+ the exempt constructor), pyright infers `Symbol` **covariant**, which:

1. clears `embed.py` errors #1 and #2;
2. fixes the caller ergonomics in Finding B (`Symbol[list[float]] → Symbol[Sequence[float] | np.ndarray]`);
3. removes the module-level `TypeVar` (a `python.md` violation) and the legacy
   `Generic[T]` subclass (ruff UP046 — currently dormant, see note below);
4. leaves `T_co` fallback (option b) available if the team wants to defer the full PEP 695
   migration and stay consistent with `decoding.py`'s explicit-covariant `Decoder[T_co]`.

**Stability of the inferred variance:** if a future edit ever added `T` in a real parameter
position (e.g. a setter), pyright would auto-flip the inference to invariant and surface the
regression — which is the desired safety behavior, since the design forbids mutation.

### Does the fix clear the embed errors, or does the union need its own treatment?

**Covariance alone fully clears #1 and #2.** With `Symbol` covariant, `X <: object` for all
`X`, so `Symbol[Sequence[float] | np.ndarray] <: Symbol[object]` and
`Symbol[Sequence[Sequence[float]] | np.ndarray] <: Symbol[object]`. The union
`Sequence[float] | np.ndarray` needs **no separate handling** — it is not the cause; the
invariance of the outer `Symbol` was. (For Finding B callers, covariance also gives
`list[float] <: Sequence[float] <: Sequence[float] | np.ndarray`, so `Symbol[list[float]]`
flows in cleanly.)

### The other 5 errors — not `Symbol`-variance, not fixed by covariance

- **#3 / #4 (`runtime.py:60‑61`, `_validate_aliases`)** — these ARE a variance issue, but
  of `Mapping`, not `Symbol`. `Mapping` is invariant in its **key** type, so
  `dict[str, LanguageModelEngine]` is not assignable to the param
  `engines: Mapping[object, object]`. **Fix:** type the key as `str`, not `object`:
  `engines: Mapping[str, object]` (Mapping's value type is covariant, so `object` is fine
  for the value). Independent of the `Symbol` change.
- **#5 / #6 (`runtime.py:210`, `execute()` union-dispatch)** — `selected: LanguageModelEngine
  | EmbeddingEngine` receiving `request: LanguageModelRequest | EmbeddingRequest`. pyright
  cannot prove the narrowed engine matches the narrowed request across the two `isinstance`
  branches, because the `selected.execute(...)` call happens **after** the `if/elif` block,
  where both are re-widened to their unions. **Fix:** perform the dispatch *inside* each
  branch (call `.execute` where both the engine and the request are already narrowed), or
  add an internal typed dispatch. Not variance-related.
- **#7 (`symbol.py:15`, `__hash__ = None`)** — intentional per `SYMBOL_REDESIGN.md` §4.2
  ("Symbol is unhashable … because wrapped mutable values remain legal"). pyright flags it
  because `object.__hash__` is declared `(self) -> int`. **Fix:** a targeted
  `__hash__ = None  # pyright: ignore[reportAssignmentType]`, or declare
  `__hash__: ClassVar[None] = None`. Not variance-related; the covariance change leaves it.

## Cross-cutting note — PEP 695 mixing is real but the ruff rule is currently masked

- The repo already uses PEP 695 **function** syntax in 26 places (`ops/*.py` `def equals[LeftT, RightT]`,
  `runtime.py` `_resolve_engine[EngineT]`), which requires Python **3.12+** to even parse.
  Yet `pyproject.toml` sets `requires-python = ">=3.11"` and `ruff.toml` sets
  `target-version = "py311"`. On 3.11 the code is a `SyntaxError`, so the effective floor is
  already 3.12 — migrating `Symbol` to `class Symbol[T]` adds **no new runtime constraint**.
- Because `ruff target-version = "py311"`, UP046/UP047 are **gated off** — `ruff check
  symai/symbol.py --select UP046,UP047` reports "All checks passed!" today. The finding is
  **dormant, not absent**: bumping `target-version` to `py312` (which the code already
  requires) would flag `symbol.py`, `decoding.py`, and the three provider `transport.py`
  `Generic[T]` classes. Option (a) pre-empts that for `Symbol`.

## What is already good and should be kept

- **The shallow-immutability discipline is exactly what makes covariance safe** — `__slots__`,
  raising `__setattr__`/`__delattr__`, read-only `value` property, operators over `object`
  returning fresh `Symbol[Any]`. Keep this contract; it is the soundness guarantee, not
  incidental.
- **`decoding.py` already models covariance correctly** (`Decoder(Protocol[T_co])` with
  `T_co = TypeVar("T_co", covariant=True)`) — there is a working in-repo precedent for a
  covariant read-only surface; `Symbol` should join it (ideally via PEP 695 rather than
  re-introducing another legacy TypeVar).
- **`embed`'s runtime validation is thorough** (`_numeric_array` checks dtype kind, finiteness,
  shape) — so widening the static element type via covariance loses no safety; the runtime
  guard already enforces the numeric contract regardless of the declared element type.
