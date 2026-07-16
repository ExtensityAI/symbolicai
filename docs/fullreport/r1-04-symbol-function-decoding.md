# r1-04 — Value-layer elegance: Symbol / Function / decoding / ops.primitives

Lens: `symai/symbol.py`, `symai/function.py`, `symai/decoding.py`, `symai/ops/primitives.py`.
Assessed against the approved design in `audit/SYMBOL_REDESIGN.md` §§4, 6, 7, 12.

> **Moving-target note.** The tree was edited *during* this audit. `function.py`
> shrank from 110 → 94 lines mid-review and `symai/prompts.py` was deleted while I
> worked. Two seed signals in `_CONTEXT.md` (static/dynamic context on Function;
> Prompt coupling) were **fixed live** and are reported below as positives, not open
> problems. Anchor files `symbol.py` (184 L), `decoding.py` (161 L), `primitives.py`
> (16 L) are unmodified from HEAD `09bab6a`; `function.py` re-read at 94 L.

## Executive summary

The value layer is close to the design and, in places, genuinely elegant: `Symbol`'s
immutability construction, the `_unwrap_operand` shallow-unwrap, the operator table
(matches §4.3), and the decode/default/limit error-isolation are all clean, well-tested,
keep-as-is work. `Function` was just brought fully in line with §6 (context fields and
`Prompt` coupling both removed) — the biggest two seeds are **already fixed**. Remaining
opportunities are concentrated in `decoding.py`: (1) `PydanticDecoder` is provably
redundant with `TypeAdapterDecoder`; (2) `ConstructorDecoder` fuses three unrelated decode
strategies under one name and its container branch has **no production consumer** and a
grammar that conflicts with `TypeAdapterDecoder`; (3) `_normalize_text`'s single-quote
stripping is an asymmetric, universally-applied heuristic that silently mutates faithful
text/JSON output — a real footgun. `primitives.py` validates the two-stage design (§12
still holds) but should absorb the `_symbol_value`/`_require_text` guards now duplicated
across ops. Nothing here is a correctness bug; these are elegance/boundary refinements.

## Findings table

| # | Finding | File / symbol | Feature impact | Conf | Impact | Effort |
|---|---|---|---|---|---|---|
| 1 | `PydanticDecoder` redundant with `TypeAdapterDecoder` (identical result+type) | `decoding.py` `PydanticDecoder` | keeps-all | high | med | S |
| 2 | `ConstructorDecoder` = 3 strategies under one name; container branch unused + grammar clash | `decoding.py` `ConstructorDecoder` | drops-minimal | high | med | M |
| 3 | `_normalize_text` strips wrapping single-quotes for **all** decoders — asymmetric, content-altering | `decoding.py` `_normalize_text` | keeps-all | med | med | S |
| 4 | `_symbol_value`/`_require_text` duplicated across ops; belong in `primitives.py` | `ops/primitives.py` + ops | keeps-all | high | low | S |
| 5 | `execute_many` signature (`Sequence[Sequence[object]]`) diverges from design's flat `Sequence[object]` | `function.py` `execute_many` | keeps-all | high | low | S |
| 6 | `__eq__`/ordering force `bool(...)`; diverges from held-value semantics for array-likes | `symbol.py` comparison dunders | drops-minimal | high | low | — |
| 7 | `Function`'s `_normalize_string_sequence` duplicates `operations._string_tuple` | `function.py` | keeps-all | high | low | S |
| P1 | **FIXED live**: `static_context`/`dynamic_context` removed from Function (seed #2) | `function.py` | — | high | — | — |
| P2 | **FIXED live**: `Prompt` coupling removed; `examples: Sequence[str] \| str \| None` (seed #4, Function part) | `function.py` | — | high | — | — |

---

## Detailed findings

### 1. `PydanticDecoder` is redundant with `TypeAdapterDecoder`

**What.** Two decoders decode a Pydantic model; one is a strict subset of the other.

**Where.** `decoding.py`:

```python
@dataclass(frozen=True, slots=True)
class TypeAdapterDecoder(Generic[T]):
    adapter: TypeAdapter[T]
    def decode(self, text: str, /) -> T:
        return self.adapter.validate_json(_normalize_text(text))

@dataclass(frozen=True, slots=True)
class PydanticDecoder(Generic[ModelT]):
    model: type[ModelT]
    def decode(self, text: str, /) -> ModelT:
        return self.model.model_validate_json(_normalize_text(text))
```

**Why it matters.** For any `BaseModel` subclass the two are functionally identical.
Verified live:

```text
Answer.model_validate_json('{"value": 7}')          -> value=7  (type Answer)
TypeAdapter(Answer).validate_json('{"value": 7}')   -> value=7  (type Answer)
equal: True   same type: True
```

Design §7 enumerates the standard decoders as **Text / Constructor / TypeAdapter** and its
model example uses `TypeAdapterDecoder(TypeAdapter(list[User]))`. `PydanticDecoder` is
undocumented extra surface (it is exported in `__init__.__all__` and pinned by
`test_decoding.py` / `tests/typecheck/function_decoding.py`, but absent from the design).
It exists only as ergonomic sugar for the single-model case
(`PydanticDecoder(Answer)` vs `TypeAdapterDecoder(TypeAdapter(Answer))`).

**Proposed change.** Either drop `PydanticDecoder`, or — better — remove the friction that
motivates it by letting `TypeAdapterDecoder` accept a bare type and wrap it once:

```python
# before
TypeAdapterDecoder(TypeAdapter(Answer))
# after: TypeAdapterDecoder normalizes `type[T] | TypeAdapter[T]` at construction
TypeAdapterDecoder(Answer)
```

That collapses `PydanticDecoder`, removes the `TypeAdapter(...)` boilerplate the design's
own examples carry, and keeps one decoder-per-concept. **Feature impact: keeps-all.**

---

### 2. `ConstructorDecoder` fuses three decode strategies under one name

**What.** A single class branches into three semantically different behaviors, and one
branch is dead in production while clashing with `TypeAdapterDecoder`.

**Where.** `decoding.py` `ConstructorDecoder.decode`:

```python
if self.constructor is str:
    return cast("T", normalized)
if self.constructor is bool:
    return cast("T", _decode_boolean(normalized))
if self.constructor in (list, tuple, set, dict):
    value = ast.literal_eval(normalized)
    if type(value) is not self.constructor:
        ...
    return cast("T", value)
converter = cast("Callable[[str], T]", self.constructor)
return converter(normalized)
```

**Why it matters.** Three unrelated strategies wear one name:

- `int`/`float`/custom callable → **calls the constructor** (`constructor(text)`);
- `bool` → **keyword parse** (`_decode_boolean`);
- `list`/`tuple`/`set`/`dict` → **does not call the constructor** — `ast.literal_eval` +
  exact-type check.

The container branch is problematic on three counts:
1. **No production consumer.** The only `ConstructorDecoder` used by any op is
   `ConstructorDecoder(bool)` in `ops/compare.py`. `int`/containers appear only in
   `test_decoding.py`. (map/rank/etc. decode to `str` via `TextDecoder`, never to a parsed
   collection.)
2. **Grammar clash.** It overlaps `TypeAdapterDecoder` for containers but with a *different*
   input grammar — Python literals (`['a']`, `(1, 2)`, `{1, 2}`) vs pydantic's JSON
   (`["a"]`). Two ways to decode a `list`, accepting different text. Design §7's own example
   routes containers through `TypeAdapter`.
3. **Misleading name.** "Constructor" implies `constructor(text)`, which is exactly what the
   container branch does *not* do.

**Proposed change.** Narrow `ConstructorDecoder` to scalar constructors (`int`, `float`,
custom `Callable[[str], T]`) plus `bool`; delete the `list/tuple/set/dict` branch and route
containers through `TypeAdapterDecoder` (per §7). If Python-literal parsing is a wanted
feature, give it an honest separate name (`LiteralDecoder`) rather than hiding it inside a
type check. **Feature impact: drops-minimal** — the literal container path has no op
consumer and `TypeAdapterDecoder` already covers containers (via JSON).

---

### 3. `_normalize_text` single-quote stripping is a hidden, asymmetric footgun

**What.** A shared helper strips one layer of wrapping **single** quotes from **every**
decoder's input, silently altering faithful output.

**Where.** `decoding.py`:

```python
def _normalize_text(text: str) -> str:
    normalized = text.strip()
    if len(normalized) >= 2 and normalized.startswith("'") and normalized.endswith("'"):
        normalized = normalized[1:-1].strip()
    return normalized
```

Called by `TextDecoder`, `ConstructorDecoder`, `TypeAdapterDecoder`, `PydanticDecoder`.

**Why it matters.**
- **Content-altering for text.** `TextDecoder` is meant to return the model's text, yet
  `_normalize_text("'Twas the night'")` → `Twas the night` — the leading apostrophe of the
  intended `'Twas` is dropped. Any summarize/translate/query result the model wraps in
  quotes, or that legitimately begins and ends with a quote, is mutated.
- **Asymmetric.** Only `'…'` is stripped; `"…"` is preserved
  (`_normalize_text('"value"')` → `"value"` with quotes intact). Inconsistent and surprising.
- **Doesn't help the decoders it runs for.** JSON grammars use double quotes, so stripping
  single quotes before `validate_json` never helps `TypeAdapter`/`Pydantic` and can only
  break input. The heuristic really exists to normalize the ops' `'value'`-style scalar
  echoes for `bool`/scalar decoding — a `ConstructorDecoder` concern that leaked into the
  shared path.

**Proposed change.** Keep `TextDecoder` faithful (whitespace-strip only); move quote-stripping
into the scalar/bool path of `ConstructorDecoder` where the `'value' =>` echo convention
actually originates; do not pre-strip before `validate_json`. **Feature impact: keeps-all**
(bool/scalar still normalized; text and JSON become faithful).

---

### 4. `_symbol_value` / `_require_text` guards duplicated across ops

**What.** Input-validation helpers are copy-pasted in `ops/text.py`, `ops/reason.py`,
`ops/compare.py` (and inlined in `ops/rank.py`), while their natural home — the shared
`ops/primitives.py` that already owns `_execute_language` — carries none of them.

**Where.** `ops/primitives.py` is just:

```python
def _execute_language[T](runtime, function, values, decoder, *, engine) -> Symbol[T]:
    response = function(runtime, *values, engine=engine)
    return Symbol(decode_output(response, decoder))
```

while `def _symbol_value` appears in text/reason/compare and `def _require_text` in
text/reason (verified by grep).

**Why it matters.** These guards are identical and load-bearing (every op calls
`_symbol_value` first). Duplication is the seed-#7 smell; since `primitives.py` is already
the ops-shared helper module, the guards belong beside `_execute_language`.

**Proposed change.** Hoist `_symbol_value` and `_require_text` into `primitives.py`; import
them where used. **Feature impact: keeps-all.** (Minor related note: `_execute_language` is
imported across modules by its underscore name — acceptable for an intra-package shared
helper, but if it becomes the ops' canonical entry point a non-underscore name would read
better.)

---

### 5. `execute_many` diverges from the design signature

**What.** Design §6.1 shows `execute_many(runtime, inputs: Sequence[object], ...)` (flat);
the implementation takes `Sequence[Sequence[object]]` (nested) and splats `*values`.

**Where.** `function.py`:

```python
def execute_many(self, runtime, inputs: Sequence[Sequence[object]], *, engine=None):
    if isinstance(inputs, str): ...
    for values in inputs:
        if isinstance(values, str): ...
    return tuple(self(runtime, *values, engine=engine) for values in inputs)
```

**Why it matters.** The nested form is arguably *better* — it is consistent with
`__call__(self, runtime, *values)` and supports multi-value inputs per call, with careful
str-guards at both levels. But it silently contradicts the documented signature: callers
following the design write `execute_many(rt, ["a", "b"])` and get a `TypeError` ("each input
must be a sequence…"). Either the design or the code should move so the two agree.

**Proposed change.** Keep the nested implementation (it is the more correct one) and update
§6.1 to `inputs: Sequence[Sequence[object]]`. **Feature impact: keeps-all** — this is a
doc/signature reconciliation, not a behavior change.

---

### 6. `__eq__`/ordering force `bool(...)`, diverging from held-value semantics for array-likes

**What.** Comparison dunders coerce the underlying comparison to a scalar `bool`.

**Where.** `symbol.py`:

```python
def __eq__(self, other: object) -> bool:
    return bool(self._value == _unwrap_operand(other))
def __lt__(self, other: object) -> bool:
    return bool(self._value < _unwrap_operand(other))
```

**Why it matters.** §4.3 mandates equality/ordering → `bool`, so this is *intended*. The
edge is that for array-like held values (e.g. numpy), `self._value == other` returns an
array and `bool(array)` raises `ValueError("truth value … is ambiguous")` — where the native
`arr == other` would not. So `Symbol(np.array(...)) == …` diverges from the wrapped value's
own semantics. No current op wraps arrays for comparison (embed ops wrap the numpy result
but never `==` it), so impact is low; worth a one-line docstring/caveat rather than a code
change. **Feature impact: drops-minimal** (array-likes can't be equality-compared through
Symbol — a deliberate consequence of "equality returns bool").

---

### 7. `Function._normalize_string_sequence` duplicates `operations._string_tuple`

**What.** Two near-identical "reject a bare `str`, coerce to tuple" validators in the value
layer.

**Where.** `function.py` `_normalize_string_sequence` (also checks every element is `str`)
vs `operations.py` `_string_tuple` (checks only that the arg isn't a bare `str`).

**Why it matters.** Minor seed-#7 duplication; the two differ slightly (element-type check),
which is itself a smell — the stricter check should be the shared one. Low priority.

**Proposed change.** Consolidate on the stricter validator in one place (e.g.
`operations.py`) and import it into `function.py`. **Feature impact: keeps-all.**

---

## Already-fixed (positives observed live)

### P1 — `static_context` / `dynamic_context` removed from `Function` (was seed #2)

The current `function.py` has **no** context fields and no `_system_prompt`; `request()`
passes `self.prompt` directly:

```python
def request(self, *values: object) -> LanguageModelRequest:
    return language_request(
        self.prompt,
        " ".join(str(value) for value in values),
        examples=self.examples, max_tokens=self.max_tokens, stop=self.stop,
    )
```

This now matches §6 exactly ("There is no framework-level static/dynamic context concept").
Both `static_context`/`dynamic_context` are in `test_public_cutover.FORBIDDEN_IDENTIFIERS`
and are gone from the value layer. **Report status: FIXED.**

### P2 — `Prompt` coupling removed from `Function` (Function part of seed #4)

`examples` is now `Sequence[str] | str | None`; the `from symai.prompts import Prompt`
import and the `Prompt` branch in `_normalize_examples` are gone. `test_components.py::
test_examples_annotation_has_no_prompt_hierarchy` — which was **red** earlier in this audit
(annotation still carried `symai.prompts.Prompt`) — now passes. `symai/prompts.py` itself
was deleted and ops inline their own example tuples (per §8). **Report status: FIXED.**
(Caveat: `test_public_cutover.py` still shows other, non-value-layer cutover items red —
e.g. root `__all__` still populated — but those are outside this lens.)

---

## What to keep (already good)

- **Symbol immutability construction.** `__slots__ = ("_value",)` + `object.__setattr__`
  in `__init__` + raising `__setattr__`/`__delattr__` is the clean, minimal way to get
  shallow immutability with no `__dict__`. Fully covered by `test_symbol.py`. Keep.
- **`_unwrap_operand`.** A three-line shallow unwrap that is correct across *all* operator
  families — equality against raw values, `Symbol op Symbol` arithmetic, `__getitem__` with
  a Symbol key, membership (`Symbol in Symbol`), reflected ops. It deliberately unwraps one
  level only, and native `TypeError`/`KeyError`/`IndexError` propagate unchanged (tested).
  Elegant. Keep.
- **Operator table.** The ~40 dunders match §4.3 precisely: comparisons/membership → `bool`,
  arithmetic/bitwise/unary/index → `Symbol`, `__iter__` → iterator of Symbols, `len`/truth
  → native, `str`/`int`/`float`/`bool` casts → native. 3-arg `__pow__`/`__rpow__` handled.
  `__hash__ = None` (explicit, if technically redundant given `__eq__`). Keep.
- **Decode error isolation.** `decode_output`'s `default` catches only `DecodeError`, never
  transport/selection/programming errors; `_output_text` selects by `output.index` and
  raises `IndexError` when absent; `_limit_value` truncates list/tuple/dict deterministically
  and passes sets through. This is the sharpest part of the layer and exactly matches §7 —
  including the subtle, tested behaviors (default doesn't hide `RuntimeError`; limit/index
  errors bypass `default`). Keep.
- **`primitives._execute_language` validates the two-stage design.** §12 rejected putting a
  decoder on `Function`, predicting "a small composition wrapper … if repetition proves the
  need." That repetition materialized (18 ops route through `_execute_language`), and this
  one 3-line helper is precisely that wrapper: `Function` stays non-generic, `decode_output`
  stays independent, and the ops layer owns the collapse-into-`Symbol`. The two-stage
  friction is real only for direct (non-ops) callers, which is the intended audience for the
  explicit form. **§12's reasoning still holds; no change recommended.**
