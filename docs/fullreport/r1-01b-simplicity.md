# r1-01b — Accidental complexity / over-engineering / YAGNI

Lens: can this be **simpler** while keeping features? Hunt abstractions that don't
earn their keep. Snapshot near `09bab6a`; tree is a moving target, findings anchored by
symbol + quoted snippet.

## Executive summary

1. **Two genuinely over-built abstractions survive the ratified cutover and are the highest-value simplifications:** the bespoke `JsonObject`/`JsonArray`/`JsonEntry` Pydantic "AST" (a `dict → parse → to_builtin → dict` round-trip that ends in a `cast` to Pydantic's own `JsonValue`), and the `LanguageModelSpec` **dead capability fields** (`message_roles`, `content_types`, `response_formats`, `context_tokens`) whose sole reason to exist is to be populated — never read — dragging three whole StrEnums (`MessageRole`, `ContentType`, `ResponseFormatType`) with them.
2. The **largest** complexity sinks (ambient `current_runtime`/ContextVar, `Function.static_context`/`dynamic_context`, the entire 1042-LOC `prompts.py` with jinja2 + dead `python-box` dep) are **already ratified for deletion** by both `SYMBOL_REDESIGN.md` and machine-checked cutover tests (`test_public_cutover.py`) — the live code simply hasn't caught up. These are STILL-OPEN but likely owned by the active editor; I flag them as verification items, not new proposals.
3. Real but smaller de-dup: per-provider `settings.py` (4 identical model bodies), the `load_*` functions, and the ops-module helpers (`_symbol_value`/`_require_text` copied across 4 files).
4. `decode_output`'s `default`/`limit`/`output_index` + `_limit_value`, and `Function.execute_many`, are **spec'd** but exercised only by tests — low-priority "does it earn keep yet" items, not clear removals.
5. **Verified NOT over-built (keep):** the ~40 `Symbol` operator dunders realize the spec's §4.3 native-operator contract completely and uniformly — explicit is correct here; the discriminated-union message/response models and `FrozenModel` strictness are load-bearing.

## Findings table

| ID | Finding | Feature impact | Conf | Impact | Effort |
|----|---------|----------------|------|--------|--------|
| S1 | `JsonObject`/`JsonArray`/`JsonEntry` AST is a validated round-trip back to builtin `dict` | keeps-all | high | high | M |
| S2 | `LanguageModelSpec` dead fields + orphan `MessageRole`/`ContentType`/`ResponseFormatType` StrEnums | keeps-all | high | high | M |
| S3 | Ratified-but-still-live legacy: `current_runtime`/ContextVar/`_token`, `static_context`/`dynamic_context`, `prompts.py`+jinja2+python-box | keeps-all | high | high | L |
| S4 | Per-provider boilerplate: `settings.py` ×4 identical, `load_*` funcs, redundant model pre-check | keeps-all | high | med | S–M |
| S5 | Duplicated ops helpers `_symbol_value`/`_require_text` across text/reason/compare/rank | keeps-all | high | low | S |
| S6 | `decode_output` `default`/`limit`/`output_index` + `_limit_value`; `Function.execute_many` — spec'd, test-only | drops-minimal if cut | high | low | S |
| S7 | `Runtime` takes `_lifecycle_lock` on every `execute()` for a single-owner-thread object | keeps-all | med | low | S |

---

## Detailed findings

### S1 — The `JsonObject`/`JsonArray`/`JsonEntry` AST is a validated round-trip to nowhere

**What.** `runtime/models.py` defines a bespoke frozen-Pydantic JSON "AST" (~75 LOC:
`JsonScalar`, `JsonEntry`, `JsonArray`, `JsonObject`, `_parse_json_value`,
`_json_value_to_builtin`, plus three `model_rebuild(...)` calls). Its **only** production
consumer is `JsonSchemaResponseFormat.json_schema: JsonObject`. Every engine that touches
it immediately converts it straight back to a builtin dict and casts to Pydantic's own
`JsonValue`.

**Where.**
- `models.py`:
  ```python
  class JsonSchemaResponseFormat(FrozenModel):
      ...
      json_schema: JsonObject
  ```
  and `JsonObject.parse` / `.to_builtin`:
  ```python
  @classmethod
  def parse(cls, mapping: Mapping[str, object]) -> JsonObject: ...
  def to_builtin(self) -> dict[str, object]: ...
  ```
- Consumers (openai `responses.py`, cerebras `chat_completions.py`):
  ```python
  schema=cast("JsonValue", response_format.json_schema.to_builtin())
  ```
  where `JsonValue` is `from pydantic import JsonValue` — i.e. the endpoint already accepts
  builtin JSON. Callers must first wrap: `JsonObject.parse({...})` (see every engine test).

**Why it matters.** The lifecycle is `dict → JsonObject.parse → (stored) → to_builtin → dict
→ cast to pydantic JsonValue`. The AST's claimed value-adds mostly evaporate:
- **Unique-key validation** (`validate_unique_keys`) can never fire from `.parse()` — a
  `Mapping` already de-dups keys. It only guards direct `JsonObject(entries=(...))`
  construction, which only tests do (`tests/runtime/test_models.py:99-104`).
- **Deep immutability** is discarded at the first engine boundary (`to_builtin()` yields a
  mutable dict), so it protects nothing downstream.
- **JSON-serializability validation** is exactly what Pydantic's built-in `JsonValue`
  already provides — and that's the type the client side uses anyway.

Net: ~75 LOC + 3 public exports (`JsonObject`/`JsonArray`/`JsonEntry`) + a mandatory
`JsonObject.parse(...)` ergonomic tax on every caller, to end up back at a plain dict.

**Proposed change (do not apply).** Type the field with a validated JSON mapping instead of
the AST. Minimal sketch: `json_schema: Mapping[str, JsonValue]` (Pydantic `JsonValue`),
optionally with a `@field_validator` that coerces to an immutable mapping and rejects
non-object tops. Delete `JsonEntry`/`JsonArray`/`JsonObject`/`_parse_json_value`/
`_json_value_to_builtin` and the `model_rebuild` dance; callers pass a plain dict; engines
pass `response_format.json_schema` (already builtin-shaped) straight through. Drop the three
`__init__` re-exports.

**Feature impact:** `keeps-all` (schema still validated as JSON-serializable & object-typed;
the only thing lost is deep-immutability that was already thrown away at the boundary).
**Confidence:** high. **Impact:** high. **Effort:** M.

---

### S2 — `LanguageModelSpec` carries dead fields, and three StrEnums exist only to feed them

**What.** Of the 11 fields on `LanguageModelSpec`, four are **populated by every engine but
never read anywhere** for enforcement: `message_roles`, `content_types`, `response_formats`,
`context_tokens` (and `EmbeddingModelSpec.context_tokens` likewise). The StrEnums
`MessageRole`, `ContentType`, `ResponseFormatType` are used **only** to populate those dead
fields — the actual message/content/format models discriminate on `Literal[...]`, not these
enums. Capability enforcement is done by parallel **hardcoded** checks in each engine using
the *other* spec fields.

**Where.** Verified by exhaustive grep over `symai/` (excluding tests):
- `.message_roles` — **no reads.** Populated via `_ALL_MESSAGE_ROLES = tuple(MessageRole)`
  (openai/cerebras) / explicit tuple (deepseek).
- `.content_types` — **no reads.** Populated, and itself derived from `vision`:
  ```python
  content_types=(ContentType.TEXT, ContentType.IMAGE) if spec.vision else (ContentType.TEXT,)
  ```
  The engine that checks images uses `self.model_spec.vision`, never `content_types`.
- `.response_formats` — **no reads.** Populated via `_ALL_RESPONSE_FORMATS = tuple(ResponseFormatType)`.
- `.context_tokens` — **no reads** on the normalized spec (the `spec.context_tokens` hits are
  reads of the *provider's raw* spec while populating). Only `response_tokens` is enforced.
- Fields that ARE read (keep): `response_tokens`, `reasoning_fields`, `reasoning_efforts`,
  `reasoning_summaries`, `reasoning_formats`, `sampling_fields`, `vision` — e.g.
  `SamplingField.TEMPERATURE not in self.model_spec.sampling_fields` in `responses.py`.

**Why it matters.** Three StrEnums, four spec fields, `Field(min_length=1)` constraints, and
population code duplicated across all four `_normalized_model_spec` functions exist purely to
be written and never consulted. It reads as a "capability matrix" but the matrix isn't the
enforcement mechanism — the hardcoded per-engine `_validate_request` is. That's a maintenance
trap: the dead fields imply a contract the code doesn't actually honor.

**Proposed change (do not apply).** Drop `message_roles`, `content_types`, `response_formats`,
and `context_tokens` from `LanguageModelSpec` (and `context_tokens` from `EmbeddingModelSpec`);
delete the `MessageRole`, `ContentType`, `ResponseFormatType` StrEnums and their `_ALL_*`
module constants; remove the corresponding lines from all four `_normalized_model_spec`
builders and their tests. Keep the read fields exactly as-is. If a declarative capability
matrix is genuinely wanted later, build it *as the enforcement path* (one shared checker
reading the spec) rather than as parallel populated-but-ignored data.

**Feature impact:** `keeps-all` (no validation behavior changes — the enforcing checks don't
use these fields). **Confidence:** high. **Impact:** high. **Effort:** M (touches 4 engines +
tests). **Caveat:** if a future observability/introspection feature is meant to surface a
model's declared roles/formats, that's a real (currently-absent) feature — but today nothing
reads them, so this is YAGNI.

---

### S3 — Ratified-for-deletion legacy still live (verification item, likely owned by active editor)

**What.** The biggest raw-LOC complexity is legacy that both the design doc **and** the
machine-checked cutover tests already declare must be gone. The live tree lags.

**Where / evidence of ratification.**
- **Ambient runtime discovery.** `runtime/runtime.py` still defines `_CURRENT_RUNTIME`
  (`ContextVar`), `current_runtime()`, the `_token` slot, sets/reset the ContextVar in
  `__enter__`/`__exit__`, and `errors.py` keeps `NoActiveRuntimeError`; `__init__.py` exports
  both. But `SYMBOL_REDESIGN.md` §"All Function and semantic operation execution receives
  Runtime explicitly. `current_runtime()` and ambient `ContextVar` discovery are removed."
  and `tests/runtime/test_runtime.py::test_runtime_has_no_ambient_registry_slot_or_module_state`
  asserts `"_token" not in Runtime.__slots__` and `not hasattr(runtime_module, "current_runtime")`.
  `test_public_cutover.py` lists `_CURRENT_RUNTIME`, `current_runtime`, `NoActiveRuntimeError`
  in `FORBIDDEN_IDENTIFIERS` (walked over the whole production AST). → These tests are
  currently **failing** against live code.
- **Function context.** `function.py` still has `static_context`/`dynamic_context` fields and
  `_system_prompt()` composing `<STATIC_CONTEXT/>`/`<DYNAMIC_CONTEXT/>`. `static_context` is a
  `FORBIDDEN_IDENTIFIER`.
- **prompts.py (1042 LOC).** `test_deleted_modules_have_no_import_spec` asserts
  `find_spec("symai.prompts") is None`, and `Prompt`/`PromptRegistry` are `FORBIDDEN_IDENTIFIERS`.
  Yet `prompts.py` still defines `PromptRegistry` (jinja2 `Environment` + `tomllib`), the
  `Prompt` base, and **34** `Prompt` subclasses of which ops import only ~13. `PromptRegistry`
  is never instantiated anywhere; `jinja2`/`tomllib` are used only inside it; `python-box`
  (`python-box>=7.1.1` in `pyproject.toml`) has **zero** references anywhere in the tree — a
  dead declared dependency, and `jinja2` becomes dead once `PromptRegistry` goes.

**Why it matters.** This is the single largest simplification available, but it is not a new
proposal — it's ratified and in progress. The relevant simplicity note is the **end-state
shape** for the ~13 few-shot example sets ops actually use (`Modify`, `MapExpression`,
`Format`, `ReplaceText`, `IncludeText`, `CombineText`, `ExtractPattern`, `FuzzyEquals`,
`ContainsValue`, `IsInstanceOf`, `RankList`, `LogicExpression`, `SimpleSymbolicExpression`):
since `Prompt` itself is forbidden, these should land as plain module-level tuples of strings
next to (or inside) the ops that consume them, not as a class hierarchy. That removes the
`Prompt` base, `PromptRegistry`, jinja2, python-box, and ~21 unused example classes in one
move.

**Proposed change (do not apply).** Complete the ratified deletion; re-home the ~13 live
few-shot sets as constants; drop `jinja2` and `python-box` from `pyproject.toml`. Remove
`static_context`/`dynamic_context` from `Function` (and the `<STATIC_CONTEXT/>` composition);
remove `current_runtime`/`_CURRENT_RUNTIME`/`_token`/`NoActiveRuntimeError` and their exports.

**Feature impact:** `keeps-all` (design explicitly rejects ambient discovery and wrapped
context as anti-features; unused prompt classes carry no live behavior).
**Confidence:** high. **Impact:** high. **Effort:** L. **Status:** STILL-OPEN, but the failing
cutover tests indicate this is actively being done — treat as a verify-on-completion item.

---

### S4 — Per-provider boilerplate: identical `settings.py`, near-identical `load_*`, redundant model pre-check

**What.** Three `settings.py` files hold **four structurally identical** Pydantic models, and
the `load_*` loader functions are near-verbatim copies differing only in the settings/client/
engine/specs they reference — plus a model-existence check that duplicates what the engine
constructor already enforces.

**Where.**
- `providers/{deepseek,cerebras}/settings.py::ChatCompletionsSettings` and
  `providers/openai/settings.py::{ResponsesSettings, EmbeddingSettings}` are all:
  ```python
  class X(FrozenModel):
      api_key: SecretStr = Field(min_length=1)
      model: str = Field(min_length=1)
      request_timeout: PositiveFiniteFloat = 600.0
      connect_timeout: PositiveFiniteFloat = 10.0
      connect_retries: int = Field(default=0, ge=0)
  ```
- `providers/openai/loading.py::load_responses` / `load_embedding`,
  `providers/{cerebras,deepseek}/loading.py::load_chat_completions` share the same body:
  parse settings → lazy-import httpx+client+engine → `if parsed.model not in MODEL_SPECS: raise
  UnsupportedModelError` → build `Client(... httpx.Timeout ...)` → return engine. The
  `MODEL_SPECS` pre-check duplicates the engine's own `MODEL_SPECS[model]` KeyError→
  `UnsupportedModelError` (the engine `__init__` even closes the client on failure), so the
  pre-check is redundant except as a micro-optimization avoiding client construction.

**Why it matters.** A shared `_client` layer already exists (`_client/models.py`), so the
settings duplication is gratuitous. Each new provider currently re-pastes both a settings body
and a loader body.

**Proposed change (do not apply).** Hoist one `ProviderSettings(FrozenModel)` into
`providers/_client/` and have all three providers alias/reuse it (they're identical). Factor a
single `build_engine(settings_cls, client_factory, engine_factory, *, label)` helper (or a
small `load_engine` that takes the parsed settings + a `client → engine` callable), and drop
the redundant `MODEL_SPECS` pre-check in favor of the engine constructor's existing validation.

**Feature impact:** `keeps-all`. **Confidence:** high. **Impact:** med. **Effort:** S–M.
(Note: `test_deleted_production_tree_and_adapter_inventory` pins the `client/`+`engines/` file
inventories but does **not** constrain `settings.py`/`loading.py`, so this is unblocked.)

---

### S5 — Duplicated ops helpers `_symbol_value` / `_require_text`

**What.** `_symbol_value` is defined identically in `ops/text.py`, `ops/reason.py`,
`ops/compare.py` (and inlined in `ops/rank.py` / `ops/compare.py::is_instance_of`);
`_require_text` is duplicated in `ops/text.py` and `ops/reason.py` (and inlined elsewhere).

**Where.**
```python
def _symbol_value[T](symbol: Symbol[T], field: str) -> T:
    if not isinstance(symbol, Symbol):
        msg = f"{field} must be a Symbol"
        raise TypeError(msg)
    return symbol.value
```
appears verbatim in three modules; `_require_text` in two.

**Why it matters.** `ops/primitives.py` already exists as the shared ops helper module
(`_execute_language`). These two guards belong there. Low stakes, but it's copy-paste that
will drift.

**Proposed change (do not apply).** Move `_symbol_value` and `_require_text` into
`ops/primitives.py`; import from there in text/reason/compare/rank; replace the inlined
`isinstance` guards in `rank.py`/`compare.py` with calls.

**Feature impact:** `keeps-all`. **Confidence:** high. **Impact:** low. **Effort:** S.

---

### S6 — `decode_output` `default`/`limit`/`output_index` + `_limit_value`; `Function.execute_many` — spec'd but test-only

**What.** The only production caller of `decode_output` is
`ops/primitives.py::_execute_language`, which calls `decode_output(response, decoder)` with
**none** of the optional args. `default`, `limit`, `output_index`, and the `_limit_value`
helper are exercised **only** in `tests/test_decoding.py`. `Function.execute_many` is called
only in `tests/test_components.py`. Additionally, `output_index > 0` is currently
unreachable: no request builder or `SamplingConfig` field can request multiple completions
(`n`), so multi-output responses can't be produced by this stack.

**Where.** `decoding.py::decode_output(..., output_index=0, default=MISSING, limit=None)` and
`_limit_value`; `function.py::execute_many`.

**Why it matters.** This is the honest counter-weight to S1/S2: unlike those, these ARE
specified — `SYMBOL_REDESIGN.md` explicitly lists `output_index`/`default`/`limit` on
`decode_output` and documents `execute_many` as stable-order sequential execution. So this is
**not** a clear removal; it's a "does the forward-looking surface earn its keep before the
second real caller exists" question. Flagged for a judgment call, not action.

**Proposed change (do not apply).** Option A (keep): leave as spec'd, accept test-only
coverage as the contract guard. Option B (defer): trim `limit`/`_limit_value` (pure convenience
with no consumer and no spec-critical role) and reintroduce when an op needs truncation;
keep `output_index`/`default` since they're cheap and clearly anticipated. I lean Option A
given the explicit spec — noting it only so the team confirms intent.

**Feature impact:** `drops-minimal` if cut (spec'd conveniences, no live consumer).
**Confidence:** high (that they're unused in production). **Impact:** low. **Effort:** S.

---

### S7 — `Runtime` acquires `_lifecycle_lock` on every `execute()` for a single-owner-thread object

**What.** `Runtime` is documented "Single-threaded lifecycle owner" and rejects foreign-thread
access via `_require_owner_thread`. Yet `execute()` (the hot path) takes `_lifecycle_lock`
around engine selection on every call, in addition to the ownership check.

**Where.** `runtime/runtime.py::execute`:
```python
with self._lifecycle_lock:
    self._require_owner_thread("execute")
    ...
    selected = self._resolve_engine(...)
return selected.execute(request)
```

**Why it matters.** Under the ratified single-owner-thread contract (and once ambient
ContextVar discovery is removed per S3), only the owner thread can ever reach `execute()`,
so a per-call lock guards no concurrent access on the steady-state path. The lock's genuine
job is the tiny pre-ownership window (two threads racing the first `__enter__` before
`_owner_thread_id` is set). Taking it on every `execute()`/`close()` is defensive machinery
beyond what the concurrency model requires.

**Proposed change (do not apply).** Keep the lock scoped to lifecycle transitions
(`__enter__`/`__exit__`/`close`) where the pre-ownership race exists; drop it from `execute()`
and rely on `_require_owner_thread` + the `ACTIVE` state check. (Correctly note: I/O already
happens outside the lock, so this doesn't change concurrency of the network call.)

**Feature impact:** `keeps-all`. **Confidence:** med (the pre-ownership window is subtle;
worth a careful read before acting). **Impact:** low. **Effort:** S.

---

## What's already good — keep

- **`Symbol`'s ~40 operator dunders are NOT over-built.** They are the complete, uniform
  realization of the spec's §4.3 "native operator contract" (equality/ordering, membership,
  arithmetic/bitwise/unary → new `Symbol[U]`, indexing, iteration, casts). The symmetry
  (including `__r*__` reflected forms, `__matmul__`, `__divmod__`, `__pow__` with modulo) is
  what makes "wrap any `T`, use native Python operators" true; dropping any creates a contract
  hole. Explicit hand-written dunders beat a metaclass loop here — better for pyright and
  matches the repo's explicitness rules. Verified against the design; keep as-is.
- **Discriminated-union message/response/response-format models** (`Message`, `Content`,
  `ResponseFormat` with `Field(discriminator=...)` at exactly one level) are the right shape
  and correctly avoid nested-discriminator pitfalls.
- **`FrozenModel` (`frozen + strict + extra="forbid")` and the `_client` `StrictModel`/
  `TolerantModel` split** are earning their keep as the normalized-contract backbone.
- **The decoder `Protocol` + tiny frozen decoder dataclasses** (`TextDecoder`,
  `ConstructorDecoder`, `TypeAdapterDecoder`, `PydanticDecoder`) are a clean, minimal strategy
  set — no over-abstraction there.
- **Lazy provider imports in `loading.py`** (thin `_load_*` thunks + builtin registry) keep
  `import symai` inert, which the cutover test enforces; the public-vs-generic loading split is
  justified, not accidental.
- **`Runtime`'s explicit single-owner-thread lifecycle** (minus the ambient ContextVar per S3
  and the per-execute lock per S7) is a sound, simple ownership model.
