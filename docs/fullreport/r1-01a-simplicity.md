# r1-01a — Accidental complexity / over-engineering / YAGNI

## Executive summary

- The codebase is mostly lean and well-bounded; the biggest simplicity wins are **vestigial machinery the ratified design already deletes on paper** but that still lives in code — and, in two cases, live tests are *currently failing* demanding its removal (the editing agent is likely mid-cutover).
- **Load-bearing removals:** the ambient-runtime `ContextVar`/`current_runtime()`/`NoActiveRuntimeError` triad (F1, keeps-all, tests failing now), and `Function.static_context`/`dynamic_context` (F2, drops-minimal, forbidden by an AST-scan test).
- **Real over-engineering that does not drop features:** the `LanguageModelSpec` capability matrix is a *parallel structure* shadowing the hardcoded per-field validation — 4 of its fields and 3–4 StrEnums (`MessageRole`/`ContentType`/`ResponseFormatType`/`ReasoningField`) are populated-but-never-read-for-enforcement (F3); the bespoke `JsonObject`/`JsonArray`/`JsonEntry` Pydantic AST is reachable only through an as-yet-uncalled JSON-schema path and is flattened straight back to `pydantic.JsonValue` at the boundary (F4).
- **Over-parameterized-for-one-caller:** `decode_output`'s `default`/`limit`/`output_index` + `_limit_value` (F5) and `Function.execute_many` (F6) have zero production callers; per-provider `settings.py` duplicates one 5-field model four times (F7).
- **Explicitly NOT over-engineering:** `Symbol`'s ~46 operator dunders are a spec-mandated, uniform, I/O-free value-DSL contract (§4.3) — keep them. `ops.primitives._execute_language`, the frozen discriminated-union contracts, and the loader preflight are clean and should stay.

## Findings table

| ID | Finding | Feature impact | Conf | Impact | Effort |
|----|---------|----------------|------|--------|--------|
| F1 | Dead ambient-runtime machinery (`_CURRENT_RUNTIME`, `current_runtime`, `NoActiveRuntimeError`, `_token`, enter/exit token dance) — **tests failing now** | keeps-all | high | high | S |
| F2 | `Function.static_context`/`dynamic_context` + `_system_prompt()` composition — vestigial, forbidden by design + AST-scan test | drops-minimal | high | med | S |
| F3 | `LanguageModelSpec` capability matrix parallels hardcoded validation; `context_tokens`/`message_roles`/`content_types`/`response_formats` + 3–4 StrEnums populated-but-unread | drops-minimal | med | high | M |
| F4 | Bespoke `JsonObject`/`JsonArray`/`JsonEntry` AST — no production caller, flattened to `pydantic.JsonValue` at boundary | drops-minimal | med | med | M |
| F5 | `decode_output` `default`/`limit`/`output_index` + `_limit_value` — sole production caller passes none | drops-minimal | high | med | S |
| F6 | `Function.execute_many` — one test caller, no ops use | drops-minimal | high | low | S |
| F7 | `providers/*/settings.py` — one 5-field model duplicated 4× across 3 files | keeps-all | high | low | S |
| F8 | Duplicated `_symbol_value`/`_require_text` helpers across `ops/text|reason|compare` | keeps-all | high | low | S |
| F9 | Runtime concurrency triad: once F1 lands, `_lifecycle_lock` largely shadows `_require_owner_thread` for a "single-threaded owner" | keeps-all | low | low | M |

---

## Detailed findings

### F1 — Ambient-runtime machinery is dead, contradicts the design, and breaks live tests

**What.** `runtime/runtime.py` still carries the entire ambient-discovery apparatus the ratified design removed: a module `ContextVar`, a public `current_runtime()`, a `NoActiveRuntimeError`, a `_token` slot, and the `__enter__`/`__exit__` set/reset token dance.

**Where.** `symai/runtime/runtime.py`:

```python
_CURRENT_RUNTIME: ContextVar[Runtime | None] = ContextVar("symai_active_runtime", default=None)

def current_runtime() -> Runtime:
    runtime = _CURRENT_RUNTIME.get()
    if runtime is None:
        ...
        raise NoActiveRuntimeError(msg)
```

and in `__enter__`: `token = _CURRENT_RUNTIME.set(self); self._token = token`; in `__exit__`: `_CURRENT_RUNTIME.reset(token)`. `NoActiveRuntimeError` lives in `runtime/errors.py` and is re-exported from `symai/__init__.py`.

**Why it matters.** Verified: `current_runtime()` has **zero callers** anywhere (production or tests). `_CURRENT_RUNTIME` is read only inside the never-called `current_runtime()`. So `__enter__`/`__exit__` maintain a ContextVar + token that nothing consumes — pure accidental complexity, and it's one of the three concerns bundled into the lifecycle lock (see F9). This is not merely unused: the ratified spec says ambient discovery is removed, and live tests demand it. I ran them:

```
tests/runtime/test_runtime.py::test_runtime_has_no_ambient_registry_slot_or_module_state  FAILED
  assert '_token' not in Runtime.__slots__          -> AssertionError
tests/test_public_cutover.py::test_runtime_module_exposes_no_ambient_registry_or_provider_clients  FAILED
  assert not hasattr(runtime, '_CURRENT_RUNTIME')   -> AssertionError
```

`test_public_cutover._production_ast_violations` additionally AST-scans all of `symai/` and forbids the identifiers `_CURRENT_RUNTIME`, `current_runtime`, `NoActiveRuntimeError`. The editing agent may already be removing this; report reflects the live tree.

**Proposed change (do not apply).** Delete `_CURRENT_RUNTIME`, `current_runtime`, the `_token` slot, and `NoActiveRuntimeError`; drop them from `symai/__init__.py`. `__enter__` collapses to `require owner → check state CREATED → set owner_thread_id/state ACTIVE`; `__exit__` drops the token reset and just calls `close()`. Removes ~20 lines plus an export.

**Feature impact.** keeps-all (the removed capability — implicit global runtime lookup — is exactly what the redesign forbids). **Confidence** high. **Impact** high. **Effort** S.

---

### F2 — `Function.static_context`/`dynamic_context` are vestigial and forbidden

**What.** `Function` still declares `static_context`/`dynamic_context` fields and composes them into the system prompt.

**Where.** `symai/function.py`:

```python
def _system_prompt(self) -> str:
    parts = [self.prompt]
    if self.static_context:
        parts.append(f"<STATIC_CONTEXT/>\n{self.static_context}")
    if self.dynamic_context:
        parts.append(f"<DYNAMIC_CONTEXT/>\n{self.dynamic_context}")
    return "\n".join(part for part in parts if part)
```

**Why it matters.** No `ops.*` operation ever sets these — every op constructs `Function(prompt, examples=...)`. They are non-default only in `tests/test_components.py`. Meanwhile `tests/test_public_cutover.py` lists `static_context` in `FORBIDDEN_IDENTIFIERS`, and its AST scan flags any production use of that attribute — so `function.py` currently violates that scan while `test_components.py` still exercises the field. That internal contradiction is the fingerprint of an in-progress cutover: the feature is slated for deletion. `_system_prompt()` exists solely to weave two always-empty strings; without them it is `return self.prompt`.

**Proposed change (do not apply).** Drop both fields and their `__init__` params; replace `_system_prompt()` with direct use of `self.prompt` in `request()`. `Function` becomes `prompt + examples + max_tokens + stop`.

**Feature impact.** drops-minimal (a two-slot prompt-prefix convention no operation uses; callers who want context concatenate into `prompt`). **Confidence** high. **Impact** med. **Effort** S.

---

### F3 — `LanguageModelSpec` capability matrix is a parallel structure shadowing hardcoded validation

**What.** `LanguageModelSpec` advertises a broad capability matrix (11 fields), but the actual per-request enforcement in each engine's `_validate_request` is hand-written field-by-field. Several spec fields — and the StrEnums that feed them — are populated but never read for enforcement.

**Where.** `symai/runtime/models.py` `LanguageModelSpec`, populated by `_normalized_model_spec` in all three engines. Verified reads across `symai/**` (non-test):

- **Read for enforcement:** `response_tokens` (max-tokens bound, all 3), `reasoning_efforts` (membership, all 3), `reasoning_summaries` (openai), `reasoning_formats` (cerebras), `vision` (openai), `sampling_fields` (openai, only `TEMPERATURE`/`TOP_P` membership-tested), `reasoning_fields` (openai, **truthiness only**).
- **Never read outside tests:** `context_tokens`, `message_roles`, `content_types`, `response_formats`.

The enums that feed only the unread fields:

```python
_ALL_MESSAGE_ROLES = tuple(MessageRole)          # -> message_roles (never read)
_ALL_RESPONSE_FORMATS = tuple(ResponseFormatType) # -> response_formats (never read)
content_types=(ContentType.TEXT, ContentType.IMAGE) if spec.vision else (ContentType.TEXT,)  # never read; vision is checked directly
```

`ReasoningField`'s individual members (`ENABLED`/`EFFORT`/`SUMMARY`/`FORMAT`/`CLEAR`) are **never membership-tested** — engines hardcode `if reasoning.enabled is not None: unsupported(...)`, `if reasoning.format is not None: ...`, etc. `reasoning_fields` is consumed only as `if not self.model_spec.reasoning_fields` in openai (a bool). Likewise most `SamplingField` members are populated into tuples but enforcement for `stop`/`seed`/`frequency_penalty`/… is hardcoded per field.

**Why it matters.** This is the anti-pattern called out in the repo's own python.md ("Avoid parallel data structures … create a synchronization invariant the type system cannot enforce"). The matrix must be kept in lockstep with the real hardcoded checks, yet contributes nothing the checks don't already encode; two of the three engines fill `message_roles`/`response_formats` with the *same constant* `tuple(Enum)`, i.e. zero per-model information. Four StrEnums exist only to populate advertisements no code consumes.

**Proposed change (do not apply).** Reduce `LanguageModelSpec` to the fields that actually drive validation: `response_tokens`, `reasoning_efforts`, `reasoning_summaries`, `reasoning_formats`, `vision`, plus explicit booleans where a bare truthiness is meant (`supports_reasoning`, `supports_temperature`, `supports_top_p`). Delete `context_tokens`/`message_roles`/`content_types`/`response_formats` from the spec and the `MessageRole`/`ContentType`/`ResponseFormatType` enums (message roles/content/format discriminators already live as `Literal[...]` on the message and response-format models, so nothing else depends on those enums). Collapse `ReasoningField` to the booleans above.

**Feature impact.** drops-minimal — the only thing lost is capability *introspection* (`engine.model_spec.message_roles`), which no production code performs; a few provider tests assert on it and would move to the slimmed fields. If the team wants the spec kept as a documented public capability surface, downgrade to "keep but stop pretending it enforces anything" — but then delete the unread enums at minimum. **Confidence** med (tests lock the current surface; intent may be to keep as docs). **Impact** high. **Effort** M.

---

### F4 — Bespoke `JsonObject`/`JsonArray`/`JsonEntry` AST is round-tripped straight back to builtins

**What.** `runtime/models.py` defines a hand-rolled, frozen Pydantic JSON AST (`JsonEntry`/`JsonArray`/`JsonObject` + `JsonScalar`/`JsonValue`, a recursive `model_rebuild`, `_parse_json_value`, `_json_value_to_builtin`, `to_builtin`) used only as the type of `JsonSchemaResponseFormat.json_schema`.

**Where.** `symai/runtime/models.py` (lines ~116–192, 261). Consumers:

```python
# openai + cerebras engines, at the client boundary:
schema=cast("JsonValue", response_format.json_schema.to_builtin())   # pydantic.JsonValue
body=cast("JsonValue", response_format.json_schema.to_builtin())
```

**Why it matters.** Verified: (1) **no production code constructs it** — `JsonObject.parse(...)` / `JsonSchemaResponseFormat(json_schema=...)` appear only in tests; the ops layer's `language_request` always defaults to `TextResponseFormat`, so the whole json-schema path is test-only today. (2) At the one boundary where it's read, it is immediately flattened by `to_builtin()` and `cast` to **`pydantic.JsonValue`** — the very type the provider client models already use for their schema fields (`from pydantic import JsonValue` in `openai/client/responses.py`, `cerebras/client/chat.py`). So the structural typing is discarded exactly where the value crosses into the client. The AST's net contribution over `pydantic.JsonValue` is deep-freeze + unique-key + finite-number validation — but the containing model is `frozen=True` anyway, dict keys are unique by construction, and Pydantic validates JSON-serializability and finite numbers natively.

**Proposed change (do not apply).** Type `JsonSchemaResponseFormat.json_schema` as `pydantic.JsonValue` (or a constrained `Mapping[str, JsonValue]`). Delete `JsonEntry`/`JsonArray`/`JsonObject`/`_parse_json_value`/`_json_value_to_builtin`/`to_builtin` and the `model_rebuild` dance; engines pass `response_format.json_schema` through directly (no `.to_builtin()`, no `cast`). Removes ~75 lines and 3 public exports.

**Feature impact.** drops-minimal — loses a deep-immutable/round-trip-identity schema representation that is cast away at the boundary regardless. **Confidence** med. **Impact** med. **Effort** M.

---

### F5 — `decode_output` is over-parameterized for its single production caller

**What.** `decode_output` carries `output_index`, `default`/`Missing` sentinel, and `limit`, with a helper `_limit_value` that truncates decoded list/tuple/dict.

**Where.** `symai/decoding.py`:

```python
def decode_output(response, decoder, *, output_index=0, default=MISSING, limit=None) -> T:
    text = _output_text(response, output_index)
    try:
        value = decoder.decode(text)
    except DecodeError:
        if isinstance(default, Missing):
            raise
        value = default
    return _limit_value(value, limit)
```

**Why it matters.** The **only** production caller is `ops/primitives._execute_language`, which calls `decode_output(response, decoder)` — no optional args. `output_index`, `default`, `limit`, the `Missing` marker class, and `_limit_value` (with its three container-type branches) are exercised only by `tests/test_decoding.py`. `limit` in particular — truncating a *decoded* list/tuple/dict — is an odd responsibility for a decoder boundary and reads as a leftover from the pre-redesign framework.

**Proposed change (do not apply).** Trim to `decode_output(response, decoder)` (optionally retain `output_index` if multi-output responses are on the near-term roadmap). Delete `default`/`Missing`/`limit`/`_limit_value`. Callers wanting a fallback wrap in try/except; callers wanting truncation slice the returned value.

**Feature impact.** drops-minimal (fallback-on-DecodeError and post-decode truncation, neither used by any operation). **Confidence** high. **Impact** med. **Effort** S.

---

### F6 — `Function.execute_many` has no production caller

**What.** A sequential batch-execute convenience with defensive string-guards.

**Where.** `symai/function.py` `execute_many` (guards: `isinstance(inputs, str)` and per-item `isinstance(values, str)`).

**Why it matters.** Only caller is `tests/test_components.py`. No `ops.*` uses it. It adds a public method + two defensive type checks for a one-line `tuple(self(runtime, *v, engine=engine) for v in inputs)` callers can write themselves.

**Proposed change (do not apply).** Remove it; if a batch helper is genuinely wanted later, add it when a real second caller appears (per python.md "defer until the second real caller").

**Feature impact.** drops-minimal. **Confidence** high. **Impact** low. **Effort** S.

---

### F7 — Per-provider `settings.py` duplicates one model four times

**What.** Four settings classes across three files with **identical** 5-field bodies (`api_key`, `model`, `request_timeout`, `connect_timeout`, `connect_retries`).

**Where.** `openai/settings.py` (`ResponsesSettings`, `EmbeddingSettings`), `cerebras/settings.py` (`ChatCompletionsSettings`), `deepseek/settings.py` (`ChatCompletionsSettings`) — byte-for-byte the same fields/defaults.

**Why it matters.** Pure duplication; a field/default change (e.g. bumping `request_timeout`) must be made in four places. `providers/_client/` already exists as the home for shared client concerns.

**Proposed change (do not apply).** Define one `EngineSettings(FrozenModel)` in `providers/_client/` (or `runtime/models.py`) and have each `loading.py` validate against it (they already only read `.api_key`/`.model`/timeouts). If per-engine divergence is anticipated, subclass; otherwise use the shared class directly.

**Feature impact.** keeps-all. **Confidence** high. **Impact** low. **Effort** S.

---

### F8 — Duplicated `_symbol_value`/`_require_text` across ops modules

**What.** Identical private helpers copied across ops modules (confirmed seed #7).

**Where.** `_symbol_value` in `ops/text.py`, `ops/reason.py`, `ops/compare.py`; `_require_text` in `ops/text.py`, `ops/reason.py` (`ops/rank.py`/`ops/compare.py` inline equivalent checks).

**Why it matters.** Three copies of the same 3-line guard; `ops/primitives.py` already exists as the shared-helper home (`_execute_language`).

**Proposed change (do not apply).** Hoist `_symbol_value` and `_require_text` into `ops/primitives.py`; import from there. Minor.

**Feature impact.** keeps-all. **Confidence** high. **Impact** low. **Effort** S.

---

### F9 — Runtime concurrency triad thins out once F1 lands

**What.** `Runtime` guards its state machine with three mechanisms: a `_lifecycle_lock`, a `_owner_thread_id` ownership check (`_require_owner_thread`), and (F1) the ContextVar token.

**Where.** `symai/runtime/runtime.py` — every public method opens `with self._lifecycle_lock:` then calls `self._require_owner_thread(...)`.

**Why it matters.** The class is documented as a "Single-threaded lifecycle owner," and `_require_owner_thread` already rejects any non-owner thread. For a single-owner object the lock's remaining job is narrow (serializing the brief window during `__enter__` before `_owner_thread_id` is set, where ownership is not yet asserted). With the ContextVar gone, it's worth asking whether the lock earns its keep versus asserting ownership and a plain state check. This is nuanced (the lock does close a genuine pre-entry race), so it's a low-confidence prompt to reconsider, not a clear cut.

**Proposed change (do not apply).** After F1, evaluate replacing the per-method lock with ownership-assertion + state check, keeping a lock only around the `CREATED→ACTIVE` transition if the pre-entry race is real for the intended usage. Do not do this blindly — confirm the concurrency contract first.

**Feature impact.** keeps-all (single-thread contract preserved). **Confidence** low. **Impact** low. **Effort** M.

---

## What's already good — keep

- **`Symbol`'s ~46 operator dunders are NOT over-engineering.** The seed asks whether all are needed; verification says yes. §4.3 of `SYMBOL_REDESIGN.md` ratifies "Arithmetic / bitwise / unary value operations → new `Symbol[U]`" as a *contract family*. The dunders are uniform two-line transparent forwards over `_unwrap_operand`, perform no I/O, and give a coherent value DSL. Dropping bitwise/matmul/divmod would break a documented family and the uniformity for negligible LOC savings. Keep as-is.
- **`ops.primitives._execute_language`** — correct shared seam for the language-op pattern; good dedup already in place.
- **Frozen, `strict`, `extra="forbid"` normalized contracts + single-level discriminated unions** (`Message`, `Content`, `ResponseFormat`) — clean and exactly per house style.
- **Loader design** (`runtime/loading.py` allocation-free preflight + reverse-order failure cleanup; `symai/loading.py` builtin+extension merge) — well-bounded, no over-abstraction.
- **`ops/embed.py` numeric core** — the metric/kernel dispatch is genuine feature surface (cosine/dot, euclidean/manhattan/minkowski, linear/rbf/polynomial, MMD) with real validation, not accidental complexity. Keep.

---

### Cross-references / prior-audit labels
- Seed #1 (ambient runtime) → **STILL-OPEN**, now with failing tests (F1).
- Seed #2 (static/dynamic context) → **STILL-OPEN** (F2).
- Seed #6 (JSON AST + spec matrix) → **STILL-OPEN** (F3, F4).
- Seed #7 (duplicated ops helpers) → **STILL-OPEN**, minor (F8).
