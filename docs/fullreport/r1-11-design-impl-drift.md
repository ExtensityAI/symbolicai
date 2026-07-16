# R1-11 — Design ↔ Implementation Coherence

Lens: does the code match `audit/SYMBOL_REDESIGN.md` (§§4–11) and `audit/FIXPLAN.md`
(§§4–10), and what is the status of the `FINDINGS.md` register against current code?

**Audit window is a MOVING TARGET.** Frozen HEAD = `09bab6a`. Another agent cut the
tree over *live while I audited*. I therefore report two verdicts where they differ:
the state at **HEAD `09bab6a`** (what the seed signals captured) and the state in the
**working tree** at my final verification. Every working-tree verdict was re-confirmed
after the edits landed. Full suite at final check: **`620 passed`**; the dedicated
cutover suites (`test_public_cutover.py`, `test_symbol_runtime_cutover.py`) are
**66/66 green** (they were 6-red at first read).

---

## Executive summary

1. **The seed drifts were real at `09bab6a` and were closed during the audit.** At the
   frozen HEAD, `current_runtime()`/`_CURRENT_RUNTIME`/`NoActiveRuntimeError` still
   existed and were exported; `Function` still had `static_context`/`dynamic_context`;
   the root `__init__.py` was a ~90-name facade that omitted `Symbol`; `prompts.py`
   (71 KB) and `backend/` still existed. I watched all of these get removed
   file-by-file. Working tree now **matches** the design on every one.
2. **The Symbol/Function/decoding/ops cutover is coherent and complete.** `Symbol` is
   shallow-immutable, unhashable (`__hash__ = None`), no `__getattr__`, no `.sem/.syn`,
   no context/embedding/persistence state, native operators propagate original Python
   exceptions. ops are explicit free functions taking `Runtime`+`Symbol`; `template`
   and the embed-math ops are local (no Runtime). This is the strongest part of the work.
3. **No tool-calling residue anywhere** — no `tool_calls`/`tools`/`function_call`
   symbols in `symai/`, and `FinishReason` has no tool member. An unknown provider
   finish reason falls through to an error, per FIXPLAN §8.
4. **Durable drifts survive on surfaces the cutover didn't touch** (`decoding.py`,
   `runtime/models.py`, `providers/`, `pyproject.toml`): an **extra `PydanticDecoder`**
   the design's decoder list never names; the bespoke **`JsonObject` AST** (CX-01);
   the **logprobs request/response hole** (CON-02); packaging still frozen at
   **`1.18.0`** with dead **`jinja2`/`python-box`** deps.
5. Provider-path correctness findings are largely **fixed**: model-echo rejection
   (BUG-05/06) is gone (separate `requested_model`/`response_model`), the credential
   boundary (SEC-01) now validates control chars/whitespace without leaking the key,
   and the null-content contract (BUG-07) admits a content-filter terminal state.

---

## Table A — Design-claim drift (SYMBOL_REDESIGN §§4–11, FIXPLAN §§4–10)

`@09bab6a` = frozen HEAD; `now` = working tree at final verification.

| # | Design claim | Code reality | Verdict |
|---|---|---|---|
| A1 | §9 / FIXPLAN §5: `current_runtime()` + ambient `ContextVar` **removed** | `@09bab6a`: `runtime.py` defined `_CURRENT_RUNTIME`, `current_runtime()`, set/reset it in `__enter__/__exit__`; `errors.py` had `NoActiveRuntimeError`; `__init__` exported both. `now`: `grep current_runtime\|_CURRENT_RUNTIME\|NoActiveRuntimeError symai` → **0 hits** | **DRIFT@09bab6a → MATCHES now** |
| A2 | §6 / FIXPLAN §6: Function has **no static/dynamic context** | `@09bab6a`: `function.py` had `static_context`/`dynamic_context` fields + `_system_prompt()` composing `<STATIC_CONTEXT/>`. `now`: fields gone; `request()` passes `self.prompt` directly | **DRIFT@09bab6a → MATCHES now** |
| A3 | §3.1: root is **empty, not a compatibility facade**; canonical imports from owning modules | `@09bab6a`: `__init__.py` = 142 lines, `__all__` of ~90 names, and it **omitted `Symbol`/`ops`**. `now`: `__init__.py` is **0 bytes**; `test_canonical_modules_own_their_public_types` green | **DRIFT@09bab6a → MATCHES now** |
| A4 | §8: `Expression`,`Result`,graph/linker/nodes/edges absent from **code AND tests** | No production refs; only forbidden-name lists in tests. `test_production_ast_has_no_legacy_graph_references` green | **MATCHES** |
| A5 | §8: `.sem`/`.syn`/`_semantic` removed | `grep` → only in tests' forbidden lists. `symbol.py` has no semantic flags | **MATCHES** |
| A6 | §8: `static_context`/`dynamic_context`/`global_context` removed | `now`: 0 production hits (only test forbidden-lists) | **DRIFT@09bab6a → MATCHES now** |
| A7 | §8: `adapt()`/`clear()` + Symbol embedding cache removed | No `adapt`/`clear`; `symbol.py` `__slots__=("_value",)` only | **MATCHES** |
| A8 | §8: `save()`/`load()` / pickle removed from Symbol | No `save`/`load`/`pickle` in `symai/` | **MATCHES** (supersedes PERSIST-01, SEC-03) |
| A9 | §4.4: no broad `__getattr__` forwarding on Symbol | `symbol.py` has no `__getattr__`; `_unwrap_operand` only reaches into operands | **MATCHES** |
| A10 | §8 / FIXPLAN §6: `sym_return_type` removed; Function non-generic | No `sym_return_type`; `Function` is a plain frozen dataclass returning `LanguageModelResponse` | **MATCHES** |
| A11 | §8: operation-mixin inheritance removed | `Symbol(Generic[T])` inherits nothing else; `test_old_mixin_context_and_symbol_surfaces_are_absent` green | **MATCHES** |
| A12 | §8: `Prompt`/`PromptRegistry` hierarchy removed; jinja2/box/tomllib | `@09bab6a`: `prompts.py` (71 KB) + those deps present. `now`: `prompts.py` **deleted**; ops own inlined `_MODIFY_EXAMPLES = (...)` tuples | **DRIFT@09bab6a → MATCHES now** (deps still declared — see B/PKG) |
| A13 | §2 / FIXPLAN §8: **no tool/function calling** residue | 0 hits for `tool_calls`/`tools`/`function_call`/`ToolCall`; `FinishReason` = `{stop,length,content_filter,error}`, no tool member; unknown finish reason → error path | **MATCHES** |
| A14 | §7: decoder set is `TextDecoder`, `ConstructorDecoder`, `TypeAdapterDecoder` | `decoding.py` ships a **4th, `PydanticDecoder`**, not named in §7. Redundant with `TypeAdapterDecoder(TypeAdapter(Model))` shown in §7's `list[User]` example | **DRIFT (extra surface, minor)** |
| A15 | §7: "nested/container typing uses `TypeAdapter`, not bare runtime classes" | `ConstructorDecoder.decode` special-cases bare `list/tuple/set/dict` via `ast.literal_eval` (`decoding.py:57`). Mild tension with the "scalar conversion" role FIXPLAN §6 assigns it | **PARTIAL (minor)** |
| A16 | §6 / FIXPLAN §6: `request()` = preview (no I/O); metadata always present; `execute_many` stable-order sequential | `Function.request()` builds only; `__call__` → `runtime.execute`; `execute_many` is `tuple(self(...) for ...)`; `ResponseMetadata` non-optional on response | **MATCHES** |
| A17 | §7: `default` catches only decode failure; index selection raises `IndexError`; sets pass through | `decode_output` catches `DecodeError` only; `_output_text` raises `IndexError`; `_limit_value` leaves non-list/tuple/dict (incl. `set`) unchanged | **MATCHES** (supersedes BUG-09, BUG-10) |
| A18 | §5 / FIXPLAN §7: ops take `Runtime`+`Symbol`, forward `engine=` only, no provider/model option; local ops take no Runtime | text/reason/compare/rank take `(runtime, Symbol, ..., *, engine)`; `text.template` + all `embed` math take no runtime/engine; `is_instance_of(type_description: str)` | **MATCHES** |
| A19 | FIXPLAN §5: named engine instances, owner-thread affinity, reverse cleanup, no ambient discovery | `Runtime` keyed by name maps, `_require_owner_thread`, `close()` iterates `reversed(acceptance_order)` aggregating into `BaseExceptionGroup` | **MATCHES** |
| A20 | FIXPLAN §4: model identity records **both** requested + returned, no equality rejection | every engine sets `requested_model=self.model, response_model=raw.model`; no `raw.model != self.model` rejection anywhere | **MATCHES** (supersedes BUG-05/06) |

---

## Table B — Prior-finding status (FINDINGS.md, verified against current code)

| ID | Sev | Status | One-line code citation |
|---|---|---|---|
| BUG-01 | High | **FIXED** | `runtime.py`: no `Condition`/`_in_flight`/`CLOSING`/`wait_for`; `close()` sets `CLOSED` under lock then closes engines |
| BUG-02 | Low | **SUPERSEDED** | `current_runtime()` deleted; ops take Runtime explicitly, no ContextVar discovery |
| BUG-03 | Low | **SUPERSEDED** | ambient ContextVar removed; `__exit__` owner-thread-guarded before any reset |
| BUG-05/API-01 | Crit | **FIXED** | `responses.py:397` `requested_model=self.model, response_model=raw.model`; no equality reject |
| BUG-06 | High | **FIXED** | `openai/engines/embedding.py:178-179` same separate-model recording |
| BUG-07 | High | **FIXED** | `models.py:381` `validate_content_reasoning_or_refusal` permits empty when `finish_reason is CONTENT_FILTER`; `refusal` field + `"content_filter"` maps |
| BUG-08 | Med | **PARTIAL/CANT-VERIFY** | `LanguageModelResponse.outputs` still `min_length=1`; reasoning-only truncation handling not confirmable from fixtures alone |
| BUG-09 | High | **SUPERSEDED** | old `limit=1` path gone; `decode_output` default `limit=None` (no truncation) |
| BUG-10 | High | **SUPERSEDED/FIXED** | `_limit_value` returns `set` unchanged (no `TypeError`) |
| BUG-11 | Low | **SUPERSEDED** | recursive `_recursive_literal` coercion path removed with old `operations.py` |
| BUG-12 | Med | **FIXED** | no `_dynamic_context` class global on `Symbol`; `symbol.py __slots__=("_value",)` |
| BUG-13 | High | **FIXED** | `Symbol.__hash__ = None` |
| BUG-14 | Med | **FIXED** | `Symbol.__getitem__` = `self._value[key]`, no blanket `except`; native `IndexError/KeyError/TypeError` propagate |
| SEC-01 | High | **FIXED** | `_client/headers.py:authorization_header` rejects empty/leading/trailing-space and control chars (`<0x20`/`0x7F`); raises bare `ValueError` (no key in message) |
| CX-01 | High | **STILL-OPEN** | `models.py:119-160` `JsonEntry/JsonArray/JsonObject` AST + `to_builtin()` + private `model_rebuild(_types_namespace=...)` intact |
| CX-02 | High | **PARTIAL** | matrix now *drives* some checks (`reasoning_efforts`, `vision`, `sampling_fields` read in engines) but is still `if`-ladder-enforced, not spec-driven |
| CX-04 | Med | **FIXED** | dormant concurrency machinery gone; owner-thread affinity enforced |
| CX-05 | Low | **FIXED/SUPERSEDED** | `backend/engine_handle.py` deleted; no `EngineHandle` lock/`owns_resources` |
| CON-01 | High | **PARTIAL** | some spec fields now read; `context_tokens`/`message_roles`/`content_types` still not enforced (no token counting) |
| CON-02 | High | **STILL-OPEN** | `SamplingConfig.logprobs/top_logprobs` sent (`cerebras:167`, `deepseek:199`) but `LanguageModelOutput` has no logprobs field |
| CON-03 | Med | **STILL-OPEN (by design)** | `LanguageModelResponse.outputs` is a tuple; no `n` request field exists |
| SOC-01 | High | **FIXED** | `runtime/factory.py` deleted; loading split into `runtime/loading.py` + `providers/*/loading.py`; import-boundary tests green |
| SOC-02 | High | **FIXED** | `test_import_symai_is_subprocess_isolated_and_inert` green: `symai_modules == ["symai"]`, `public_names == []` |
| SOC-03 | High | **PARTIAL** | shared `_client/{models,errors,headers}` extracted, but three engine modules (openai responses / cerebras+deepseek chat_completions) remain separate implementations |
| SOC-04 | Med | **PARTIAL** | `_client/transport`/models shared; per-provider `client/` still re-model OpenAI-compatible chat |
| SOC-05 | Med | **PARTIAL** | `_client/errors.py` (26 L) shared, yet per-provider `errors.py` persist with the same >20-line spread (openai 36 / cerebras 57 / deepseek 53) |
| SOC-07 | Low | **FIXED** | `symai/backend/` deleted entirely |
| CLI-04 | Med | **CANT-VERIFY-FROM-CODE** | error classification lives in `_client`; needs targeted read of `_raise_for_status` to confirm body surfacing |
| API-02/05/06/07/09/12 | var | **CANT-VERIFY-FROM-CODE** | per-model capability matrices exist and are read; correctness vs live docs is a provider-doc claim (R3 scope) |
| API-08/13 | n/a | **PARTIAL** | `FinishReason.ERROR` still mapped (`cerebras:88 "error"`, `deepseek:102`); no tool finish reason; phantom-`error` validity is a doc question |
| EXT-05 | High | **FIXED** | `Runtime` holds name→engine maps; two same-model engines with different keys coexist (`test_runtime` two-instance paths green) |
| EXT-03 | n/a | **FIXED/N-A** | no `EngineHandle` injection seam remains |
| PERSIST-01 / SEC-03 | High | **FIXED** | Symbol persistence removed (no `save`/`load`/`pickle`) |
| PKG-01 | High | **STILL-OPEN** | `pyproject.toml:11 version = "1.18.0"` unchanged despite full API replacement |
| PKG-02/03 | Med | **RESOLVED-BY-DESIGN** | root empty by design; migration guide is the documented contract, not a shim |
| PKG-04 | Med | **STILL-OPEN** | `numpy>=1.26.4,<=2.1.3` cap intact after torch removal |
| PKG-05 | Low | **CONFIRMED (intentional)** | no `[project.scripts]` |
| (deps) seed-4 | — | **STILL-OPEN** | `jinja2>=3.1.0` + `python-box>=7.1.1` still declared; 0 imports after `prompts.py` deletion → dead deps |

---

## Detail on the durable drifts (not closed by the cutover)

### D1 — Extra `PydanticDecoder` beyond the design's decoder set (A14)
**What.** `SYMBOL_REDESIGN §7` names exactly three decoders and shows `list[User]`
decoding via `TypeAdapterDecoder(TypeAdapter(list[User]))`. `decoding.py` adds a
fourth:

```python
@dataclass(frozen=True, slots=True)
class PydanticDecoder(Generic[ModelT]):
    model: type[ModelT]
    def decode(self, text, /): return self.model.model_validate_json(_normalize_text(text))
```

**Why it matters.** `TypeAdapterDecoder(TypeAdapter(Model))` already decodes a single
Pydantic model, so `PydanticDecoder` is redundant public surface the spec never
sanctioned. FIXPLAN §6 requires "Pydantic models" be *covered*, which the TypeAdapter
form satisfies. The design doc was actively edited this window but §7's decoder list
was **not** updated to admit `PydanticDecoder`, so this is a genuine spec/impl gap.
**Proposed change.** Either drop `PydanticDecoder` (rely on `TypeAdapterDecoder`) or
add it to §7 as a sanctioned convenience. **Feature impact:** keeps-all.
**Confidence high · Impact low · Effort S.**

### D2 — Bespoke `JsonObject` AST persists (CX-01)
`models.py:119-160` keeps the frozen `JsonEntry/JsonArray/JsonObject` tree, its
`to_builtin()` round-trip, and three `model_rebuild(_types_namespace={"JsonValue":…})`
calls on pydantic's **private** kwarg — used only to carry a schema into
`JsonSchemaResponseFormat`. Pydantic ships `JsonValue`; a `dict[str, JsonValue]` gives
the same guarantee without the AST or the private-API dependency. Not a
`SYMBOL_REDESIGN` claim, but an open FINDINGS item on a stable surface.
**Confidence high · Impact med · Effort M.**

### D3 — logprobs request/response coherence hole (CON-02)
`SamplingConfig` exposes `logprobs`/`top_logprobs` and engines forward them
(`cerebras:167,184`, `deepseek:199-200`; OpenAI rejects the toggle), but
`LanguageModelOutput` (`models.py:375-401`) has **no field to hold returned
logprobs**. The normalized contract can request what it cannot represent.
**Confidence high · Impact med · Effort M.**

### D4 — Packaging not cut over (PKG-01, PKG-04, dead deps)
`version = "1.18.0"` is unchanged though the entire public API was replaced; `jinja2`
and `python-box` are still declared but unimported after `prompts.py` deletion; the
`numpy<=2.1.3` cap survives torch removal. FIXPLAN §12 makes the major-version bump a
release gate. **Confidence high · Impact high (release) · Effort S.**

---

## What is already good and should be kept

- **`symbol.py`** is an exemplary realization of §4: 185 lines, `__slots__`, unhashable,
  no forwarding, native exceptions propagate, comparisons/casts raw and
  arithmetic/indexing return `Symbol`. Keep as-is.
- **The cutover *tests*** (`test_public_cutover.py` AST guard, subprocess-isolation
  inertness check, `test_symbol_runtime_cutover.py`) are strong executable
  specifications — they are exactly what caught the incomplete cutover and drove it to
  green. Keep and keep running them in CI.
- **Credential boundary** (`_client/headers.py`) is the right fix location (shared
  header construction, not per-`Config`) and correctly avoids putting the key in the
  exception — directly satisfying FIXPLAN §4.
- **Separate `requested_model`/`response_model`** on `ResponseMetadata` is the precise,
  non-`startswith` correction FIXPLAN §4 asked for.
- **Local vs remote op split** (`text.template` and all `embed` math take no Runtime)
  matches §5 exactly and keeps the deterministic surface honest.

---

## Method notes / caveats

- Verdicts labeled `DRIFT@09bab6a → MATCHES now` were each re-confirmed by grep/test
  **after** the working-tree edit removing them landed; the frozen-HEAD half is
  attributable to commit `09bab6a` (HEAD) and the working-tree half to uncommitted
  edits observed via `git status`/`git diff` during the session.
- `CANT-VERIFY-FROM-CODE` marks provider-behaviour claims (R3-class) that need live
  docs/keys, not a code read.
- Full suite `620 passed` and cutover suites `66/66` are the load-bearing evidence that
  the design is realized end-to-end at final verification; because the tree is a moving
  target, treat these as an instant snapshot.
