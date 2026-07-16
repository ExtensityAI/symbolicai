# r2-a1 — System-level architecture synthesis (cross-cutting)

**Round 2 · cross-cutting.** Steps back from individual R1 findings and judges the whole
decomposition now that the legacy cutover is **done** (verified: `symai/__init__.py` is
0 bytes, no `prompts.py`/`backend/`, ambient `_CURRENT_RUNTIME`/`current_runtime`/
`NoActiveRuntimeError`/`_token` all gone from `runtime/runtime.py` + `runtime/errors.py`).
Inputs: `r1-01a/01b` (simplicity), `r1-02` (duplication), `r1-03a` (boundaries),
`r1-04` (value layer), `r1-05` (runtime), `r1-06` (contracts), `r1-07` (adapters). Every
claim below re-verified against the current tree; provenance IDs (F#/S#/B#/C#/PA#/D#/R5#)
point back to the R1 report that first raised it.

---

## System verdict (one paragraph)

**The layer *shape* and *dependency direction* are right, the vertical layer *count* is
justified, and the redesign's central invariant holds — but the decomposition is
lopsided: it is disciplined and near-dead-code-free in the *value/runtime* core, and
under-factored with dead surface at the two ends (provider tier + `runtime/models.py`).**
The DAG `symbol ← ops.* → function → operations/decoding → runtime → engines → client →
_client` is acyclic, `Symbol` is imported only by `ops.*`, and nothing below `Symbol`
imports or returns it (B-lens, re-verified). No layer is *missing* and none is redundant
as a concept. The problems are horizontal, not vertical: (1) the provider engine tier has
**no shared base**, so a ~40% scaffolding skeleton (error ladder ×4, lifecycle ×4,
choice-loop ×2) and a whole `settings.py`/`client` transport shell are kept in sync by
copy-paste; (2) `runtime/models.py` carries ~150 LOC of surface that no request can
exercise (the `JsonObject` AST, four dead `LanguageModelSpec` fields + three orphan enums,
the N-output tuple machinery) — modeled as public contract but never produced or read;
(3) the decoder set and `decode_output` are over-populated for their one production caller.
The provider duplication is a **missing base class, not justified independence** — but only
for the scaffolding tier; the wire schemas and capability catalogs are correctly separate.
Fix the two missing bases and cut the dead contract surface and this is an excellent
architecture. Net removable/relocatable: ~500–650 LOC and ~8 public types with `keeps-all`
or `drops-minimal` impact.

---

## Does each layer earn its keep?

| Layer | Earns it? | Verdict |
|---|---|---|
| `symbol.py` (value DSL) | **Yes** | keep as-is — the whole point; operator table is spec-mandated (F-01a) |
| `ops/*` (Symbol-wrapping ergonomic ops) | **Yes** | keep — sole Symbol-wrap layer; `_execute_language` two-stage is validated (§12) |
| `function.py` (one request build+exec) | **Yes** | keep — distinct "build+execute" concern, now context-free post-cutover |
| `operations.py` (request builders) | **Yes, but misnamed+misplaced** | keep the module, rename→relocate (Move 8); collides with `ops/`, builds only `runtime.models` types |
| `decoding.py` (decoder strategies) | **Yes as concept, over-populated** | slim (Move 6): `PydanticDecoder` redundant, `ConstructorDecoder` fused, `decode_output` over-parameterized |
| `runtime/*` (lifecycle + contracts + loading) | **Yes** | strong core; slim `models.py` dead surface (Moves 3–5); minor lock/validation tidies |
| `providers/*/engines/*` (adapters) | **Yes as a layer, wrong factoring** | the client↔runtime crossing point is real; **missing base** (Move 1) |
| `providers/*/client/*` (raw HTTP) | **Yes, under-shared** | per-provider wire binding is real; **missing client base** (Move 2) |
| `providers/*/settings.py` | **No — not a per-provider concern** | one 5-field model copied 4× across 3 files; collapse to one (Move 7) |
| `providers/*/loading.py` | **Marginal** | near-identical loaders; share a `build_http_engine` helper (Move 7) |
| `providers/_client/*` (shared client seam) | **Yes, under-populated** | right idea, absorbs only leaf helpers today; grow it (Move 2) |

**Layer count verdict.** The vertical depth (~8 hops) is *not* over-layering: `build`
(`operations.py`), `execute` (`function.py`), `parse` (`decoding.py`) are three genuinely
distinct concerns between `ops` and `runtime`, each a small single-purpose module. The
defect is horizontal duplication within the provider tier and dead surface inside
`runtime/models.py` — not too many layers.

---

## Abstractions that don't earn their keep (keep / slim / delete)

| Abstraction | Verdict | Feature impact | Evidence (re-verified live) |
|---|---|---|---|
| `JsonObject`/`JsonArray`/`JsonEntry` AST (`models.py` 119–192) | **Delete → `pydantic.JsonValue`** | drops-minimal | Only consumer `JsonSchemaResponseFormat.json_schema: JsonObject`; both engines do `cast("JsonValue", …json_schema.to_builtin())`; no op produces it (C4/S1/F4) |
| `LanguageModelSpec` dead fields `message_roles`/`content_types`/`response_formats`/`context_tokens` + `MessageRole`/`ContentType`/`ResponseFormatType` enums | **Slim OR promote** (see Move 1 vs 4) | drops-minimal (delete) / keeps-all (gate) | 0 `model_spec.<field>` reads for the three; enums only populate them; Runtime never exposes `model_spec` (C2/S2/F3/PA-2) |
| N-output tuple machinery (`outputs: tuple … min_length=1`, `LanguageModelOutput.index`, per-engine dedup/sort, `decode_output(output_index=…)`, `_output_text` scan) | **Slim → single `output`** | drops-minimal | No `n`/`candidates`/`best_of` field anywhere; ops always call `decode_output(response, decoder)` → index 0 (C3/S6) |
| `PydanticDecoder` | **Delete** (fold into `TypeAdapterDecoder(bare_type)`) | keeps-all | Functionally identical to `TypeAdapterDecoder` for any `BaseModel` (r1-04 #1) |
| `ConstructorDecoder` container branch (`list/tuple/set/dict` via `ast.literal_eval`) | **Delete branch** | drops-minimal | No op uses it (only `ConstructorDecoder(bool)` in `compare`); grammar clashes with `TypeAdapterDecoder` (r1-04 #2) |
| `decode_output` `default`/`Missing`/`limit`/`_limit_value` | **Delete** | drops-minimal | Sole production caller passes none (F5/S6/r1-04) |
| `operations.py` as its own module | **Keep, rename+relocate** | keeps-all | Real "request-builder" concern, but `operations`↔`ops` is a naming hazard; authors only `runtime.models` types (B4) |
| Provider `settings.py` layer | **Delete per-provider; one shared model** | keeps-all | 4 byte-identical 5-field bodies (F7/S4/D8) |

---

## Missing abstractions (should exist, don't)

1. **Shared provider engine base — `providers/_engine/` (the #1 gap).** No such package
   exists (verified: no `providers/_engine/`). It would own the error→runtime mapper
   (PA-1/D1, ×4 verbatim), the `__init__`/cleanup + `close` + `model`/`model_spec` +
   `_retry_after` + `_unsupported` + `_error_metadata` lifecycle (PA-3/D5, 5 methods ×4),
   the **capability gate** (PA-2), and a `ChatCompletionsAdapter` for the cerebras/deepseek
   choice-loop (PA-4/D6). This is the single highest-leverage missing piece.
2. **A capability *gate*** that reads `LanguageModelSpec` as the single enforcement source,
   replacing the parallel hardcoded `if <unsupported>: _unsupported(...)` walls. This is
   what turns three of the four "dead" spec fields into live data (they are per-provider
   constants — deepseek omits `DEVELOPER`/`JSON_SCHEMA`/`IMAGE` — exactly what a gate reads).
3. **Shared client transport shell — `BaseClient` in `providers/_client/`.** `_client.py`
   is 83% duplicated (D2); `transport.py`/`headers.py` are byte-identical modulo docstring
   (D3, re-verified); error `__init__` bodies identical ×3 (D4). `_client/` exists but holds
   only leaf helpers.
4. **One `EngineSettings` model** (Move 7) — removes the settings.py non-layer.
5. **Cleanly-typed implementation-id.** `ImplementationId = Annotated[str, BeforeValidator]`
   exists, but the builtin registry writes `cast("ImplementationId", "openai:responses")`
   ×4 in `loading.py` — a no-op cast papering over the fact that literal ids aren't
   constructed through the validator. Minor: type the registry keys so the cast disappears.
6. (Minor) `runtime/base.py` for `FrozenModel`/`*FiniteFloat` so `models.py` isn't both the
   wire-contract module and the base-pydantic toolkit (B7).

---

## Provider 3× duplication: missing base class or justified independence?

**Definite position: it is a missing base class for the scaffolding tier, and justified
independence for the schema tier — the line between them is sharp and already visible in
the code.**

- **Missing base (dedup — `keeps-all`):** error→runtime mapping, construction/cleanup
  lifecycle, `close`/`model`/`model_spec`/`_retry_after`/`_unsupported`/`_error_metadata`,
  the transport shell (`_client.py`/`transport.py`/`headers.py`), the error `__init__`
  bodies, `settings.py`, and the cerebras↔deepseek choice-loop. All of this references
  **zero provider schema** and every provider error already subclasses
  `providers/_client/errors.py`, so a base catching the shared classes is safe. This is
  ~40% of each engine and most of each `client/`.
- **Justified independence (must stay split):** the per-provider wire schemas
  (`chat.py`/`responses.py`/`embeddings.py`), the `MODEL_SPECS` catalogs, and
  `_normalized_model_spec` (D9 — "the dedup that couples unrelated provider schemas"), and
  **OpenAI Responses as a standalone engine** (status-based finish, heterogeneous output
  items — genuinely a different wire contract from Chat Completions). Forcing these into a
  shared base would relocate divergence into call sites without removing it.

So the answer to "does the 3× duplication indicate a missing base?" is **yes for
infrastructure, no for schemas** — and the shared `_client/` seam already proves the team
drew the line correctly; it is just under-populated.

---

## Top structural moves, ranked by (value × confidence)

> Feature-impact legend: `keeps-all` / `drops-minimal` / `drops-real`.

### Move 1 — Introduce `providers/_engine/`: base + error-mapper + capability-gate + chat-completions base
- **What.** New shared package: `base.py` (lifecycle: `__init__`/cleanup, `close`,
  `model`/`model_spec`, `_retry_after`, `_unsupported`, `_error_metadata`), `mapping.py`
  (one `map_execution` catching the shared `_client.errors.*` base classes → `runtime.errors.*`),
  `gate.py` (data-driven capability enforcement reading `LanguageModelSpec`), `chat.py`
  (`ChatCompletionsAdapter`: choices dedup/sort loop + `_FINISH_REASONS`-driven mapping +
  reasoning hook). Cerebras/DeepSeek become thin subclasses supplying `MODEL_SPECS`, the wire
  builder, a `_validate_provider_specifics` hook, and the finish map. OpenAI Responses and
  the embedding engine subclass `base` only (Responses stays standalone).
- **Why.** Collapses the largest duplication in the codebase (error ladder ×4, lifecycle ×4,
  choice-loop ×2 — all verified live) **and simultaneously resolves the "populated-but-unread
  matrix" defect** by making `LanguageModelSpec` the single enforcement source. One move,
  two systemic problems.
- **Feature-impact:** `keeps-all` (identical rejections, same messages via a display label).
- **Effort:** L. **Dependencies:** independent of Moves 3/5/6; naturally lands with Move 2
  and Move 7 (same package). **Supersedes** the "delete `message_roles`/`content_types`/
  `response_formats`" half of Move 4 (the gate makes them live). Do Move 5 (collapse N-output)
  *before or with* the chat base so the choice-loop simplifies to a single-output extract.

### Move 2 — Grow `providers/_client/` into a real client base (transport shell + envelopes + error `__init__`)
- **What.** Hoist a `BaseClient` (transport construction/ownership + cleanup, `close`,
  `_raise_for_status` parameterized by label+errors module, `_parse_response`) and the base
  `ResponseMetadata`/`APIResponse[T]` + header constants + `extract_response_metadata` into
  `providers/_client/`. OpenAI/DeepSeek re-use directly; Cerebras subtypes to add
  `rate_limit`. Put the shared error `__init__` bodies on the `_client/errors` classes.
- **Why.** `_client.py` 83% dup, `transport.py`/`headers.py` byte-identical (re-verified),
  error `__init__` ×3. The seam exists and is clean — it is just under-populated (D2–D4).
- **Feature-impact:** `keeps-all` (per-endpoint `by_alias`/`model_dump` variance passed as args).
- **Effort:** M. **Dependencies:** independent; complements Move 1.

### Move 3 — Replace the `JsonObject` AST with `pydantic.JsonValue`
- **What.** Type `JsonSchemaResponseFormat.json_schema: JsonValue`; delete
  `JsonObject`/`JsonArray`/`JsonEntry`/`JsonScalar`/`_parse_json_value`/`_json_value_to_builtin`/
  `to_builtin`/`parse` + the three `model_rebuild` calls; engines pass `json_schema` straight
  through (drop the `to_builtin()` + `cast`).
- **Why.** ~75 LOC + 3 public types for a value the engines flatten back to a builtin dict and
  `cast` to `pydantic.JsonValue` at the boundary anyway; no op produces it (C4/S1). Deep-freeze
  is the only property lost and nothing consumes it.
- **Feature-impact:** `drops-minimal` (json_schema requests still work; loses unused deep-immutability).
- **Effort:** M. **Dependencies:** independent. Confirm no stated "hashable/deep-frozen request"
  invariant first (none found).

### Move 4 — Resolve the `LanguageModelSpec` matrix (gate it, or delete the dead surface)
- **What.** Two coherent end-states, pick one: **(a)** if Move 1's gate is built, the three
  membership fields become live — then only delete `context_tokens` (Language + Embedding,
  which the gate still won't read); **(b)** if Move 1 is deferred, delete
  `context_tokens`/`message_roles`/`content_types`/`response_formats` and the
  `MessageRole`/`ContentType`/`ResponseFormatType` enums outright.
- **Why.** Today the matrix has the shape of a shared capability contract but half its fields
  have zero readers and it never crosses the engine boundary (C2). The status quo — shared-model
  shape, engine-private use, half unread — is the one arrangement that is *not* coherent.
- **Feature-impact:** `keeps-all` (gate) or `drops-minimal` (delete).
- **Effort:** M. **Dependencies:** **mutually exclusive with the gate half of Move 1** — do
  not both delete the fields *and* build the gate. `context_tokens` dies in both branches.

### Move 5 — Collapse N-output to a single output
- **What.** `LanguageModelResponse.output: LanguageModelOutput` (drop the `outputs` tuple,
  `min_length=1`, `LanguageModelOutput.index`, the per-engine `seen_indices` dedup + `sort`,
  `decode_output(output_index=…)`, and the `_output_text` index scan).
- **Why.** No `n`/`candidates` request field exists on any layer (re-verified); every response
  is exactly one output at index 0 (C3). The tuple/index/dedup/sort is permanently
  single-element defensive machinery presented as a many-outputs capability.
- **Feature-impact:** `drops-minimal` (re-add `n` only if multi-candidate becomes a product goal).
- **Effort:** S–M (models + 2 chat engines + decoding). **Dependencies:** simplifies Move 1's
  chat base and Move 6's `output_index` decision — sequence it just before/with Move 1.

### Move 6 — Slim the decoder layer
- **What.** Delete `PydanticDecoder` (let `TypeAdapterDecoder` accept a bare `type[T]` and wrap
  once — also removes the `TypeAdapter(...)` tax from the design's own examples); drop
  `ConstructorDecoder`'s container branch (route containers through `TypeAdapterDecoder`);
  remove `decode_output`'s `default`/`Missing`/`limit`/`_limit_value`; move `_normalize_text`'s
  single-quote stripping out of the shared path into the scalar/bool decode path (it silently
  mutates faithful text/JSON today — a real footgun).
- **Why.** One-decoder-per-concept; the sole production caller of `decode_output` passes no
  optional args (r1-04 #1–3, F5).
- **Feature-impact:** `keeps-all` (Pydantic + bool/scalar decoding preserved) / `drops-minimal`
  (unused literal-container path, unused fallback/limit).
- **Effort:** S–M. **Dependencies:** `output_index` removal coordinates with Move 5.

### Move 7 — Collapse provider `settings.py` + loaders
- **What.** One `EngineSettings(FrozenModel)` in `providers/_client/` (or `runtime`); all three
  providers reuse it. Factor a `build_http_engine(settings, *, client_factory, engine_factory)`
  helper; drop the loader's redundant `model not in MODEL_SPECS` pre-check (the engine ctor
  already raises `UnsupportedModelError`).
- **Why.** 4 byte-identical settings bodies + 3 near-identical loaders + a double-validated
  model check (F7/S4/D8, re-verified).
- **Feature-impact:** `keeps-all`. **Effort:** S. **Dependencies:** none (cutover tests don't
  pin `settings.py`/`loading.py` inventory); lands naturally with Moves 1–2.

### Move 8 — Naming & placement hygiene (cheap clarity)
- **What.** Rename `operations.py` → `runtime/requests.py` (kills the `operations`/`ops`
  collision; it authors only `runtime.models` types) via `pyseam`; disambiguate the two
  `loading.py`/`load_runtime` (e.g. generic → `compose_runtime`, public → `builtins.py`);
  hoist `_symbol_value`/`_require_text` into `ops/primitives.py` (verified duplicated:
  `_symbol_value` ×3, `_require_text` ×2); drop the `cast("ImplementationId", …)` no-ops by
  typing the registry keys.
- **Why.** Pure readability/placement; the shared homes already exist.
- **Feature-impact:** `keeps-all`. **Effort:** S. **Dependencies:** none.

### Ranking summary

| Rank | Move | Value | Conf | Effort | Feature-impact |
|---|---|---|---|---|---|
| 1 | `providers/_engine/` base + gate + chat base | very high | high | L | keeps-all |
| 2 | `providers/_client/` client base (transport/envelope/errors) | high | high | M | keeps-all |
| 3 | JSON AST → `pydantic.JsonValue` | high | high | M | drops-minimal |
| 4 | Resolve spec matrix (gate xor delete) | med-high | high | M | keeps-all / drops-minimal |
| 5 | Collapse N-output | med | high | S–M | drops-minimal |
| 6 | Slim decoder layer + `_normalize_text` footgun | med | high | S–M | keeps-all / drops-minimal |
| 7 | Collapse settings + loaders | med | high | S | keeps-all |
| 8 | Naming/placement hygiene | low-med | high | S | keeps-all |

**Sequencing.** Moves 3/5/6/7/8 are independent and can land in any order. Move 4 is gated by
the Move 1 decision (gate xor delete — never both). Move 5 should precede/accompany Move 1 so
the chat base is written against a single-output shape. Moves 1, 2, 7 share the
`providers/_engine/` + `providers/_client/` packages and are best done as one provider-tier
pass. **One unresolved product question spans this:** logprobs are sent + parsed but
`LanguageModelOutput` has no field to hold them (C1) — decide **close the loop** (add a
logprobs output field) or **cut the request side** before finalizing the contract; the same
owner call also settles the usage-consistency stance (PA-5: degrade to `usage=None` vs
hard-fail a valid completion).

---

## The architecture is already right here — keep

- **The dependency DAG and layer direction.** Acyclic; `Symbol` imported only by `ops.*`;
  `Function`/`Runtime`/decoders neither import nor return `Symbol`; `ops.*` is the sole
  Symbol-wrapping layer. This is the redesign's central invariant and it holds in live code
  (B-lens, re-verified). The legacy cutover that R1 flagged is **complete** — empty root,
  ambient runtime gone, prompts/backend gone.
- **`Symbol`'s operator table + immutability construction** (`__slots__` + `object.__setattr__`
  + raising setters, `_unwrap_operand`). Spec-mandated, uniform, I/O-free. Not over-engineering.
- **The strict/tolerant boundary.** Provider responses parsed by `TolerantModel`
  (`extra="allow"`), normalized contracts assembled as `FrozenModel` (`strict`, `extra="forbid"`).
  The model layer's strongest decision (C7). Keep.
- **Single-level discriminated unions** (`Message`/`Content`/`ResponseFormat` with
  `Field(discriminator="type")` at exactly one level). Correct per house style.
- **`ops.primitives._execute_language`** two-stage design — §12's predicted composition wrapper
  materialized across ~18 ops; keeps `Function` non-generic and `decode_output` independent.
- **Runtime core:** the `_resolve_engine` selection ladder (named→default→sole→ambiguous→
  unsupported), the named-instance model, allocation-free preflight + reverse-order failure
  cleanup, `execute()` releasing the lock before provider I/O, and narrow `execute`+`close`
  engine protocols (R5-5). Genuinely well-built; leave the core alone (minor lock/validation
  tidies R5-2/R5-3/R5-4 are optional polish, not structural).
- **The `providers/_client/` seam concept** and the client's own error hierarchy distinct from
  `runtime/errors.py` (engine is the single crossing point). Right idea — Move 2 just populates it.
- **Lazy provider loading + mechanism/policy loader split** keeping `import symai` inert
  (enforced by import-boundary tests). Preserve the discipline.
- **`TokenUsage`/`RateLimitMetadata`** as honest provider-fidelity public data (single-provider
  fields reflect real provider differences, not modeling waste — C6). Keep.
- **The cutover test-suite as an executable module-boundary spec** (`test_public_cutover.py`,
  `test_import_boundaries.py`). Keep driving code to green against it.
