# r2-a2 — System-level architecture synthesis

**Round 2 cross-cutting synthesis.** Inputs: r1-01a/01b (simplicity), r1-02 (duplication),
r1-03b (boundaries), r1-04 (value layer), r1-05 (runtime), r1-06 (contracts), r1-07
(adapters). Every structural claim below was **re-verified against the live tree** at
`84f703b` (legacy cutover done: empty root `__init__.py`, no `prompts.py`, no `backend/`,
no ambient `current_runtime`/`_token`). Line numbers approximate; anchors are symbol + snippet.
This report forms its own system view — it does not merely aggregate R1.

---

## System verdict (one paragraph)

**The decomposition is fundamentally sound and the layer *shape* is right — the defects are
not in the seams but in two places: one whole tier is *missing* and one whole tier is
*over-modeled*.** The vertical spine `Symbol ← ops.* → Function → Runtime → engines →
clients` (+ `ops.* → decoding`) is acyclic, correctly directed, and each of its six tiers
earns its keep: Symbol is contained (referenced only in `symbol.py` + `ops/*`), clients
never import `runtime`, engines are the sole provider→runtime crossing, and the
strict/tolerant + frozen-contract discipline is genuinely excellent. The **missing tier** is
a shared *provider adapter/client base*: `providers/_client/` exists but holds only leaf
helpers, and there is **no `providers/_engine/` at all**, so ~350–450 LOC of zero/low-coupling
scaffolding (error-mapping ×4, transport/headers byte-identical, `_client.py` ~83%, engine
lifecycle ×4, `settings.py` ×4) is kept in sync by hand across three providers — this is a
missing base class, *not* justified independence. The **over-modeled tier** is the contract
layer (`runtime/models.py`): it advertises four capabilities the system cannot actually
exercise — the `JsonObject`/`JsonArray`/`JsonEntry` AST (no producer, round-tripped to
`pydantic.JsonValue`), the `LanguageModelSpec` capability matrix (4 fields + 3 enums with
**zero reads**, verified live), the N-output tuple/index/dedup machinery (no `n` request
field exists), and write-only logprobs (sent + billed, but `LanguageModelOutput` can't hold
them). Alongside, the decoding module is oversized for what the system uses (only `TextDecoder`
+ `ConstructorDecoder(bool)` are consumed by ops; `TypeAdapterDecoder`/`PydanticDecoder` and
all of `decode_output`'s optional params have **no production caller**), and two name
collisions (`operations.py`↔`ops/`, two `loading.py`) tax every reader. Net: **fill the
missing provider base, make the capability matrix authoritative instead of decorative, delete
the AST and the dead N-output/logprobs surface, and disambiguate the names** — after which
the architecture is close to ideal.

**Layer count:** correct. Six vertical tiers, each pulling weight. The only tier that is
*thin* (not wrong) is the exec seam `Function` (94 LOC) + `operations.py` (request builders)
+ `ops.primitives` — three small modules that together form one "build+execute one request"
concern. That is acceptable (each is a real reuse unit), but `operations.py` is misplaced and
misnamed (see Move 4).

---

## Does each layer earn its existence?

| Tier | Earns keep? | Note |
|---|---|---|
| `Symbol` (value DSL) | **Yes** | Contained, immutable, spec-mandated operator family. Keep as-is. |
| `ops.*` (semantic ops) | **Yes** | Sole Symbol re-wrap point; `primitives._execute_language` is the right two-stage seam. |
| `decoding` (typed decode seam) | **Yes, but oversized** | Seam is correct; only 2 of 4 decoders + none of `decode_output`'s optional params are used. Slim (Move 6). |
| `operations.py` (request builders) | **Yes as a module — misplaced** | Pure `runtime.models` constructor; belongs in `runtime/`, name collides with `ops/`. Relocate (Move 4). |
| `Function` (configured callable) | **Yes, thin** | Real reuse unit across 18 ops; delegates to `operations` + `Runtime`. Marginal but fine. |
| `Runtime` (lifecycle/selection) | **Yes, strong** | Selection ladder, ownership, preflight+cleanup — the best-built part of the system. |
| `runtime/models.py` (contracts) | **Yes, over-modeled** | Core is sound; 4 sub-surfaces model capabilities the system can't exercise. Slim (Moves 2,3,7,8). |
| `runtime/config` + two `loading.py` | **Yes** | Config/registry split keeps runtime provider-agnostic. Keep the split; rename (Move 4). |
| `providers/*/client` (raw HTTP) | **Yes** | Faithful bindings, never import runtime. But heavily duplicated → missing base (Move 1). |
| `providers/*/engines` (adapters) | **Yes** | Sole crossing point. But ~40% scaffolding + no shared base (Move 1). |
| `providers/_client` (shared toolkit) | **Yes — under-populated** | Right idea, only holds leaf helpers; should absorb transport/base-client/settings (Move 1). |
| `providers/_engine` (shared adapter base) | **MISSING** | The keystone gap. Should exist (Move 1). |

---

## The unifying theme: aspirational contract surface

Four of the highest-value findings across all R1 lenses are the *same architectural mistake*
seen from different angles — the type system **advertises capabilities the running system
cannot exercise**:

- **JSON-schema structured output** — `JsonSchemaResponseFormat` exists, but no op ever emits
  it (`language_request` always defaults to `TextResponseFormat()`), and its `JsonObject` AST
  is flattened back to `pydantic.JsonValue` at the client boundary.
- **Capability introspection** — `LanguageModelSpec` reads like a public capability contract,
  but `Runtime` never exposes it and 4 fields + 3 enums have **zero readers** (verified live).
- **N candidate outputs** — `LanguageModelResponse.outputs` is a `tuple` with `index`/dedup/
  sort, but **no request can ask for N>1** (no `n`/`candidates`/`best_of` field exists anywhere).
- **Logprobs** — `SamplingConfig` sends `logprobs`/`top_logprobs`/`logit_bias` (providers even
  *bill* for it), but `LanguageModelOutput` has no field to return them.

This is worth naming because the *fix* is one decision repeated four times: **for each, either
wire the capability end-to-end so it becomes real, or delete the half that dangles.** Given the
"keep intended features, minimal loss acceptable" mandate and that nothing downstream consumes
any of the four, the default is *delete the dangling half* — except the capability matrix,
which is uniquely worth making real because doing so also removes ~60 LOC of parallel hardcoded
enforcement (Move 2).

---

## Top structural moves (ranked by value × confidence)

| # | Move | Feature impact | Effort | Depends on | Value×Conf |
|---|------|----------------|--------|-----------|-----------|
| 1 | Fill the missing provider base tier (`_engine/` + expand `_client/`) | keeps-all | L | — | **highest** |
| 2 | Make `LanguageModelSpec` authoritative via a data-driven capability gate | keeps-all | M | **Move 1** | high |
| 3 | Delete the `JsonObject`/`JsonArray`/`JsonEntry` AST; type `json_schema` as `pydantic.JsonValue` | drops-minimal | M | — | high |
| 4 | Kill the two name collisions: `operations.py`→`runtime/requests.py`; rename the generic loader | keeps-all | S | — | high |
| 5 | Consolidate `settings.py` ×4 + drop redundant loader model-check + drop no-op `ImplementationId` casts | keeps-all | S | Move 1 (settings part) | med-high |
| 6 | Slim the decoding surface (fold `PydanticDecoder`; trim `decode_output`; fix `_normalize_text` footgun) | drops-minimal / keeps-all | S-M | — | med |
| 7 | Resolve logprobs write-only hole (cut request side, or close the loop) | drops-minimal | S / M | — | med |
| 8 | Collapse dead N-output machinery to a single output | drops-minimal | S | Move 6 | med-low |

---

### Move 1 — Fill the missing provider base tier (the keystone)

**What.** Create the shared adapter/client tier the layering already implies but never built:
1. **Expand `providers/_client/`** to absorb the byte-identical/near-identical plumbing:
   base `ResponseMetadata` + generic `APIResponse[T]` (from `transport.py`, verbatim across
   openai≡deepseek, cerebras a strict superset); the `REQUEST_ID_HEADER`/`RETRY_AFTER_HEADER`
   constants + base `extract_response_metadata` (from `headers.py`); a `BaseClient` transport
   shell (`_client.py` is ~83% identical cerebras↔deepseek — transport ownership, cleanup,
   `close`, `_raise_for_status`, `_parse_response`); the shared `APIError`/`ResponseError`/
   `TransportError` `__init__` bodies; and one `EngineSettings` model (see Move 5).
2. **Introduce `providers/_engine/`** holding an `Adapter` base: `__init__`/construction-cleanup
   (the `except BaseException: client.close(); add_note(...)` block, verbatim ×4), `close`,
   `model`/`model_spec` properties, `_retry_after`, `_unsupported`, `_error_metadata`, and the
   **error-mapping ladder** (the 5-branch `except` that translates `client_errors.*` →
   `runtime.errors.*`, copied ×4 — safe because every provider error already subclasses the
   shared `_client/errors.py` base). Plus a `ChatCompletionsAdapter(LanguageAdapter)` carrying
   the cerebras/deepseek choices-loop + `_FINISH_REASONS`-driven finish mapping (~55% identical).

**Why.** This is the definite answer to the brief's question — **the 3× provider duplication is
a missing base class, not justified independence.** Verified: the duplicated parts (error map,
transport, headers, lifecycle, settings) reference **no provider schema**; the genuinely
independent parts (`chat.py`/`responses.py` wire schemas, `_normalized_model_spec`, `MODEL_SPECS`
catalogs, and OpenAI's Responses parse path) stay per-provider. The line is clean and both r1-02
(D9) and r1-07 (PA-4) draw it in the same place. Concrete residue removed: ~350–450 LOC.

**Boundary to respect (do NOT cross).** Keep OpenAI **Responses** standalone (status-based
finish, heterogeneous output-item walk, "reasoning ⇒ one assistant message" — a real different
wire contract). Never merge `_normalized_model_spec` or `MODEL_SPECS` — those *are* each
provider's capability truth; sharing them would couple unrelated model catalogs.

**Feature impact:** keeps-all. **Effort:** L. **Deps:** none — foundational; enables Moves 2, 5.

---

### Move 2 — Make the capability matrix authoritative (resolves the R1 fork)

**What.** In the Move-1 `Adapter` base, add a data-driven `_gate_capabilities(request)` that
consults `spec.message_roles` / `content_types` / `response_formats` / `sampling_fields` /
`reasoning_*` as the **single source of truth** (`if message.role not in spec.message_roles:
_unsupported(...)`, etc.). Each engine then keeps only a `_validate_provider_specifics` hook for
rules a membership-tuple genuinely can't express (DeepSeek user-id regex, "temperature ignored
unless thinking disabled", stop-count bounds, `top_logprobs requires logprobs`, `max_tokens ≤
response_tokens`).

**Why / my position (differs from r1-06).** r1-06 (C2) and r1-01 (F3) recommend *deleting* the 4
dead fields + 3 enums; r1-07 (PA-2) recommends *making them authoritative*. **I side with making
them authoritative — but only if Move 1 lands.** Verified live: `message_roles`, `content_types`,
`response_formats`, `context_tokens` have **zero reads**, and `Runtime` never exposes `model_spec`;
meanwhile each `_validate_request` hardcodes the same facts imperatively (DeepSeek rejects images
in *three* places). Deleting the fields removes the *pretense* but leaves the parallel hardcoded
checks — the actual drift trap. The gate removes the parallelism *and* gives the fields a real
consumer; the membership-tuple shape is exactly right for it. This is strictly the better
end-state, so it earns the fields' keep. **Fallback if Move 1 is not done:** delete
`context_tokens`/`message_roles`/`content_types`/`response_formats` + the `MessageRole`/
`ContentType`/`ResponseFormatType` enums (they discriminate nothing — the message/format models
use `Literal[...]`). Do not ship the status quo (populated-but-unread) either way.

**Feature impact:** keeps-all (identical rejections, single source). **Effort:** M. **Deps:**
**Move 1** (the gate needs a shared base to live in; without it you'd re-duplicate the gate ×4).

---

### Move 3 — Delete the `JsonObject`/`JsonArray`/`JsonEntry` AST

**What.** Remove `JsonEntry`/`JsonArray`/`JsonObject`/`JsonScalar`/`JsonValue`/`_parse_json_value`/
`_json_value_to_builtin`/`to_builtin`/`parse` and the three `model_rebuild` calls. Type
`JsonSchemaResponseFormat.json_schema: JsonValue` (pydantic's own, already imported in the
consuming engines). Engines pass `response_format.json_schema` straight through — no `to_builtin()`,
no `cast`.

**Why.** Verified live: **no production code constructs it** (grep for `JsonObject(`/`JsonObject.parse`/
`JsonSchemaResponseFormat(` in `symai/` non-test → nothing; the ops layer always defaults to
`TextResponseFormat`), and at the one boundary that reads it, engines do
`cast("JsonValue", …​.to_builtin())` to `pydantic.JsonValue` — flattening the AST back to a plain
dict. The only property the AST adds over validated `pydantic.JsonValue` is deep-immutability,
which is discarded at that boundary and consumed by nothing (~70 LOC + 3 public exports for a
round-trip to nowhere). `validate_unique_keys` can't fire from `.parse(Mapping)`. If a hard
deep-freeze/hashable-request invariant is ever required, prefer a single `frozendict` wrapper or a
canonical JSON string — not a three-model AST.

**Feature impact:** drops-minimal (json_schema requests still validated JSON; only unused
deep-immutability lost). **Effort:** M. **Deps:** none.

---

### Move 4 — Kill the two name collisions

**What.** (a) Move `symai/operations.py` → `symai/runtime/requests.py` (co-locate request
*builders* with the request *models* they construct — its only import is `runtime.models`), ending
the `operations.py` ↔ `ops/` shadowing. (b) Rename the generic `runtime/loading.py::load_runtime` →
`build_runtime` (or `load_runtime_from_registry`) so the public `symai/loading.py::load_runtime`
no longer needs `as _load_runtime`; optionally rename `symai/loading.py` → `symai/builtins.py`.

**Why.** Two sibling surfaces both called "operations" and two modules both named `loading.py`
both exporting `load_runtime` force constant reader disambiguation and an alias hack. The layer
*splits* are correct (r1-03b verified runtime stays provider-agnostic); only the *names* mislead.
Pure cohesion/clarity win.

**Feature impact:** keeps-all. **Effort:** S. **Deps:** none.

---

### Move 5 — Consolidate settings + drop redundant checks and no-op casts

**What.** (a) Replace the four byte-identical `settings.py` models (`ResponsesSettings`,
`EmbeddingSettings`, two `ChatCompletionsSettings`) with one `EngineSettings(FrozenModel)` in
`providers/_client/`. (b) Drop the loader preflight `if parsed.model not in MODEL_SPECS: raise
UnsupportedModelError` — the engine `__init__` already raises that exact error (and cleans up the
client). (c) Drop the no-op `cast("ImplementationId", "openai:responses")` calls in
`symai/loading.py`: `ImplementationId` is `Annotated[str, BeforeValidator(...)]`, so the cast is a
runtime no-op and cosmetic-only; the values are validated when indexed by the loader anyway.

**Why.** 4 identical 5-field models + 3 near-identical loader bodies + a double-validated
"is this model supported" invariant + 4 casts that do nothing. All low individual weight, broad
reach.

**Feature impact:** keeps-all. **Effort:** S. **Deps:** the settings model rides on Move 1's
`_client/` expansion, but (b) and (c) are independently doable.

---

### Move 6 — Slim the decoding surface

**What.** (a) Let `TypeAdapterDecoder` accept a bare `type[T]` and wrap it once, then **delete
`PydanticDecoder`** (a strict subset — `model.model_validate_json` ≡ `TypeAdapter(model).validate_json`).
(b) Trim `decode_output` to `(response, decoder)` — delete `default`/`Missing`/`limit`/`_limit_value`/
`output_index`; the **sole production caller** (`ops/primitives._execute_language`) passes none.
(c) Narrow `ConstructorDecoder` to scalar + `bool`; route containers through `TypeAdapterDecoder`
(the `list/tuple/set/dict` `ast.literal_eval` branch has no op consumer and clashes with
`TypeAdapter`'s JSON grammar). (d) Keep `TextDecoder` faithful: move the single-quote stripping
out of the shared `_normalize_text` into the scalar/bool path — today it silently mutates faithful
text (`'Twas → Twas`) for every decoder and only ever helps the bool/scalar echo convention.

**Why.** Verified live: ops instantiate only `TextDecoder()` and one `ConstructorDecoder(bool)`;
`TypeAdapterDecoder`/`PydanticDecoder` have **zero non-test references**. The decoding module is
carrying a forward-looking, user-facing decoder set that the system itself never exercises. **Honest
caveat:** `output_index`/`default`/`limit` are named in `SYMBOL_REDESIGN.md`, so trimming them is a
"does forward-looking surface earn keep before the second caller" judgment, not a clear cut — the
unambiguous wins here are the `PydanticDecoder` fold and the `_normalize_text` footgun (the latter is
keeps-all and a correctness improvement). Note `output_index` interacts with Move 8.

**Feature impact:** drops-minimal (test-only conveniences) + keeps-all (the `_normalize_text` fix).
**Effort:** S-M. **Deps:** none (but sequence with Move 8).

---

### Move 7 — Resolve the logprobs write-only hole

**What.** `SamplingConfig.logprobs`/`top_logprobs`/`logit_bias` are forwarded to all providers and
the client DTOs even parse the returned logprobs, but `LanguageModelOutput` has no field to hold
them. Either **close the loop** (add `logprobs: tuple[TokenLogprob, ...]` to `LanguageModelOutput`,
map it in the three `_output` builders) or **cut the request side** (drop the three `SamplingConfig`
fields + `LogitBias` + `validate_unique_logit_bias_tokens` + the `SamplingField.LOGPROBS/…` members
+ the client-side logprobs DTOs).

**Why.** A request/response coherence hole in the public type system, and providers may bill
differently when `logprobs=true` — so it is not free to send. This is the one finding where the
intended-feature question genuinely gates direction; **flag for the owner.** Default (nothing in
ops/decoding consumes logprobs): cut.

**Feature impact:** drops-minimal (cut). **Effort:** S (cut) / M (close). **Deps:** none.

---

### Move 8 — Collapse the dead N-output machinery

**What.** No `n`/`candidates`/`best_of` field exists on any request (verified). Collapse
`LanguageModelResponse.outputs: tuple[...]` → a single `output: LanguageModelOutput`; drop
`LanguageModelOutput.index`, the per-choice dedup/sort, and `decode_output`'s `output_index` scan.
(Or add `n: int = 1` and make the machinery real — but nothing wants N>1.)

**Why.** Dead many-outputs modeling across `models.py` + both chat engines + `decoding.py`, always
exercising the single-element path. Lower urgency than the others because the dedup/sort also
defensively validates provider payloads — but that defends against a case the request layer cannot
produce.

**Feature impact:** drops-minimal. **Effort:** S. **Deps:** Move 6 (shares `output_index`).

---

## Missing abstractions (named)

1. **Shared provider adapter base (`providers/_engine/`)** — the keystone gap. Move 1.
2. **`BaseClient` transport shell + base `ResponseMetadata`/`APIResponse` + shared error `__init__`
   bodies** — `providers/_client/` should own these; today they're re-copied per provider. Move 1.
3. **A capability gate** — the mechanism that makes `LanguageModelSpec` the source of truth it
   pretends to be (lives in the Move-1 base). Move 2.
4. **A genuinely typed implementation-id** — `ImplementationId` is `Annotated[str, …]` and the
   loader `cast("ImplementationId", …)` calls are no-ops. This is a *mis-applied* abstraction more
   than a missing one: either make it a real `NewType`-style distinct type used consistently, or
   drop the cosmetic casts (Move 5c). Recommend dropping — the `BeforeValidator` already normalizes
   at the validation boundary.

---

## The architecture is already right here — keep

- **`Symbol` containment + immutability + operator DSL.** `__slots__` + `object.__setattr__` +
  raising `__setattr__`; ~40 uniform, I/O-free dunders realizing SYMBOL_REDESIGN §4.3;
  referenced *only* in `symbol.py` + `ops/*`. The load-bearing invariant of the whole design, and
  it holds. Do not "simplify" the dunder family.
- **The client↔engine seam** (the memory's key concern). Clients import only `_client`, never
  `runtime`; engines are the **sole** provider→runtime crossing; no cross-provider imports; runtime
  stays provider-agnostic (no `runtime/*` imports `providers`). Faithful-binding-client /
  crossing-point-engine is real and verified.
- **Acyclic graph; `symbol.py` and `runtime/models.py` are true leaves** (stdlib + pydantic only).
- **Runtime core.** Selection ladder (named→default→sole→ambiguous→unsupported), single-owner-thread
  ownership, allocation-free preflight + reverse-order failure cleanup, `execute()` dropping the lock
  *before* provider I/O, narrow `execute`+`close` protocols with no handle leaks. The strongest tier.
- **Strict/tolerant boundary.** Request DTOs → `StrictModel(extra="forbid")`; response DTOs →
  `TolerantModel(extra="allow")`; internal normalized `FrozenModel` never parses raw provider JSON.
  Correctly placed — do not move it.
- **Single-level discriminated unions** (`Message`, `Content`, `ResponseFormat`) — right shape, avoids
  nested-discriminator pitfalls.
- **`ops.primitives._execute_language`** — the two-stage composition seam that validates §12's bet
  (Function stays non-generic, `decode_output` stays independent, ops own the collapse-into-`Symbol`).
- **Lazy provider loading** — thin `_load_*` thunks + `__getattr__` provider facades + the
  import-boundary test keeping `import symai.loading` inert.
- **Per-provider wire schemas + `_normalized_model_spec` + `MODEL_SPECS` catalogs, and OpenAI's
  standalone Responses parse path** — correctly separate. This is the *right* independence; merging it
  would be false DRY. The Move-1 base must stop exactly at this line.
- **Construction-cleanup discipline** (`add_note` on cleanup failure without masking the original) —
  uniform and correct; centralize it (Move 1), don't rewrite it.

---

## How this differs from the R1 reports

- **Ranks the shared provider base as the keystone and makes Move 2 depend on it.** R1 treated the
  base, the matrix, and the gate as separate findings; the synthesis point is that resolving the
  capability matrix *well* (a gate, not deletion) is unlocked by, and should follow, Move 1.
- **Takes a definite side on the matrix fork:** make it authoritative (r1-07) rather than delete it
  (r1-06/r1-01), because the gate removes the parallel hardcoded enforcement that deletion leaves behind.
- **Names the unifying theme** — four independent R1 findings (AST, matrix, N-output, logprobs) are one
  mistake: *contract surface for capabilities the system can't exercise.* One decision, four applications.
- **Grounds "decoding is oversized" in live usage** (only `TextDecoder` + `ConstructorDecoder(bool)`
  consumed by ops; `decode_output`'s single caller passes no options) rather than treating each decoder
  finding in isolation.
- **Position on `operations.py`: relocate, not delete** — it earns its keep as a cohesive request-builder
  module; only its name and placement are wrong.
