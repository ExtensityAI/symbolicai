# r2-b2 — One ranked, feature-preserving simplification plan

**Round 2 cross-cutting synthesis.** Merges every simplification proposal from the seven
Round-1 reports (`r1-01a`, `r1-01b`, `r1-02`, `r1-04`, `r1-05`, `r1-06`, `r1-07`) into a
single de-duplicated, ordered worklist. Every item re-verified against the **current** tree
(HEAD `84f703b`, the completed-cutover commit). Line numbers approximate; anchored by symbol
+ snippet. Read-only audit — nothing here is applied.

Regime: greenfield / pre-release / breaking rewrite. Optimize for best end-state; backward
compat and minimal-diff carry ~zero weight. "Keeps all intended features" is the bar —
only minimal/irrelevant feature loss is acceptable.

---

## Executive summary

1. **The R1 "biggest wins" are already done.** The cutover commit removed the ambient
   `ContextVar`/`current_runtime`/`NoActiveRuntimeError` triad, `Function.static_context`/
   `dynamic_context`, `prompts.py` + jinja2/box/tomllib, and root `__all__`. Verified: **zero
   production references** to any of them. R1 items F1, F2, S3, R5-1, P1, P2 are **RESOLVED** —
   do not re-open.
2. **Safe-now, feature-preserving simplification totals ~300 LOC** across ~15 low-risk items
   (ops-helper dedup, provider `settings.py`/`loading.py` boilerplate, the 4× error→runtime
   mapper folded into the *existing* `_client/errors.py`, transport/header/error base shapes
   folded into *existing* `_client` files, `PydanticDecoder` fold-in, the `ConstructorDecoder`
   container branch that contradicts the design, the `_normalize_text` quote-strip footgun, the
   divergent Runtime/RuntimeConfig validators). None needs a design decision.
3. **A ~350-LOC second tier (BaseClient / BaseHttpEngine / ChatCompletionsAdapter) is
   keeps-all but gated by a structural decision** R1 under-weighted: `test_public_cutover.py`
   **pins the exact file inventory** of `providers/_client/` (4 files) and every provider's
   `client/`+`engines/` dir. New shared files can only land in a *new, unpinned* `providers/_engine/`
   package or by editing that inventory test — a deliberate "where does the shared engine layer
   live" call, not a free refactor.
4. **Four data-model surfaces (spec matrix, logprobs, N-output, JSON AST) need a decision, not a
   cut** — and two of them are traps the design doc *blesses*: `decode_output`'s
   `output_index`/`default`/`limit` and `Function.execute_many` are **explicitly ratified**
   forward-looking surface (SYMBOL_REDESIGN §6.1, §7). Cutting them (R1 F5/F6/S6, C3-collapse)
   would drop spec'd capability. Keep.
5. **Top "don't do this" trap:** deleting `operations.image_request` / `data_uri` /
   `ImageContent` because they have **zero ops-layer callers** — the multimodal path is wired
   **end-to-end** (Cerebras builds `chat_api.ImageContentPart`; OpenAI gates on
   `model_spec.vision`; DeepSeek rejects images). It is a real, publicly-importable capability
   that merely lacks an ergonomic op. Removing it is a silent feature deletion.

---

## Status of the seven R1 report's proposals (de-duplicated)

Cross-report merges: the ops-helper dedup appears as **F8 = S5 = D7 = r1-04#4**; the spec
matrix as **F3 = S2 = C2 = PA-2**; the JSON AST as **F4 = S1 = C4**; `decode_output` params +
`execute_many` as **F5 = F6 = S6**; the provider error ladder as **D1 = PA-1**; engine infra as
**D5 = PA-3**; chat-completions parse as **D6 = PA-4**; settings/loading as **F7 = S4 = D8**.

| # | Item (merged R1 IDs) | Current status | Feature impact | LOC | Risk | Group |
|---|---|---|---|---|---|---|
| — | Ambient runtime triad (F1/S3/R5-1) | **RESOLVED** (0 prod refs) | keeps-all | — | — | done |
| — | `static/dynamic_context` (F2/S3/P1) | **RESOLVED** | keeps-all | — | — | done |
| — | `prompts.py`+jinja2/box/tomllib (S3) | **RESOLVED** | keeps-all | — | — | done |
| — | root `__all__` (R5-1) | **RESOLVED** | keeps-all | — | — | done |
| **1** | ops `_symbol_value`/`_require_text` → `primitives.py` (F8/S5/D7/r1-04#4) | STILL-OPEN | keeps-all | ~20 | low | **1** |
| **2** | `settings.py` 4× identical → shared base (F7/S4/D8) | STILL-OPEN | keeps-all | ~30 | low | **1** |
| **3** | `loading.py` preflight `not in MODEL_SPECS` drop (D8/PA-6) | STILL-OPEN | keeps-all | ~8 | low | **1** |
| **4** | error→runtime mapper 4× → free fn in *existing* `_client/errors.py` (D1/PA-1) | STILL-OPEN | keeps-all | ~96 | low | **1** |
| **5** | `transport.py`+`headers.py` base → *existing* `_client/{models,headers}.py` (D3) | STILL-OPEN | keeps-all | ~50 | low | **1** |
| **6** | provider `errors.py` `__init__` bodies → base in *existing* `_client/errors.py` (D4) | STILL-OPEN | keeps-all | ~30 | low | **1** |
| **7** | `PydanticDecoder` fold into `TypeAdapterDecoder(bare type)` (r1-04#1) | STILL-OPEN | keeps-all | ~10 | low | **1** |
| **8** | `ConstructorDecoder` container branch drop (contradicts §7) (r1-04#2) | STILL-OPEN | drops-minimal | ~10 | low | **1** |
| **9** | `_normalize_text` single-quote strip footgun (r1-04#3) | STILL-OPEN | keeps-all (fix) | ~2 | low | **1** |
| **10** | `Function._normalize_string_sequence` vs `_string_tuple` (r1-04#7) | STILL-OPEN | keeps-all | ~10 | low | **1** |
| **11** | divergent `Runtime` vs `RuntimeConfig` validators → shared (R5-3) | STILL-OPEN | keeps-all | ~10 | low | **1** |
| **12** | `_lifecycle_lock` over-applied in `execute()` — doc/scope (F9/S7/R5-2) | STILL-OPEN | keeps-all | ~2 | low | **1** |
| **13** | structural `ModelSpec`/`ReasoningSpec` shape → `_client/models.py` (D9-safe) | STILL-OPEN | keeps-all | ~20 | low-med | **1** |
| **14** | `_execution_metadata` wrapper drop + embedding `_build_request` parity (PA-6) | STILL-OPEN | keeps-all | ~12 | low | **1** |
| **15** | `AssistantMessage`/`AssistantOutputMessage` DRY (base+validator subclass) (C5) | STILL-OPEN | keeps-all | ~6 | low | **1** |
| **16** | `BaseClient` transport shell 83% dup (D2) | STILL-OPEN | keeps-all | ~100 | med | **2** |
| **17** | `BaseHttpEngine` infra skeleton 4× (D5/PA-3) | STILL-OPEN | keeps-all | ~150 | med | **2** |
| **18** | `ChatCompletionsAdapter` cerebras↔deepseek (D6/PA-4) | STILL-OPEN | keeps-all | ~120 | med | **2** |
| **19** | JSON AST → `pydantic.JsonValue` (F4/S1/C4) | STILL-OPEN | keeps-all* | ~75 | med | **2** |
| **20** | spec matrix: drop dead fields **or** data-driven gate (F3/S2/C2/PA-2) | STILL-OPEN | keeps-all* | ~60–100 | med | **2** |
| **21** | logprobs write-only: cut request side **or** close loop (C1/CON-02) | STILL-OPEN | drops-minimal* | ~S–M | med | **2** |
| **22** | N-output: keep (spec'd `output_index`) **or** collapse+drop index (C3/CON-03) | STILL-OPEN | see trap | ~S | med | **2** |
| **23** | `execute_many` signature flat(design) vs nested(code) reconcile (r1-04#5) | STILL-OPEN | keeps-all | ~S | low | **2** |
| **24** | usage-consistency `InvalidResponseError` → degrade/relax (PA-5) | STILL-OPEN | drops-minimal | ~S–M | med | **2** |
| **25** | engine name global-unique vs capability-scoped (R5-4) | STILL-OPEN | drops-minimal | ~S | low | **2** |
| **T1** | delete `image_request`/`data_uri`/`ImageContent`/`vision` | — | **drops-real** | — | — | **3** |
| **T2** | prune single-provider `TokenUsage` fields | — | **drops-real** | — | — | **3** |
| **T3** | cut `decode_output` `default`/`limit`/`output_index` (F5/S6) | — | **drops-spec'd** | — | — | **3** |
| **T4** | remove `Function.execute_many` (F6/S6) | — | **drops-spec'd** | — | — | **3** |
| **T5** | prune `RateLimitMetadata` (cerebras-only) (C6) | — | **drops-real** | — | — | **3** |
| **T6** | force-share `_normalized_model_spec`/`MODEL_SPECS` (D9-tension) | — | false DRY | — | — | **3** |
| **T7** | extend `ChatCompletionsAdapter` to OpenAI Responses (PA-4) | — | false DRY | — | — | **3** |

\* keeps-all conditioned on a decision — see detail.

**Safe-now (Group 1) removable ≈ 300 LOC.** Group 2 adds ~350 LOC of provider consolidation +
~135 LOC of data-model cleanup once the decisions land.

---

## Group 1 — safe now (keeps-all / design-aligned, low risk, no decision)

These need no product/design ruling and (crucially) **no new files in a pinned directory** —
each folds into an existing module. Verified live.

**1. ops helper dedup** — `def _symbol_value` appears verbatim in `ops/text.py:467`,
`ops/reason.py:154`, `ops/compare.py:177`; `def _require_text` in `ops/text.py:475`,
`ops/reason.py:162` (and inlined in `rank.py`/`compare.py`). `ops/primitives.py` already houses
`_execute_language` — hoist both guards there and import. Zero coupling.

**2. provider `settings.py`** — four byte-identical 5-field bodies (`ResponsesSettings`,
`EmbeddingSettings`, cerebras/deepseek `ChatCompletionsSettings`). Define one
`HttpProviderSettings(FrozenModel)` and alias. `settings.py` is **not** inventory-pinned
(verified: `test_deleted_production_tree_and_adapter_inventory` globs only `_client`, `client`,
`engines`), so this is unblocked.

**3. loading preflight** — every `loading.py` re-checks `if parsed.model not in MODEL_SPECS`
(`openai/loading.py:16,40`, `cerebras/loading.py:19`, `deepseek/loading.py:19`), duplicating the
engine `__init__`'s own `MODEL_SPECS[model]` → `UnsupportedModelError` (which also closes the
client on failure). Drop the preflight; keep the engine as the single source. `loading.py` is
not pinned.

**4. error→runtime mapper (highest LOC/risk ratio)** — all four engines carry the identical
5-arm `except client_errors.*` ladder (verified: 5 arms each). Every provider error subclasses
the shared `providers/_client/errors.py` base hierarchy (`ClientError → Transport/Response/API →
Auth/RateLimit`), so **one free function catching the base classes** replaces all four, zero
schema coupling. **Home it in the existing `_client/errors.py`** (a pinned-but-existing file) —
no new file, no inventory-test edit. ~96 LOC.

**5. transport/header base** — `openai/client/transport.py` ≡ `deepseek/client/transport.py`
verbatim modulo docstring; `headers.py` ≡ modulo import path (verified by `diff`). Cerebras is a
strict superset (adds `RateLimitState`/`x-ratelimit-*`). Move base `ResponseMetadata` +
`APIResponse[T]` into the existing `_client/models.py`, header constants + `extract_response_metadata`
into the existing `_client/headers.py`; Cerebras subclasses additively. No new `_client` file.

**6. provider error `__init__` bodies** — `APIError`/`ResponseError`/`TransportError.__init__`
(3 each, all providers) have identical bodies bar the provider prefix. Put them on the shared
`_client/errors.py` classes parameterized by a `provider` ClassVar. Enables #4's mapper to rely on
uniform `error.metadata`/`error.body`.

**7. `PydanticDecoder`** — production-unused (0 callers in `symai/`). It is a strict subset of
`TypeAdapterDecoder` (`model.model_validate_json` vs `TypeAdapter(model).validate_json` — verified
identical result+type). Not in the design's §7 decoder set (Text/Constructor/TypeAdapter). Fold in
by letting `TypeAdapterDecoder` accept a bare `type[T] | TypeAdapter[T]`; drop `PydanticDecoder`.
(Adjusts `test_decoding.py` + `tests/typecheck/function_decoding.py` — greenfield tests move with
the code.)

**8. `ConstructorDecoder` container branch** — the `list/tuple/set/dict` → `ast.literal_eval`
branch has **no op consumer** (only `ConstructorDecoder(bool)` in `ops/compare.py:104` is used)
and **directly contradicts** SYMBOL_REDESIGN §7 line 272 ("nested/container typing uses
`TypeAdapter`, not bare runtime classes"). Narrow `ConstructorDecoder` to scalars + `bool`; route
containers through `TypeAdapterDecoder`. Keep the scalar path — `ConstructorDecoder(int)` is a §7
example. drops-minimal + design-aligned.

**9. `_normalize_text` footgun** — strips one layer of wrapping **single** quotes for *every*
decoder: `_normalize_text("'Twas the night'")` → `Twas the night` (drops the intended apostrophe),
and it's asymmetric (`"…"` preserved). It never helps JSON decoders (JSON uses double quotes).
Keep `TextDecoder` whitespace-only; move quote-stripping into `ConstructorDecoder`'s scalar/bool
path where the `'value'` echo convention originates. Fixes a silent content mutation.

**10. Function string-sequence dup** — `function.py:83 _normalize_string_sequence` (rejects bare
`str`, checks every element is `str`) vs `operations.py:103 _string_tuple` (rejects bare `str`
only). Consolidate on the stricter one in a single home.

**11. divergent validators** — `Runtime._validate_aliases` (runtime.py:86) checks str-type +
non-empty but **not** whitespace; `RuntimeConfig._validate_aliases` (config.py:76) checks
whitespace but **not** str-type. So `Runtime(language_models={" chat ": e})` is accepted while the
same via `RuntimeConfig` is rejected. Factor a shared alias/default validator; decide the
whitespace rule once (recommend reject everywhere). Keep `_validate_engine_identities` Runtime-only
(it operates on live instances).

**12. lock scope** — `execute()` (runtime.py:184) takes `_lifecycle_lock` though post-entry the
owner-thread check (`_require_owner_thread`) already guarantees single-threaded access; the lock's
only irreplaceable job is the pre-entry no-owner window (`__enter__`/`close`). Either add a
one-line comment stating that scope, or drop the lock from `execute()`. Low-confidence right-sizing
— confirm the concurrency contract first; do not expand.

**13. structural `ModelSpec`/`ReasoningSpec` shape** — the frozen dataclass *shape*
(`context_tokens`, `response_tokens`, `reasoning`, `vision`) is identical across
`openai/client/responses.py`, `cerebras/client/chat.py`, `deepseek/client/chat.py`. Hoist the
container shape to the existing `_client/models.py`. **The `ReasoningEffort` enums and `MODEL_SPECS`
catalogs stay per-provider** — those are the provider's model truth (see T6). Touches 3 client
files (low-med effort).

**14/15. symmetry + model DRY** — drop the `_execution_metadata` wrapper (it's exactly
`self._error_metadata(response.metadata)`); give the embedding engine a `_build_request` for shape
parity. Make `AssistantOutputMessage` the base and `AssistantMessage` the validator-adding subclass
(the split is a real input/output lifecycle distinction — keep both types, DRY the fields).

---

## Group 2 — needs a small decision (keeps-all conditioned, or a fork with a clear default)

### Provider base extraction (16–18) — one structural decision, then keeps-all

`BaseClient` (D2, ~100 LOC), `BaseHttpEngine` (D5/PA-3, ~150 LOC), and `ChatCompletionsAdapter`
(D6/PA-4, cerebras↔deepseek only, ~120 LOC) are all keeps-all consolidations of verbatim
scaffolding — **but they need a home for genuinely new classes**, and
`test_public_cutover.py:279-312` pins `providers/_client/` to exactly `{__init__, errors, headers,
models}.py` and each provider's `client/`+`engines/` dirs to exact file sets. **The decision:**
create a new, unpinned `providers/_engine/` package (r1-07's proposal — allowed, the test doesn't
glob it) **or** relax the pinned `_client` inventory. Recommend the former. Coupling caveats
(verified): `BaseClient` must pass `by_alias`/`model_dump` mode as an argument (deepseek differs);
`ChatCompletionsAdapter` must stay generic over each provider's `Choice`/`Usage` (hooks, not shared
fields) and **must not** absorb OpenAI Responses (T7). Sequence after Group-1 items 4/5/6/13.

### 19. JSON AST → `pydantic.JsonValue` (keeps-all if deep-freeze isn't an invariant)

`JsonEntry`/`JsonArray`/`JsonObject` + `_parse_json_value`/`_json_value_to_builtin`/`to_builtin` +
3 `model_rebuild` calls (models.py:119-192) exist only to type `JsonSchemaResponseFormat.json_schema`
— which **no op produces** (ops always default to `TextResponseFormat`) and which both consuming
engines immediately flatten via `to_builtin()` and `cast("JsonValue", …)` to **`pydantic.JsonValue`**
— the type the client fields already use. The only net add over `pydantic.JsonValue` is deep
immutability, discarded at the boundary. **Decision:** is a deep-frozen / hashable request a stated
invariant? If no (nothing hashes/caches the request today), type the field `pydantic.JsonValue` and
delete ~75 LOC + 3 public types. Recommend replace.

### 20. spec matrix — drop dead fields **vs** make it authoritative (a real fork)

Verified live: `LanguageModelSpec.message_roles` / `content_types` / `response_formats` have
**0 reads** anywhere; normalized `context_tokens` has 0 reads (the `spec.context_tokens` hits read
the *raw client* spec while populating); `EmbeddingModelSpec.context_tokens` unread. The
`MessageRole`/`ContentType`/`ResponseFormatType` enums exist **only** to fill the dead fields.
`reasoning_fields` is read but **only as truthiness** (`if not self.model_spec.reasoning_fields`,
responses.py:177,272) — its members are never membership-tested. Two coherent directions, R1
reports split on them:

- **(a) Demote / drop** (r1-01a/b, r1-06): delete the 4 dead fields + 3 enums; collapse
  `reasoning_fields` to a `supports_reasoning` bool. ~60 LOC, lower risk. Leaves the hardcoded
  per-engine `_validate_request` checks as the (real) enforcement.
- **(b) Data-driven gate** (r1-07 PA-2): make the matrix the *single source of truth* — a shared
  `_gate_capabilities` reads `message_roles`/`content_types`/`response_formats`/`sampling_fields`,
  removing ~60 lines of parallel hardcoded checks; each engine keeps only a
  `_validate_provider_specifics` hook. Larger, but resolves the drift trap (today editing the matrix
  does nothing; editing a check silently disagrees).

Recommend **(b)** for the best end-state given the greenfield stage — it removes *more* code and
makes the matrix honest — **provided** the team wants `LanguageModelSpec` to be a public capability
contract (it is currently never exposed via `Runtime`). If not, take (a). Either way the orphan
enums stop being dead. This is the one item that most benefits from a deliberate ruling.

### 21. logprobs write-only (C1) — cut vs close

`SamplingConfig.logprobs`/`top_logprobs`/`logit_bias` are forwarded to all three providers and the
client DTOs **parse the returned logprobs** (verified: DTOs in all three `client/*.py`), but **no
runtime response model can hold them** (`LanguageModelOutput` has no logprobs field). The request
can ask; the response can never answer. Not in the design doc. **Decision:** cut the request side
(drop the 3 fields + `LogitBias` + validator + `SamplingField.LOGPROBS/TOP_LOGPROBS/LOGIT_BIAS`) or
close the loop (add `logprobs` to `LanguageModelOutput` + map it in 3 builders). Recommend **cut**
unless logprobs is a product goal — nothing in ops/decoding consumes it and providers may bill for it.

### 22. N-output (C3) — **leans keep** (this is a near-trap)

`LanguageModelResponse.outputs` is a `min_length=1` tuple with per-output `index`, dedup, and sort;
no `n`/`candidates` field exists on any request, so it always runs the single-element path.
**However** — SYMBOL_REDESIGN §7 explicitly specs `decode_output(..., output_index: int = 0)` with
the rule "output index selection is deterministic and raises `IndexError` when absent," and lists
output-index decoding as independently tested. Collapsing the response to one output would strip a
**ratified** surface. **Decision:** keep the tuple+index (recommended — it's cheap and spec'd), or,
only as a deliberate spec change, collapse *and* remove `output_index` together. Do **not** collapse
outputs while leaving `output_index` — that's an incoherent half-state.

### 23. `execute_many` signature reconcile (r1-04#5)

Design §6.1 shows `inputs: Sequence[object]` (flat); code implements `Sequence[Sequence[object]]`
(nested, splats `*values`). A caller following the doc gets a `TypeError`. Keep the nested code (it's
the more correct, multi-value form) and amend the doc, or vice-versa. Doc/code reconciliation, not a
behavior change. **Do not** remove `execute_many` — it's ratified (see T4).

### 24. usage-consistency policy (PA-5)

All engines raise `InvalidResponseError` on token-accounting inconsistency, **discarding a valid
completion** over billing metadata; DeepSeek is strictest (exact `cache_hit + cache_miss ==
prompt_tokens`, fragile to provider accounting changes). **Decision:** degrade to `usage=None` +
`logger.warning` (recommended — keep the answer), or at least relax exact-equality to bounded (`<=`)
and unify the policy across engines. drops-minimal (loses the arithmetic-perfect guarantee).

### 25. engine name uniqueness (R5-4)

Names are unique only *within* each capability map; `language_models={"x":a}` +
`embeddings={"x":b}` is accepted, so `engine="x"` means two engines by request type — contradicting
FIXPLAN §2 "globally unique within one Runtime." Enforce global uniqueness (recommended, add a
cross-map check to the shared validator from item 11) or amend the spec to "unique per capability."
drops-minimal (only same-name-across-capabilities).

---

## Group 3 — tempting-but-DON'T (looks like simplification, drops a real capability)

- **T1 — `image_request`/`data_uri`/`ImageContent`/`vision`/`ImageDetail`.** Zero ops-layer
  callers make these *look* dead. They are not: `ImageContent` is consumed **end-to-end** —
  `cerebras/engines/chat_completions.py:262-266` builds `chat_api.ImageContentPart`;
  `openai/engines/responses.py:172-173` gates on `model_spec.vision`; `deepseek` rejects images at
  :220/:285. `image_request`/`data_uri` are public, importable from `symai.operations`. **This is
  the wired-but-unexposed multimodal path — deleting it is a silent feature deletion.** The correct
  move is to *expose* it via an op, not remove it.
- **T2 — single-provider `TokenUsage` fields.** `cache_miss_prompt_tokens` (DeepSeek only),
  `image_tokens` + `accepted/rejected_prediction_tokens` (Cerebras only) are each populated by a
  real provider (verified). They look "mostly zero" but are honest provider-fidelity data. Pruning
  drops real returned information.
- **T3 — `decode_output` `default`/`limit`/`output_index` + `_limit_value`** (R1 F5/S6). Only the
  ops caller uses none of them — but SYMBOL_REDESIGN §7 **ratifies all three** with explicit decoder
  rules (default catches only `DecodeError`; limit is post-decode deterministic; sets pass through;
  index selection raises `IndexError`). This is forward-looking infrastructure the design asked for,
  not YAGNI. Keep.
- **T4 — `Function.execute_many`** (R1 F6/S6). Zero production callers, but §6.1 documents it as
  stable-order sequential execution. Ratified surface — keep (only reconcile its signature, item 23).
- **T5 — `RateLimitMetadata`** (cerebras-only, C6). Genuine provider difference, public return data.
  Keep; callers treat it as best-effort.
- **T6 — force-sharing `_normalized_model_spec` / `MODEL_SPECS`.** These *look* duplicated (same
  `LanguageModelSpec(...)` call ×3) but each encodes that provider's real capability truth. A shared
  builder would relocate the divergence into ~9 per-provider tuple args and couple provider catalogs.
  This is exactly the "dedup that couples unrelated schemas" the brief warns against. Keep per-provider.
- **T7 — extending `ChatCompletionsAdapter` to OpenAI Responses.** Responses is a genuinely different
  wire shape (status-based finish, heterogeneous output items, reasoning items). Sharing would be
  false DRY. Keep OpenAI Responses standalone.

---

## Recommended order

**Phase 0 (parallel, trivial, zero-coupling):** items 1, 7, 8, 9, 10, 12, 15 — value/decoding/model
layer, independent of each other and of providers.

**Phase 1 (provider boilerplate, unblocked):** items 2, 3 (settings/loading — not pinned), then 4
(error mapper into existing `_client/errors.py`), 5 (transport/header base into existing `_client`
files), 6 (error `__init__` base), 13 (ModelSpec shape into `_client/models.py`), 14. These fold into
existing files — no inventory-test churn. Item 4 must precede the Phase-3 engine base so engines can
adopt the mapper.

**Phase 2 (runtime consistency):** item 11 (shared validator), then decide 25 (name uniqueness) and
wire it into that validator.

**Phase 3 (data-model decisions — resolve the contract before the big engine refactor):** decide and
apply 19 (JSON AST), 20 (spec matrix — this gates whether Phase 4 grows a capability gate), 21
(logprobs), 22 (N-output — recommend keep), 23 (execute_many sig), 24 (usage policy). Doing these
first avoids refactoring the engines twice.

**Phase 4 (structural provider extraction — one home decision, then execute):** choose the
`providers/_engine/` home, then 16 (BaseClient) → 17 (BaseHttpEngine, adopting item 4's mapper) → 18
(ChatCompletionsAdapter, cerebras+deepseek only). If 20 chose the gate, land `_gate_capabilities`
here on the base.

**Rationale for the ordering:** trivial wins bank ~60 LOC immediately with no risk; the
existing-file provider folds (Phase 1) bank the largest low-risk LOC (~230) without touching the
pinned inventory; contract decisions (Phase 3) must precede the engine base (Phase 4) so the base is
built once against the final data model; the biggest, medium-risk extractions come last, on a settled
foundation.

---

## What is already good — keep (verified, do not touch)

- `Symbol`'s ~40 operator dunders — complete, uniform realization of §4.3; explicit hand-written
  dunders beat a metaclass loop for pyright. Keep.
- `ops.primitives._execute_language` — the correct shared two-stage seam (§12's predicted wrapper).
- Frozen/strict/`extra="forbid"` normalized contracts + single-level discriminated unions.
- The strict/tolerant client boundary (request DTOs `StrictModel`, response DTOs `TolerantModel`) —
  the model layer's strongest decision; `FrozenModel` never sees raw provider JSON (C7).
- Runtime selection ladder, named-instance model, allocation-free preflight + reverse-order teardown,
  `execute()` releasing the lock before provider I/O — the runtime core is sound (R5-5).
- Construction-cleanup discipline (`client.close()` + `add_note` on failed init) — correct and rare;
  centralize it (item 17), don't rewrite it.
- Per-provider `MODEL_SPECS` normalization at import behind `MappingProxyType`.
