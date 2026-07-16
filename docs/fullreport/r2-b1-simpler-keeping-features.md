# r2-b1 — One ranked, feature-preserving simplification plan

**Round 2 cross-cutting synthesis.** Merges every simplification proposal across the
Round-1 reports (`r1-01a`, `r1-01b`, `r1-02`, `r1-04`, `r1-05`, `r1-06`, `r1-07`) into a
single de-duplicated, ordered worklist. Every item was **re-verified against the live tree
at HEAD `84f703b`** ("refactor: remove legacy runtime and symbol surfaces" — the completed
cutover). Verdicts are adversarial: `keeps-all` / `drops-minimal:<what>` / `drops-real:<feature>`.

Regime: greenfield / pre-release / breaking rewrite. Optimize for best end-state; backward
compat and minimal-diff carry ~zero weight. The only constraint is **do not drop an intended
feature** (minimal/irrelevant loss is acceptable).

---

## Executive summary

1. **The cutover already resolved the four biggest R1 seeds.** Verified live: root
   `__init__.py` is 0 bytes; `_CURRENT_RUNTIME`/`current_runtime`/`NoActiveRuntimeError`,
   `static_context`/`dynamic_context`, `prompts.py`/`backend/`, and `jinja2`/`python-box`/
   `tomllib` code refs are **all gone**. R1 items F1, F2, S3, R5-1, P1, P2 → **RESOLVED**.
   Do not re-open them.
2. **The surviving win is the provider layer.** Infrastructure duplication (error→runtime
   mapper ×4, byte-identical `transport.py`/`headers.py`, `_client.py` ~83%, engine skeleton
   ×4, `settings.py` ×4 + loader clones) is **~400–500 LOC removable at low/no coupling risk**,
   plus ~120 more (cerebras↔deepseek chat-completions parse) at medium risk. All `keeps-all`.
   This is the safe-now bulk.
3. **Four contract sub-surfaces are dead or half-wired** and need a one-line product decision
   each: the `LanguageModelSpec` dead matrix (4 fields + 3 enums, **0 reads** confirmed),
   the `JsonObject`/`JsonArray`/`JsonEntry` AST (round-tripped to `pydantic.JsonValue`),
   N-output machinery (no `n` request field exists), and write-only logprobs.
4. **Decoder + value-layer polish** (`PydanticDecoder` redundant, `ConstructorDecoder`
   container branch prod-unused, `_normalize_text` single-quote footgun, `decode_output`
   over-parameterized, `execute_many` test-only) is small and mostly `drops-minimal`.
5. **Top trap (do NOT do):** deleting `operations.image_request`/`data_uri`/`ImageContent`/
   `ContentType.IMAGE`/`vision` when you delete the dead `content_types` field — the
   multimodal request path is **real and fully wired**, only unexposed through `ops.*`.
   Second trap: deleting any `TokenUsage` single-provider field or `RateLimitMetadata` —
   each is genuine provider-fidelity data with a producer.

---

## Ranked worklist

Merged-ID → source R1 findings. Verdict re-verified live. LOC = lines removed (net).
Risk = behavioral/coupling risk of the change. Group: **1** safe now · **2** needs a small
decision · **3** tempting-but-don't.

| # | Item (merged) | Sources | Verdict | LOC | Risk | Effort | Depends on | Grp |
|---|---------------|---------|---------|-----|------|--------|-----------|-----|
| W1 | Hoist `_symbol_value`/`_require_text`/`_string_tuple` guards → `ops/primitives.py` | F8, S5, D7, r1-04#4/#7 | keeps-all | ~20 | none | S | — | 1 |
| W2 | Remove no-op `cast("ImplementationId", "…")` ×4 in `loading.py` | seed/naming | keeps-all | ~4 | none | S | — | 1 |
| W3 | Base `ResponseMetadata`+`APIResponse` and header consts+`extract_response_metadata` → `_client/` (cerebras extends) | D3 | keeps-all | ~50 | low | S | — | 1 |
| W4 | Provider `errors.py` `__init__` bodies → shared `_client/errors` base | D4 | keeps-all | ~40 | low | S | W3 | 1 |
| W5 | `BaseClient` transport shell (`__init__`/cleanup, `close`, `_raise_for_status`, `_parse_response`) → `_client/` | D2 | keeps-all | ~100 | low | M | W3,W4 | 1 |
| W6 | Shared error→runtime mapper (`client_errors.*`→`runtime.errors.*`) replacing the 5-arm ladder ×4 | D1, PA-1 | keeps-all | ~110 | none | S–M | — | 1 |
| W7 | `BaseHttpEngine` skeleton (`__init__`/cleanup, `close`, `model`/`model_spec`, `_retry_after`, `_unsupported`, `_error_metadata`, PA-6 nits) | D5, PA-3, PA-6 | keeps-all | ~120 | low | M | W6 | 1 |
| W8 | One shared `HttpProviderSettings` + `build_http_engine` loader helper; drop redundant `MODEL_SPECS` preflight | D8, S4, F7, PA-6 | keeps-all | ~40 | low | S | — | 1 |
| W9 | Hoist structural `ModelSpec`/`ReasoningSpec` **shape** (not catalog) → `_client/models.py` | D9 (safe half) | keeps-all | ~30 | low | M | — | 1 |
| W10 | DRY `AssistantMessage`/`AssistantOutputMessage` shared fields (keep the two-type split) | C5 | keeps-all | ~6 | low | S | — | 1 |
| W11 | **Spec matrix**: delete dead `context_tokens`/`message_roles`/`content_types`/`response_formats` + `MessageRole`/`ContentType`/`ResponseFormatType` enums — **OR** build the data-driven capability gate (PA-2) making the matrix authoritative | F3, S2, C2, PA-2 | keeps-all | ~35 (delete) / ~60 net (gate) | low/med | M | W9 | 2 |
| W12 | `JsonObject`/`JsonArray`/`JsonEntry` AST → `pydantic.JsonValue` on `JsonSchemaResponseFormat.json_schema` | F4, S1, C4 | drops-minimal: deep-freeze of schema subtree (already discarded at boundary) | ~72 | low | M | — | 2 |
| W13 | Collapse N-output: `outputs: tuple` → single `output`; drop `index`/dedup/sort + `decode_output(output_index=)` | C3, D6-adj | drops-minimal: unreachable N>1 path | ~25 | low | S | W14 | 2 |
| W14 | Trim `decode_output` `default`/`limit`/`output_index` + `_limit_value` + `Missing` | F5, S6 | drops-minimal: spec'd-but-test-only conveniences | ~25 | low | S | — | 2 |
| W15 | Resolve write-only **logprobs**: cut request side (`logprobs`/`top_logprobs`/`logit_bias`+`LogitBias`) — OR add `logprobs` to `LanguageModelOutput` and map it | C1 | drops-minimal (cut) / feature-add (close) | ~30 (cut) | low/med | S–M | — | 2 |
| W16 | Decoder cleanups: fold `PydanticDecoder` into `TypeAdapterDecoder(bare type)`; narrow `ConstructorDecoder` to scalar/bool (route containers via TypeAdapter); move single-quote strip out of shared `_normalize_text` into the scalar/bool path | r1-04#1/#2/#3 | keeps-all (fixes TextDecoder footgun) | ~30 | low–med | S–M | — | 2 |
| W17 | `Function.execute_many`: remove (test-only) — OR reconcile signature with spec §6.1 | F6, r1-04#5 | drops-minimal: unused batch helper | ~15 | low | S | — | 2 |
| W18 | Right-size `_lifecycle_lock`: scope to `__enter__`+`close()` (drop from `execute`/`__exit__`), document pre-owner rationale | F9, S7, R5-2 | keeps-all | ~4 | med | S | — | 2 |
| W19 | Unify divergent `Runtime` vs `RuntimeConfig` alias/default validation via one shared validator (decide: reject outer whitespace everywhere) | R5-3 | keeps-all | ~10 | low | S | — | 2 |
| W20 | Engine-name uniqueness: enforce global-within-Runtime (add cross-map check) — OR amend FIXPLAN to "unique per capability" | R5-4 | drops-minimal: same-name-across-capabilities | ~5 | low | S | W19 | 2 |
| W21 | Usage-consistency: degrade to `usage=None` on arithmetic mismatch instead of `InvalidResponseError` (or relax exact-`==` to bounded) | PA-5 | drops-minimal: "usage is arithmetically perfect" guarantee; **gains** not discarding valid completions | ~10 | med | S–M | W7 | 2 |
| W22 | `chat_completions` parse skeleton (choices dedup/sort/finish-map) → shared `ChatCompletionsAdapter` (cerebras+deepseek only) | D6, PA-4 | keeps-all | ~120 | **med** | M | W7,W13 | 2 |
| W23 | Naming: `operations.py` vs `ops/`, two `loading.py` — rename for collision clarity | seed/naming | keeps-all | 0 (rename) | low | S | — | 2 |
| — | **KEEP `image_request`/`data_uri`/`ImageContent`/`ContentType.IMAGE`/`vision`** | trap | drops-real: **multimodal** | — | — | — | — | 3 |
| — | **KEEP `JsonSchemaResponseFormat`/`JsonObjectResponseFormat`** (replace only the AST representation, W12) | trap | drops-real: **structured output** | — | — | — | — | 3 |
| — | **KEEP all 9 `TokenUsage` fields + `RateLimitMetadata`** | C6, trap | drops-real: provider usage/rate-limit telemetry | — | — | — | — | 3 |
| — | **KEEP `_normalized_model_spec` / `MODEL_SPECS` / per-provider `chat.py`/`responses.py` per-provider** | D9 real half | drops-real: couples provider catalogs | — | — | — | — | 3 |
| — | **DO NOT extend `ChatCompletionsAdapter` to OpenAI Responses** | D6, PA-4 | drops-real: Responses wire fidelity | — | — | — | — | 3 |
| — | **KEEP `Symbol`'s ~40 operator dunders** | r1-04, r1-01a/b | keeps-all (spec §4.3 DSL) | — | — | — | — | 3 |

**RESOLVED (verified done — do not re-open):** F1/S3/R5-1 ambient runtime · F2/P1
static/dynamic context · S3/P2 `prompts.py`+jinja2/box/tomllib · root `__all__`/`__init__`.

---

## Verification notes (live tree, HEAD `84f703b`)

**RESOLVED items — confirmed gone:**
- `grep -rn "_CURRENT_RUNTIME\|current_runtime\|NoActiveRuntimeError\|static_context\|dynamic_context" symai/` → **NONE**.
- `symai/__init__.py` = **0 bytes**; `symai/prompts.py`, `symai/backend/` do not exist;
  `jinja2`/`python-box`/`tomllib` → **0 code refs**.

**Surviving items — confirmed still applicable:**
- **Dead spec fields (W11).** `grep` for `.message_roles`/`.content_types`/`.response_formats`
  attribute reads across `symai/` (non-test) → **zero**. `.context_tokens` reads are only
  `spec.context_tokens` in the four `_normalized_model_spec` builders reading the *client* spec
  while populating the normalized field — the normalized `.context_tokens` is never read back.
  `MessageRole`/`ContentType`/`ResponseFormatType` appear **only** inside the population code
  (`_ALL_MESSAGE_ROLES = tuple(MessageRole)`, `content_types=(ContentType.TEXT, …)`) and the
  `models.py` definitions — no enforcement reader.
- **JSON AST (W12).** `json_schema.to_builtin()` is consumed at exactly two boundaries
  (`openai/engines/responses.py` `schema=cast("JsonValue", …)`, `cerebras/…chat_completions.py`
  `body=cast("JsonValue", …)`), immediately flattened to a builtin dict cast to
  `pydantic.JsonValue`. No op produces the runtime `JsonSchemaResponseFormat`.
- **N-output (W13).** No `n`/`candidates`/`best_of` field on any request or client model.
  `decode_output` has one production caller (`ops/primitives.py`, no optional args), so
  `output_index` is always 0.
- **logprobs (W15).** `SamplingConfig.logprobs`/`top_logprobs`/`logit_bias` exist and are
  forwarded/validated by engines; `LanguageModelOutput` has **no** logprobs field.
- **Decoders (W16).** Ops use only `TextDecoder()` (text/reason/rank) and
  `ConstructorDecoder(bool)` (compare). `PydanticDecoder`, `TypeAdapterDecoder`, and the
  `ConstructorDecoder` container/int/float branches have **no production consumer**.
- **Provider dup (W3–W8).** `openai/client/transport.py` ≡ `deepseek/…` (modulo docstring);
  `headers.py` differ only by the transport import path; `_client.py` = 119/113/186 lines;
  the `rejected authentication` error arm appears in all 4 engines.
- **Runtime (W18–W20).** `_lifecycle_lock` acquired in `__enter__`/`__exit__`/`execute`/`close`.
  `Runtime._validate_aliases` checks `isinstance(str)`+non-empty but **no** strip; while
  `RuntimeConfig._validate_aliases` rejects `alias != alias.strip()` — so `Runtime({" chat ":e})`
  is accepted but the same alias via `RuntimeConfig` is rejected. No cross-map name-uniqueness check.

---

## Group 1 — Safe now (keeps-all, low risk)

Do these without a product meeting; they are internal dedup that changes no observable behavior.
**W1, W2, W3, W4, W5, W6, W7, W8, W9, W10.**

**Total safe-now removal: ~400–500 LOC**, dominated by the provider infrastructure layer
(W3–W9). All `keeps-all`. Coupling stays at the infrastructure line — no provider request/
response *schema* is merged (that is Group 3). Two mechanical sub-choices to standardize while
doing this, neither a design decision: pass `model_dump(by_alias=…)` as an argument (deepseek
omits `by_alias`), and standardize `_unsupported` on `-> Never` (deepseek already does).

## Group 2 — Needs a small decision

Each is a one-line product/design call; recommendations given.

- **W11 spec matrix — RECOMMEND delete.** Two mutually exclusive directions: (A) delete the 4
  dead fields + 3 enums (high-confidence, nothing reads them), keeping the hardcoded
  per-engine checks; or (B) the PA-2 data-driven gate — keep the matrix, make it the single
  enforcement source, delete ~60 lines of parallel hardcoded checks. **B is the better
  end-state only if you first re-audit each provider's true capability** (e.g. openai/cerebras
  currently set `message_roles = tuple(MessageRole)` incl. `DEVELOPER` — is that accurate?),
  and only pays off if a `Runtime.capabilities()` introspection API is wanted (none exists
  today; `Runtime` never exposes `model_spec`). Recommend **A now** (simpler, safe), rebuild as
  the gate later if introspection becomes a goal. Do W11 together with W9 — both touch
  `_normalized_model_spec`.
- **W12 JSON AST — RECOMMEND replace.** Decision: is deep-immutability of the `json_schema`
  subtree a required invariant? Today nothing consumes it (flattened at the boundary), so
  replace with `pydantic.JsonValue`. **Keep `JsonSchemaResponseFormat` itself** (Group 3 trap).
- **W13+W14 N-output & decode_output — RECOMMEND collapse+trim, together.** They are coupled via
  `output_index`. Collapsing forecloses provider `n>1` until re-added — acceptable given no
  request can ask for it. If multi-candidate is on the near roadmap, keep both and only trim
  `limit`/`_limit_value` (pure convenience, no spec-critical role).
- **W15 logprobs — RECOMMEND decide by product intent; default cut.** Nothing consumes logprobs
  in ops/decoding, so cutting is lower-risk. But the provider *clients already parse* returned
  logprobs (`TokenLogprob`/`TopLogprob`/`Logprobs` DTOs), so closing the loop is cheap if
  logprobs is an intended first-class result. Do not leave the half-state.
- **W16 decoders — RECOMMEND all three.** Folding `PydanticDecoder` and narrowing
  `ConstructorDecoder` are `keeps-all` (containers already covered by `TypeAdapter` per design
  §7). The `_normalize_text` single-quote fix is a genuine footgun fix (`TextDecoder("'Twas…'")`
  currently drops the leading apostrophe). **Caveat: `test_decoding.py` pins current behavior** —
  expect to update those tests.
- **W17 execute_many — RECOMMEND remove.** Test-only, no ops caller. r1-04 notes it is spec'd, so
  if kept, reconcile the nested `Sequence[Sequence[object]]` signature with design §6.1's flat form.
- **W18 lock — RECOMMEND tighter, carefully.** Scope the lock to `__enter__`+`close()` (the two
  pre-owner transitions); post-entry, `_require_owner_thread` already guarantees single-threaded
  access. Med confidence — the pre-ownership race is subtle; confirm the concurrency contract first.
- **W19 validation — RECOMMEND unify, reject outer whitespace everywhere.** Keep the `str`-type
  guard on the `Runtime` path (it alone accepts untyped mappings); keep `_validate_engine_identities`
  Runtime-only.
- **W20 name uniqueness — RECOMMEND enforce global.** Keeps "a name identifies one engine";
  update FIXPLAN to match whichever is chosen.
- **W21 usage-consistency — RECOMMEND degrade to `usage=None`.** Usage is telemetry, not the
  answer; discarding a valid completion over provider accounting arithmetic is a poor trade.
  Deliberate-strictness stance, so flag for the owner.
- **W22 chat-completions base — MEDIUM coupling, stop-here boundary.** Only cerebras↔deepseek;
  keep it generic over hooks (`finish_reasons`, `index_of`, `_extract_reasoning`), never pull the
  two providers' `chat.py` schemas into a shared module, and never extend to OpenAI Responses.
- **W23 naming — cosmetic.** Low priority; do last.

## Group 3 — Tempting-but-don't (would drop a real feature)

- **Multimodal path.** `operations.image_request` + `data_uri` build a `UserMessage` with
  `ImageContent`; the engines gate it via `model_spec.vision`. Fully wired, only unexposed
  through `ops.*`. When deleting the **dead** `content_types` field (W11), you MUST keep
  `ImageContent`, `ImageDetail`, `ContentType.IMAGE`, `vision`, `image_request`, `data_uri`.
  **This is the headline trap** — the dead field is named `content_types`, which invites
  deleting the whole image surface with it.
- **Structured output.** `JsonSchemaResponseFormat`/`JsonObjectResponseFormat` are a real,
  engine-wired request capability (cerebras builds `chat_api.JsonSchemaResponseFormat`; openai
  passes the schema). W12 replaces only the AST *representation* of `json_schema` — keep the
  response-format types.
- **Provider usage/rate-limit telemetry.** All 9 `TokenUsage` fields have ≥1 producer
  (`image_tokens`=cerebras, `cache_miss_prompt_tokens`=deepseek, `accepted`/`rejected_prediction_tokens`
  =cerebras). `RateLimitMetadata` is cerebras-only. These are public provider-fidelity data —
  do not prune "single-provider" fields.
- **Per-provider capability catalogs & wire schemas.** `_normalized_model_spec`, `MODEL_SPECS`,
  and `chat.py`/`responses.py` request/response models look duplicated but encode each provider's
  actual truth. Deduplicating them relocates divergence into a shared function that must change
  whenever any provider changes — false DRY. Keep split.
- **`Symbol` operator dunders.** The ~40 dunders are the complete, uniform realization of design
  §4.3; dropping any (bitwise/matmul/divmod/reflected) creates a contract hole for negligible LOC.

---

## Recommended order (definite)

**Phase 0 — RESOLVED, verify-only.** Confirm ambient runtime, static/dynamic context,
`prompts.py`, and root `__all__` are gone (done above). No work.

**Phase 1 — Group 1 provider + ops dedup (no decisions).** Bottom-up so each layer's base exists
before its consumers:
1. **W1** (ops helpers) and **W2** (cast no-op) — trivial, independent, land first.
2. **W3** (transport/headers base) → **W4** (errors `__init__` base) → **W5** (`BaseClient`
   shell). Leaf models/errors before the client that uses them.
3. **W6** (error mapper) → **W7** (`BaseHttpEngine`; it is the natural home for W6 and the PA-6
   nits) → **W9** (structural `ModelSpec`/`ReasoningSpec` shape) → **W8** (settings + loader helper).
4. **W10** (AssistantMessage DRY) — independent, land anytime in Phase 1.

**Phase 2 — Group 2 contract simplifications (one decision each).** Resolve **W11 with W9**
(both touch `_normalized_model_spec`). Then **W12** (JSON AST), **W13+W14 together** (N-output +
decode_output), **W15** (logprobs), **W16** (decoders), **W17** (execute_many).

**Phase 3 — Group 2 runtime (independent of providers).** **W19** (validation) → **W20** (names,
builds on the shared validator) → **W18** (lock right-sizing).

**Phase 4 — Group 2 provider behavior/structure.** **W21** (usage degrade) → **W22**
(chat-completions base; do after W7 exists and W13's single-output shape is settled).

**Phase 5 — polish.** **W23** (naming).

Throughout: honor every Group 3 keep. The single highest-leverage, lowest-risk block is
Phase 1 (~400–500 LOC, all `keeps-all`); the single most dangerous mistake is deleting the
multimodal surface alongside the dead `content_types` field.
