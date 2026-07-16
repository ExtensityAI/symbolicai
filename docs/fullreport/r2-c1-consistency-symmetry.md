# r2-c1 — Consistency & Symmetry (cross-cutting)

> **Historical snapshot terminology.** Configuration references below describe commit `84f703b`.
> The final target removes configured `default_*` fields, uses `EngineConfig`, and retains only
> sole-engine unnamed resolution.

**Lens:** where the codebase is *asymmetric but should be symmetric* (or where an
asymmetry is justified and should be documented as such). Synthesis round over the
whole tree, cross-checking `r1-07` (adapters), `r1-02` (duplication), `r1-08` (API
surface), `r1-05` (runtime), `r1-06` (contracts). Every claim below was re-verified
against the **live** tree (post-cutover, commit `84f703b`); anchors are symbol +
snippet, line numbers approximate.

State confirmed: root `symai/__init__.py` is **0 bytes**; `ops/__init__.py` re-exports
five namespaces; `current_runtime`/`NoActiveRuntimeError`/`_CURRENT_RUNTIME` have **0
refs**; the ambient `ContextVar` apparatus is gone from `runtime.py`. So the R1 cutover
seeds are resolved — this report is about the *surviving* symmetry gaps.

---

## Executive summary

1. **Token-usage consistency is the sharpest real asymmetry.** The same class of
   check (provider token accounting) is enforced at **four different strictness
   levels** across engines, DeepSeek uniquely uses a **fragile exact-equality**
   (`cache_hit + cache_miss == prompt_tokens`), and all four **discard a valid
   completion** by raising `InvalidResponseError`. This is the one divergence that is a
   live bug-in-waiting, not just cosmetics. (SYM-1)
2. **The capability matrix is declared symmetrically but enforced asymmetrically.**
   `LanguageModelSpec.sampling_fields`/`message_roles`/`content_types`/`response_formats`
   are populated by all three engines, but only OpenAI's temperature/top_p checks ever
   read `sampling_fields`; every other reject is hardcoded, and DeepSeek even
   *double-encodes* (matrix omits a field **and** a hardcoded check rejects it). The
   divergence in *what* each provider supports is justified; the divergence in *how* it
   is enforced is a drift trap. (SYM-2)
3. **`Runtime.__init__` and `RuntimeConfig` validate the same invariant with different
   rules** — outer-whitespace, str-type, and check-order all differ, so
   `Runtime(language_models={" chat ": e})` is accepted while the equivalent
   `RuntimeConfig` is rejected. A silent contract seam between the two construction
   paths. (SYM-3)
4. **The scaffolding is near-perfectly symmetric and that's the problem** — the
   5-branch error ladder, `_retry_after`, `__init__`/cleanup, `close`, `model_spec` are
   verbatim ×4; the *asymmetries that remain* inside that scaffolding
   (`_unsupported` return type, `_error_metadata` optionality, `_build_request`
   presence, `_execution_metadata` wrapper) are unmotivated drift a shared base erases.
   (SYM-5/6)
5. **ops are signature-symmetric (genuinely good) but validation-asymmetric.**
   `(runtime, source, …, *, engine=None) -> Symbol` holds everywhere; but
   `_symbol_value` is copy-defined ×3, `_require_text` ×2, and `rank`/`compare` inline
   drifted variants instead of calling the helper. (SYM-7)

Overall read: the layer *contracts* and the *op surface* are symmetric by design and
should be kept. The asymmetries cluster in two honest places (usage-consistency policy;
matrix-declared vs code-enforced capability) plus a residue of unmotivated method-shape
and naming drift that a shared adapter base + a shared ops-helper module remove for
free.

---

## Ranked divergences

| ID | Divergence | Class | Conf | Impact | Effort |
|----|------------|-------|------|--------|--------|
| SYM-1 | Usage-consistency: 4 strictness levels; DeepSeek exact-equality; all discard valid completion | **BUG-IN-WAITING** | high | med | S–M |
| SYM-2 | Capability matrix declared symmetrically, enforced asymmetrically (matrix vs hardcode, double-encoded) | **BUG-IN-WAITING** | high | high | M |
| SYM-3 | `Runtime.__init__` vs `RuntimeConfig` alias/default rules diverge (whitespace, str-type, order, label) | **BUG-IN-WAITING** | high | med | S |
| SYM-4 | Engine names unique per-capability, not globally (spec says global) | **BUG-IN-WAITING** (low) | high | low | S |
| SYM-5 | Error-mapping ladder verbatim ×4; embedding diverges only in message wording | COSMETIC (dedup) | high | med | S |
| SYM-6 | Engine method-shape drift: `_unsupported` return, `_error_metadata` optionality, `_build_request` presence, `_execution_metadata` wrapper | COSMETIC | high | low | S |
| SYM-7 | ops `_symbol_value` ×3 / `_require_text` ×2 + inline drift in `rank`/`compare` | COSMETIC (dedup) | high | med | S |
| SYM-8 | logprobs write-only: all 3 forward, none reads back — **symmetric** but incoherent | BUG-IN-WAITING / decide | high | med | S–M |
| SYM-9 | `by_alias` serialization: DeepSeek omits it | **JUSTIFIED** (no aliases) but fragile | high | low | S |
| SYM-10 | Naming: `operations.py` vs `ops/`; two `load_runtime`; `cast("ImplementationId")` no-op; `FrozenModel` ≡ `StrictModel` | COSMETIC | high | low | S |
| SYM-11 | finish-reason: status-walk (OpenAI) vs `_FINISH_REASONS` dict (chat) | **JUSTIFIED** (wire shape) | high | — | — |
| SYM-12 | choice-index guard: cerebras `is None` vs deepseek `< 0` | **JUSTIFIED** (nullable index) but subtly weaker | med | low | S |

---

## Symmetry matrix A — cross-provider engine behavior

| Behavior | openai/responses | openai/embedding | cerebras/chat | deepseek/chat |
|---|---|---|---|---|
| 5-branch error ladder (Auth/RateLimit/Response/Transport/API) | identical | identical **+ "embedding" in msgs** | identical | identical |
| `__init__` model-lookup + `BaseException` cleanup | identical | identical | identical | identical |
| `close` / `model` / `model_spec` props | identical | identical | identical | identical |
| `_retry_after` (`>=0 and isfinite`) | identical | identical | identical | identical |
| `_unsupported` return type | `-> None` | **absent (inlines raise)** | `-> None` | `-> Never` |
| `_error_metadata` param | non-opt **+ `_execution_metadata` wrapper** | `... \| None` | non-opt | non-opt |
| `execute → _build_request → _validate → _parse` | yes | **no `_build_request`** (inline) | yes | yes |
| finish-reason source | status-walk `_finish_reason` | n/a | `_FINISH_REASONS` dict | `_FINISH_REASONS` dict |
| choice-index guard | (item walk) | dup-index + contiguity | `index is None` | `index < 0` |
| usage optional at wire | `Usage \| None` | required (`EmbeddingList.usage`) | `Usage \| None` | **`Usage` required** |
| usage-inconsistency policy | raise | raise | raise | raise (strictest) |
| reads own `sampling_fields` for gate | temp/top_p only | n/a | **never** (matrix = all) | **never** |
| reads `message_roles`/`content_types`/`response_formats` | **never** | n/a | **never** | **never** |
| `model_dump(by_alias=…)` | `by_alias=True` | (embeddings, no alias) | `by_alias=True` | **omitted** |

The top block is *all identical* → dedup (SYM-5/6). The middle block is unmotivated
drift. The bottom block is where declared-capability and enforced-capability diverge.

## Symmetry matrix B — normalized sampling fields: SUPPORTED vs how ENFORCED

`✓` = accepted, `✗` = rejected. Parenthetical = the enforcement *mechanism*.

| Field | OpenAI-Responses | Cerebras | DeepSeek |
|---|---|---|---|
| `max_tokens` | ✓ (≤ `response_tokens`) | ✓ (≤ `response_tokens`) | ✓ (≤ `response_tokens`) |
| `temperature` | ✓/✗ (**matrix** `sampling_fields`; reasoning→off) | ✓ | ✓/✗ (**hardcode**: only if thinking off) |
| `top_p` | ✓/✗ (**matrix**) | ✓ | ✓/✗ (**hardcode**: only if thinking off) |
| `stop` | ✗ (**hardcode**) | ✓ ≤4 (**hardcode count**) | ✓ ≤16 (**hardcode count**) |
| `seed` | ✗ (**hardcode**) | ✓ | ✗ (**hardcode**) |
| `frequency_penalty` | ✗ (**hardcode**) | ✓ | ✗ (**hardcode**) |
| `presence_penalty` | ✗ (**hardcode**) | ✓ | ✗ (**hardcode**) |
| `logprobs` | ✗ (**hardcode**) | ✓ | ✓ |
| `top_logprobs` | ✓ | ✓ (**hardcode**: needs `logprobs`) | ✓ (**hardcode**: needs `logprobs`) |
| `logit_bias` | ✗ (**hardcode**) | ✓ | ✗ (**hardcode**) |

The **capability divergence is real and correct** (Cerebras supports everything, OpenAI
Responses supports few, DeepSeek is in between). The **enforcement mechanism is
incoherent**: `sampling_fields` is the declared source of truth for exactly two OpenAI
checks and is *never consulted* by Cerebras or DeepSeek, even though DeepSeek's
`_DEEPSEEK_SAMPLING_FIELDS` deliberately omits `SEED/FREQUENCY_PENALTY/PRESENCE_PENALTY/
LOGIT_BIAS` — the same four it then rejects by hand. That is two representations of one
fact (SYM-2).

## Symmetry matrix C — `_usage` consistency checks & policy

| Check | openai/resp | openai/embed | cerebras | deepseek |
|---|---|---|---|---|
| `usage is None → None` | ✓ | n/a (required) | ✓ | n/a (**required**) |
| negative-token guard | — | — | — (`or 0`) | **✓ `min(...) < 0`** |
| `total == in + out` | ✓ (always) | `total == prompt` | **only if all 3 present** | ✓ (always) |
| `cached ≤ prompt` | ✓ | n/a | ✓ (if present) | ✓ (if present) |
| `reasoning ≤ completion` | ✓ | n/a | ✓ (if present) | ✓ (if present) |
| `cache_hit + cache_miss == prompt` | n/a | n/a | n/a | **✓ EXACT equality** |
| `accepted + rejected ≤ completion` | n/a | n/a | ✓ | n/a |
| metadata for the raise | `_execution_metadata(response)` | `_error_metadata` | `_error_metadata` | `_error_metadata` |
| **failure policy** | **raise** (discard) | **raise** | **raise** | **raise** |

Four different strictness contracts for the same non-load-bearing accounting data, one
of them (DeepSeek) using additive exact-equality that breaks the moment a provider adds
a token category. This is SYM-1.

## Symmetry matrix D — ops input validation

| Module | Symbol guard | text guard | notes |
|---|---|---|---|
| `text.py` | `_symbol_value` (defined+used) | `_require_text` (defined+used) | reference pattern |
| `reason.py` | `_symbol_value` (**verbatim dup**) | `_require_text` (**verbatim dup**) | dup of text.py |
| `compare.py` | `_symbol_value` (defined+used) | **inlines** `"type_description must be text"` | no `_require_text` |
| `rank.py` | **inlines** `isinstance(source, Symbol)` | **inlines** `"measure must be text"` | no helpers; uses `source.value` directly |
| `embed.py` | bespoke `_text_inputs` / `_numeric_array` | n/a | richer contract (justified) |

`_symbol_value` verbatim ×3 (`text:467`, `reason:154`, `compare:177`); `_require_text`
verbatim ×2 (`text:475`, `reason:162`); two inline drifts. `ops/primitives.py` already
exists as the shared home (`_execute_language`) — these belong there. (SYM-7)

## Symmetry matrix E — Runtime vs RuntimeConfig validation

| Rule | `Runtime._validate_aliases/_default` | `RuntimeConfig._validate_aliases/_default` |
|---|---|---|
| alias `isinstance str` | **✓ (TypeError)** | ✗ (Pydantic enforces keys) |
| alias non-empty | ✓ | ✓ |
| alias **outer whitespace** | **✗ (accepted)** | **✓ (rejected)** |
| default `isinstance str` | **✓ (TypeError)** | ✗ |
| default **whitespace** | **✗ (accepted)** | **✓ (`!= strip()`)** |
| default membership | ✓ | ✓ |
| "≥1 engine" order | **last** (after default check) | **first** (model_validator top) |
| operation label in msgs | `"language-model"` (hyphen) | `"language model"` (space) |

Two paths that should yield one "valid runtime" invariant enforce different alias syntax
and even emit differently-worded errors. (SYM-3)

---

## Detailed findings

### SYM-1 — Usage-consistency: asymmetric strictness + fragile equality + discards valid output — BUG-IN-WAITING

**What.** Every engine treats provider token-accounting inconsistency as a hard
`InvalidResponseError` out of `execute`, throwing away a fully-generated completion over
billing metadata. And the four engines apply materially different strictness (matrix C).

**Where.** DeepSeek `_usage` (`deepseek/.../chat_completions.py`) is strictest and most
brittle:

```python
if (cache_hit is not None and cache_miss is not None
        and cache_hit + cache_miss != usage.prompt_tokens):
    msg = "DeepSeek cache token usage was inconsistent"
    raise InvalidResponseError(msg, metadata=error_metadata)
```

DeepSeek also uniquely guards negativity (`min(usage.prompt_tokens, ...) < 0`). Cerebras
by contrast validates the total **only when all three are present**:

```python
if (usage.prompt_tokens is not None and usage.completion_tokens is not None
        and usage.total_tokens is not None
        and usage.total_tokens != usage.prompt_tokens + usage.completion_tokens):
```

OpenAI checks the total unconditionally and uses `_execution_metadata(response)` (not
`_error_metadata`) for the raise.

**Why it matters.** `usage` is telemetry, not the answer. The exact-equality
`cache_hit + cache_miss == prompt_tokens` is precisely the check that fails when a
provider adds a token bucket (e.g. a third cache tier). Three providers disagreeing on
what "inconsistent" means is a maintenance hazard: a reviewer fixing one won't know the
others differ.

**Make it symmetric.** One shared usage-normalization policy for all engines:
on inconsistency **degrade to `usage=None`** (optionally `logger.warning`) and keep the
response; reserve `InvalidResponseError` for cases where the *content* can't be
normalized. If fail-loud is intentional, at minimum (a) relax exact-equality to bounded
(`cache_hit + cache_miss <= prompt_tokens`), and (b) unify the "when do we check total"
rule (always, given Pydantic already guarantees `ge=0`, dropping DeepSeek's negativity
special-case). The negativity guard is redundant with `TokenUsage`'s `Field(ge=0)`
downstream anyway.

**Feature impact:** `drops-minimal` (loses "returned usage is arithmetically perfect";
gains not discarding valid completions). **Conf:** high. **Impact:** med. **Effort:** S–M.

---

### SYM-2 — Capability declared symmetrically, enforced asymmetrically — BUG-IN-WAITING

**What.** Each engine builds a full `LanguageModelSpec` (matrix B), then enforces
capability with an imperative wall that mostly ignores the matrix and hardcodes the same
facts a second time. The enforcement *mechanism* differs per provider and per field.

**Where.** Only two reads of `model_spec.sampling_fields` exist in the whole tree, both
OpenAI (`SamplingField.TEMPERATURE not in self.model_spec.sampling_fields`, and the same
for `TOP_P`). `message_roles`, `content_types`, `response_formats`, and the normalized
`model_spec.context_tokens` have **zero** enforcement reads (grep confirmed — the only
`context_tokens` hits copy the *client* spec into the normalized one). Meanwhile
DeepSeek double-encodes:

```python
# _DEEPSEEK_SAMPLING_FIELDS omits SEED, FREQUENCY_PENALTY, PRESENCE_PENALTY, LOGIT_BIAS
# …yet _validate_request rejects each again by hand:
unsupported_sampling = (("seed", sampling.seed),
                        ("frequency_penalty", sampling.frequency_penalty),
                        ("presence_penalty", sampling.presence_penalty))
for field, value in unsupported_sampling:
    if value is not None:
        self._unsupported(f"DeepSeek does not support normalized {field}")
```

DeepSeek rejects image content **three** times: `content_types=(TEXT,)` in the matrix,
`_validate_request` `isinstance(content, ImageContent)` (line ~220), and again in
`_message` (line ~286). Today all encodings agree, so there is no live bug — but nothing
enforces the agreement: editing the matrix does nothing; editing a check silently
disagrees with the matrix that *reads* like the source of truth.

**Why it matters.** Two representations of one fact, one of them inert data that looks
authoritative — the classic drift trap the brief names. It also bloats each
`_validate_request` to 40–75 lines.

**Make it symmetric.** A shared, data-driven capability gate (in the adapter base
proposed by r1-07 PA-3) that reads the matrix as the single source of truth for
membership fields (roles, content types, response formats, and the set of *sampling
fields actually populated* by the request), leaving each engine only its genuinely
idiosyncratic rules (DeepSeek user-id regex, "temp/top_p ignored unless thinking off",
stop-count caps, `top_logprobs requires logprobs`, OpenAI `max_tokens ≤ response_tokens`).
Then the matrix becomes authoritative and the enforcement mechanism is identical across
providers. (Overlaps r1-06 C2's "expose vs demote" — either way, one representation.)

**Feature impact:** `keeps-all` (identical rejections, single source). **Conf:** high.
**Impact:** high. **Effort:** M.

---

### SYM-3 — `Runtime.__init__` vs `RuntimeConfig` validation diverge — BUG-IN-WAITING

**What.** Both validate alias syntax and default membership, with *different* rules
(matrix E). Consequence: `Runtime(language_models={" chat ": engine})` is **accepted**
(whitespace alias), while the equivalent `RuntimeConfig` is **rejected**; `Runtime`
type-checks that keys are `str`, `RuntimeConfig` relies on Pydantic and adds a
whitespace rule `Runtime` lacks. Even the "≥1 engine" check fires in a different order
(so an empty config with a stray `default_*` yields different errors), and the
operation label differs (`"language-model"` vs `"language model"`) so identical failures
produce differently-worded messages.

**Where.** `Runtime._validate_aliases` checks `isinstance(alias, str)` + `not alias`;
`RuntimeConfig._validate_aliases` checks `not alias` + `alias != alias.strip()`.
`RuntimeConfig.validate_aliases_and_defaults` runs the "≥1 engine" guard first;
`Runtime.__init__` runs it after `_validate_default`.

**Why it matters.** A caller who validates a `RuntimeConfig` then trusts the resulting
`Runtime` is fine; a caller who builds a `Runtime` directly (as the tests do throughout)
gets weaker validation. A silent contract seam between the two supported construction
paths.

**Make it symmetric.** Factor the shared alias-syntax + default-membership + ≥1-engine
rules into module-level free functions (in `runtime/config.py`) keyed on
`Mapping[str, object]`, called from both. Decide once whether outer whitespace is legal
(recommend rejecting everywhere) and unify the operation label. Keep the `str`-type
guard on the `Runtime` path only (it alone accepts untyped mappings); keep
`_validate_engine_identities` Runtime-only (it operates on live instances).

**Feature impact:** `keeps-all`. **Conf:** high. **Impact:** med. **Effort:** S.

---

### SYM-4 — Engine names unique per-capability, not globally — BUG-IN-WAITING (low)

**What.** `Runtime.__init__` validates `language_snapshot` and `embedding_snapshot`
independently; `_validate_engine_identities` dedups by `id()` across both maps but never
by name. So `language_models={"x": langA}, embeddings={"x": embB}` is accepted, and
`engine="x"` resolves to a different engine depending on request type. FIXPLAN §2 says
"names are globally unique within one Runtime."

**Where.** No intersection check anywhere in `runtime.py`; `_resolve_engine` is
capability-scoped, so it never notices the collision.

**Why it matters.** Functionally safe today (request type disambiguates) but diverges
from the written invariant and makes a name ambiguous, a latent hazard if a future
request type is capability-ambiguous.

**Make it symmetric.** Either add a cross-map name-collision check in the shared
validator (SYM-3) — recommended, keeps "a name identifies one engine" — or amend FIXPLAN
to say "unique within each capability." Code and spec must agree.

**Feature impact:** `drops-minimal` (same-name-across-capabilities). **Conf:** high.
**Impact:** low. **Effort:** S.

---

### SYM-5 — Error-mapping ladder verbatim ×4, embedding diverges only in wording — COSMETIC (dedup)

**What.** All four engines wrap the client call in the same 5-branch `except`
(Auth/RateLimit/Response/Transport/API → runtime taxonomy), differing only in the
`*_errors` alias and an English provider name. The embedding engine is the lone
asymmetry: it inserts "embedding" into three of the five messages ("OpenAI returned an
invalid **embedding** response", "OpenAI **embedding** transport failed", "OpenAI
**embedding** request failed with status …").

**Where.** `ResponsesEngine.execute`, `EmbeddingEngine.execute`, both
`ChatCompletionsEngine.execute`. Every provider error already subclasses the shared
`providers/_client/errors.py` base, so a single mapper catching the base classes is safe.

**Why it matters.** ~35 lines ×4 kept in lockstep by hand; any taxonomy change touches
four files. The embedding wording drift shows the copies already diverge without intent.

**Make it symmetric.** One shared `map_execution(call, *, provider, display, model)`
helper (r1-02 D1 / r1-07 PA-1), parametrized by the display string so wording is a
single argument — the embedding variant becomes `display="OpenAI embedding"`.

**Feature impact:** `keeps-all`. **Conf:** high. **Impact:** med. **Effort:** S.

---

### SYM-6 — Engine method-shape drift — COSMETIC

**What / Where.** Inside the otherwise-identical scaffolding, four unmotivated asymmetries:
- `_unsupported`: `-> Never` (deepseek), `-> None` (cerebras, openai/responses),
  **absent** in openai/embedding (which raises `UnsupportedFeatureError` inline). `Never`
  is strictly more precise and callers already treat it as no-return.
- `_error_metadata`: non-optional in responses/cerebras/deepseek, `... | None` in
  embedding; responses additionally wraps it in a redundant `_execution_metadata` that is
  literally `return self._error_metadata(response.metadata)`.
- `_build_request`: present in all three language engines, **absent** in embedding
  (validate + inline `CreateEmbeddingRequest(...)` in `execute`).
- `_retry_after`: identical ×4 (this one is fine — but it means one clamp is
  hand-maintained in four files, and both `ResponseMetadata.retry_after` and
  `ErrorMetadata.retry_after` are already `NonNegativeFiniteFloat`, so the clamp's only
  job is to coerce a bad transport value to `None` instead of raising).

**Make it symmetric.** Fold all of these into the shared adapter base (r1-07 PA-3):
standardize `_unsupported -> Never`, pick one `_error_metadata` shape, drop
`_execution_metadata`, and give the embedding adapter a `_build_request` for shape parity
(or explicitly document embeddings as the intentionally-simpler path).

**Feature impact:** `keeps-all`. **Conf:** high. **Impact:** low. **Effort:** S.

---

### SYM-7 — ops helper duplication + inline drift — COSMETIC (dedup)

**What / Where.** `_symbol_value` is defined verbatim ×3 (`text.py:467`, `reason.py:154`,
`compare.py:177`); `_require_text` verbatim ×2 (`text.py:475`, `reason.py:162`).
`compare.is_instance_of` inlines `"type_description must be text"` instead of calling
`_require_text`; `rank.rank` inlines *both* the Symbol guard (`"source must be a Symbol"`)
and the text guard (`"measure must be text"`) and never defines `_symbol_value` at all
(it reads `source.value` directly after the manual check). `embed.py` has a legitimately
richer bespoke contract (`_text_inputs`, `_numeric_array`) and can stay.

**Why it matters.** Three private copies plus drifted inline wording of one guard family,
when `ops/primitives.py` already exists as the shared ops-internal home. The inline
variants are exactly where wording will keep drifting.

**Make it symmetric.** Move `_symbol_value` and `_require_text` into `ops/primitives.py`;
import them in `text`/`reason`/`compare`/`rank`; replace the two inline blocks with the
helper. `embed._text_inputs` keeps its richer contract but can reuse `_symbol_value` for
its leading `isinstance(source, Symbol)` guard.

**Feature impact:** `keeps-all`. **Conf:** high. **Impact:** med. **Effort:** S.

---

### SYM-8 — logprobs is symmetric but incoherent (write-only) — BUG-IN-WAITING / decide

**What.** All three language engines forward logprobs request fields
(`top_logprobs`, and cerebras/deepseek also `logprobs`, `logit_bias`), the provider
*clients* even parse the returned logprobs, but `LanguageModelOutput` has **no** logprobs
field and **no engine reads them back** (grep: no read-back anywhere). So the asymmetry
here is unusual: the three engines are *consistent with each other* (all write-only), but
the request↔response contract is internally incoherent — you can ask for per-token
logprobs and can never receive them.

**Why it matters.** The public type advertises a capability the response half cannot
express, and `logprobs=true` can change provider billing, so it is not free to send.

**Make it symmetric (both halves).** Either close the loop (add
`logprobs: tuple[TokenLogprob, ...]` to `LanguageModelOutput` and map it in all three
`_output` builders) or cut the request side entirely (`logprobs`/`top_logprobs`/
`logit_bias` + `LogitBias` + `validate_unique_logit_bias_tokens` + the three
`SamplingField.*LOGPROB*`/`LOGIT_BIAS` members). Do not keep the half-state. Given no
consumer today, cutting is the lower-risk default unless logprobs is a product goal
(r1-06 C1 — the one place the intended-feature question genuinely gates direction).

**Feature impact:** `drops-minimal` (no consumer today). **Conf:** high. **Impact:** med.
**Effort:** S (cut) / M (close).

---

### SYM-9 — `by_alias` serialization divergence — JUSTIFIED but fragile

**What.** `openai/_client.py` and `cerebras/_client.py` dump requests with
`model_dump(mode="json", by_alias=True, exclude_none=True)`; `deepseek/_client.py` uses
`model_dump(mode="json", exclude_none=True)` — **no `by_alias`**.

**Where / verdict.** DeepSeek's request DTOs have **no aliased fields** (grep: `alias=`
appears only in cerebras `chat.py` and openai `responses.py`, both for the reserved
`schema` field), so `by_alias=True` is a no-op for DeepSeek *today* — the divergence is
**justified**. But it is fragile-by-omission: if any DeepSeek request field ever gains an
alias, serialization silently breaks with no test to catch it. Note also that
cerebras/openai belt-and-suspenders this (`serialize_by_alias=True` in the aliased
sub-model's `model_config` **and** `by_alias=True` at dump), so one of the two is
redundant there.

**Make it symmetric.** Pass `by_alias=True` in DeepSeek's dump too (harmless today,
future-proof), or route all three through the shared `BaseClient.request(...)` (r1-02 D2)
that takes `by_alias` as an explicit per-endpoint argument with one default.

**Feature impact:** `keeps-all`. **Conf:** high. **Impact:** low. **Effort:** S.

---

### SYM-10 — Naming symmetry — COSMETIC

**What / Where.**
- **`operations.py` vs `ops/`.** `symai/operations.py` (request builders:
  `language_request`, `image_request`, `embedding_request`, `data_uri`,
  `parse_embedding_response`) sits beside the `symai/ops/` package (semantic ops). Two
  near-homonym modules at the same level with unrelated jobs; grep/goto ambiguity and a
  real "which one?" for a new reader.
- **Two `load_runtime`.** `symai.loading.load_runtime` (public: composes builtins) and
  `symai.runtime.loading.load_runtime` (generic: explicit loader lists), the latter
  imported as `_load_runtime`. Rename the generic to `compose_runtime` /
  `load_runtime_from_loaders` so the names read "public entry" vs "low-level composer."
- **`cast("ImplementationId", …)` ×4** in `loading.py` builtin loader tuples is a no-op
  (`ImplementationId = Annotated[str, BeforeValidator(...)]` is statically `str`); it
  adds noise and implies a static brand that does not exist. Drop it (runtime validation
  already happens in `_index_entries`), or model builtins as a `StrEnum`.
- **`FrozenModel` ≡ `StrictModel`.** `runtime/models.py::FrozenModel` and
  `providers/_client/models.py::StrictModel` are byte-identical config
  (`frozen=True, strict=True, extra="forbid"`). Two names for one base; defensible only
  because they sit in different layers — worth a comment, not a merge.

**Make it symmetric.** Rename the generic `load_runtime`; drop the no-op casts; consider
renaming `operations.py` → `requests.py` (or folding its builders under a clearer name)
so it is not a homonym of `ops/`. Decoder names (`TextDecoder`/`ConstructorDecoder`/
`TypeAdapterDecoder`/`PydanticDecoder`) are already consistent — keep them.

**Feature impact:** `keeps-all`. **Conf:** high. **Impact:** low. **Effort:** S.

---

### SYM-11 / SYM-12 — Justified divergences (document, don't "fix")

- **SYM-11 finish-reason.** OpenAI Responses resolves finish via a status walk
  (`ResponseStatus` + `incomplete_details.reason`); the two chat engines use a
  `_FINISH_REASONS` `MappingProxyType`. This is a genuine wire-shape difference (Responses
  has no per-choice `finish_reason` string) and should stay. The two chat maps *are*
  near-duplicate (`{stop, length, content_filter, +error|insufficient_system_resource}`)
  and are the natural parameter of a shared `ChatCompletionsAdapter` (r1-07 PA-4) — that
  is dedup, not a symmetry defect.
- **SYM-12 choice-index guard.** Cerebras checks `choice.index is None` (its wire type is
  `int | None`); DeepSeek checks `choice.index < 0` (its wire type is `int`). Both are
  justified by the wire types — but note the guards are not equivalent: cerebras does
  **not** reject a negative index, relying on `LanguageModelOutput.index = Field(ge=0)`
  downstream to raise `ValidationError` (caught → `InvalidResponseError`). It works, but
  the defense is implicit on one side and explicit on the other. A shared chat base with
  one `is_valid_index` hook would make the intent uniform.

---

## What is already symmetric and should be kept

- **Op signature contract.** Every I/O op is `(runtime, source, …extra…, *, engine=None)
  -> Symbol[…]`; every deterministic op (`text.template`, `embed.similarity/distance/
  mmd/kernel`) drops both `runtime` and `engine`. The `engine=<name>` keyword is uniform
  across ops, `Function.__call__`, `Function.execute_many`, and `Runtime.execute`. This
  is the strongest symmetry in the codebase — do not touch it.
- **Decoder family.** `TextDecoder`/`ConstructorDecoder`/`TypeAdapterDecoder`/
  `PydanticDecoder` share the `*Decoder` suffix, all `@dataclass(frozen=True, slots=True)`,
  all implement `decode(self, text, /) -> T`, and all (bar the infallible `TextDecoder`)
  wrap failures uniformly via `_decode_error`. `_normalize_text` is applied identically in
  every decoder.
- **Typed, layered error taxonomy.** Every provider error subclasses the shared
  `providers/_client/errors.py` base; the runtime keeps a neutral hierarchy
  (`SymbolicAIRuntimeError` → `ExecutionError` → Auth/RateLimit/Transport/InvalidResponse).
  The engine is the single crossing point — this is exactly what makes SYM-5's shared
  mapper safe.
- **Construction-cleanup discipline.** Both clients and engines wrap partial construction
  in `try/except BaseException` with `add_note` on cleanup failure, and `close()` is
  idempotent everywhere. Identical by design — hoist it (SYM-6), don't rewrite it.
- **Strict/tolerant model boundary.** Request DTOs → `StrictModel` (`extra="forbid"`),
  response DTOs → `TolerantModel` (`extra="allow"`), internally-assembled normalized
  models → `FrozenModel` (strict). Correct per-direction choice, uniform across providers.
- **Multi-output index integrity.** Both chat engines reject duplicate/invalid indices
  and `outputs.sort(key=index)`; the embedding engine enforces contiguous indices. Uniform
  defensive normalization of untrusted payloads.

---

## Net

The symmetry work splits cleanly. **Fix the two real divergences** (SYM-1 usage policy,
SYM-2 declared-vs-enforced capability) and the **one contract seam** (SYM-3 Runtime vs
RuntimeConfig) — these are where drift becomes a bug. **Collapse the cosmetic residue**
(SYM-5/6/7/10) into the shared adapter base + shared ops-helper module that r1-02 and
r1-07 already propose — it removes the *asymmetries that remain inside otherwise-identical
code*. **Document the justified divergences** (SYM-9 by_alias, SYM-11 finish-reason,
SYM-12 index guard) as deliberate wire-shape differences, and **decide SYM-8** (logprobs)
in one direction. Keep the op signature contract, decoder family, error taxonomy, cleanup
discipline, and strict/tolerant boundary exactly as they are — those are the parts that
are symmetric on purpose.
