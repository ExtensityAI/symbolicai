# r2-c2 — Consistency & Symmetry (cross-cutting)

**Lens:** things that are asymmetric but should be symmetric — across the three language
engines, the ops modules, the two validation paths (`Runtime` vs `RuntimeConfig`), and the
naming surface. For each divergence: is it a **bug-in-waiting**, **justified** (real API/wire
difference), or **cosmetic**? Every finding is re-verified against the live tree at read time;
anchors are symbol + snippet, line numbers approximate. Snapshot is a moving target.

I confirmed the Round-2 STATE UPDATE against the code: root `symai/__init__.py` is **0 bytes**;
`jinja2`/`box`/`tomllib`/`prompts.py`/`backend/` are gone; `_CURRENT_RUNTIME`/`current_runtime`/
`NoActiveRuntimeError`/`__all__`-at-root have **zero** production references. So the r1 facade
(API-01/03), ambient-runtime (R5-1), and dead-export (API-05) findings are **RESOLVED** — I do
not re-report them. What remains is genuine asymmetry in the surviving code.

---

## Executive summary

1. **The cross-provider language engines are strongly symmetric where it counts and the
   divergences are almost all *justified* real API differences** — but the divergences are
   *enforced by hand in inconsistent ways*, so the asymmetry is in the **mechanism**, not the
   policy. The `sampling_fields` capability matrix is consulted only by OpenAI, and only for 2
   of its own fields; Cerebras and DeepSeek declare it and never read it back, hardcoding the
   same facts a second time (sharpens r1-07 PA-2 as a *symmetry* defect).
2. **Two concrete bugs-in-waiting.** (a) DeepSeek serializes requests **without** `by_alias=True`
   while OpenAI and Cerebras use it — harmless *today* only because DeepSeek has zero aliased
   request fields; add one and it silently ships the Python name. (b) `Runtime.__init__` and
   `RuntimeConfig` enforce **different** alias/default rules, so `Runtime({" chat ": e})` is
   accepted while the same alias via `RuntimeConfig` is rejected (confirms r1-05 R5-3).
3. **Usage-consistency strictness is asymmetric across providers and DeepSeek's is fragile.**
   All four engines raise `InvalidResponseError` (discarding a valid completion) on token-math
   inconsistency — uniform *policy* — but the *degree* runs openai(loosest) → cerebras(bounds)
   → deepseek(exact `cache_hit+cache_miss == prompt_tokens`, strictest). The exact-sum equality
   is a bug-in-waiting (confirms r1-07 PA-5).
4. **The ops layer is symmetric in the two things that matter to a user** (signature shape
   `(runtime, …, *, engine=None)` and Symbol-wrapping of every result) **and asymmetric in the
   two things that matter to a maintainer** (`_symbol_value`/`_require_text` are defined 3×/2×,
   inlined elsewhere with drifted wording, and absent from the one shared home `ops/primitives.py`
   that already holds `_execute_language`).
5. **Naming has one real collision** — `symai/operations.py` (request builders) vs `symai/ops/`
   (semantic operations), plus `load_runtime` ×2 and `loading.py` ×5 — all cosmetic but they blur
   which module owns what. Everything else on the naming surface (decoders `*Decoder`, the error
   taxonomy) is cleanly symmetric and should be kept.

Overall read: this is a **well-disciplined codebase whose asymmetries are mostly the residue of
hand-maintaining N copies of a contract**. The two bugs-in-waiting (by_alias, Runtime/RuntimeConfig
validation) are cheap and worth fixing now; the rest is mechanism-consistency that a shared adapter
base (r1-07 PA-1/PA-3) and a shared validator (r1-05 R5-3) erase for free.

---

## Findings table (ranked)

| ID | Divergence | Class | Conf | Impact | Effort |
|----|-----------|-------|------|--------|--------|
| S1 | `Runtime.__init__` vs `RuntimeConfig` alias/default rules diverge (str-type vs whitespace) | **BUG-IN-WAITING** | high | med | S |
| S2 | DeepSeek request serialization omits `by_alias=True`; openai/cerebras use it | **BUG-IN-WAITING** | high | low–med | S |
| S3 | Engine names unique per-capability only, not globally (spec says globally) | **BUG-IN-WAITING** | med | low | S |
| S4 | Usage-consistency strictness asymmetric; DeepSeek exact cache-sum equality fragile | **BUG-IN-WAITING** | med | med | S–M |
| S5 | `sampling_fields` matrix consulted only by OpenAI (2 fields); cerebras/deepseek hardcode | JUSTIFIED policy, inconsistent mechanism | high | med | M |
| S6 | ops input guards `_symbol_value`/`_require_text` defined N× + inlined + wording drift; not in `primitives` | COSMETIC (DRY) | high | low | S |
| S7 | `_unsupported` return type None/None/Never; `_error_metadata` opt vs non-opt + `_execution_metadata` wrapper | COSMETIC | high | low | S |
| S8 | Naming: `operations.py` vs `ops/`; `load_runtime` ×2; `loading.py` ×5; no-op `cast("ImplementationId",…)` | COSMETIC | high | low | S |
| S9 | DeepSeek image-reject encoded 3× (matrix + `_validate_request` + `_message`) | COSMETIC (drift risk) | high | low | S |
| S10 | Embedding engine breaks `execute→_build_request→validate→parse` shape; `_parse_response` except-set differs | COSMETIC | med | low | S |
| S11 | `from __future__ import annotations` in 5 ops modules, absent in `primitives`/engines | COSMETIC | high | low | S |

---

## Matrix A — Cross-provider language-engine symmetry

Files: `openai/engines/responses.py`, `cerebras/engines/chat_completions.py`,
`deepseek/engines/chat_completions.py` (+ `openai/engines/embedding.py` for scaffolding rows).

### A.1 Scaffolding (should be identical — and is)

| Behavior | openai/responses | cerebras/chat | deepseek/chat | openai/embedding | Verdict |
|---|---|---|---|---|---|
| `__init__` model-lookup + `try/except BaseException` cleanup + `add_note` | identical | identical | identical | identical | **symmetric ✓** (dedup: PA-3) |
| `close()` idempotent | identical | identical | identical | identical | **symmetric ✓** |
| `model`/`model_spec` properties | identical | identical | identical | identical | **symmetric ✓** |
| `_retry_after` clamp `>=0 and isfinite` | identical | identical | identical | identical | **symmetric ✓** |
| 5-arm error ladder Auth/RateLimit/Response/Transport/API | identical | identical | identical | identical | **symmetric ✓** |
| error classes (Auth/RateLimit/Response/Transport/API subclass shared `_client.errors`) | ✓ | ✓ | ✓ | ✓ | **symmetric ✓** |
| `_unsupported` return type | `-> None` | `-> None` | `-> Never` | (raises inline) | **asymmetric** → S7 |
| `_error_metadata` signature | non-opt **+ `_execution_metadata` wrapper** | non-opt | non-opt | **`| None`** | **asymmetric** → S7 |
| `_build_request` method exists | yes | yes | yes | **no** (inlined in `execute`) | **asymmetric** → S10 |
| `_validate_request` call site | inside `_build_request` | inside `_build_request` | inside `_build_request` | top of `execute` | **asymmetric** → S10 |
| `_parse_response` except-set | `ValidationError` | `(TypeError, ValidationError)` | `(TypeError, ValidationError)` | `(TypeError, ValidationError)` | **asymmetric** → S10 |

The whole top block is verbatim-identical — good, and exactly why a shared base is safe (owned by
r1-07 PA-3). My additions here are the four asymmetric rows at the bottom.

### A.2 Normalized sampling fields — what each engine rejects, and *how*

The reviewer's core question: do the three validate/reject the **same** normalized fields? **No —
and correctly not**, because Cerebras (Chat-Completions) genuinely supports seed/penalties/logprobs/
logit_bias while OpenAI-Responses and DeepSeek do not. The values below are **justified**. The
smell is the **mechanism** column.

| Sampling field | openai/responses | cerebras | deepseek | mechanism |
|---|---|---|---|---|
| `stop` | reject any (`if sampling.stop`) | allow, `len ≤ 4` | allow, `len ≤ 16` | all **hardcoded**; count limit never in matrix |
| `seed` | reject | **allow** | reject | hardcoded (openai/deepseek); cerebras: matrix has all |
| `frequency_penalty` | reject | **allow** | reject | hardcoded |
| `presence_penalty` | reject | **allow** | reject | hardcoded |
| `logprobs` | reject | **allow** | **allow** | hardcoded |
| `top_logprobs` | allow (in `sampling_fields`) | allow + requires `logprobs=true` | allow + requires `logprobs=true` | mixed |
| `logit_bias` | reject | **allow** | reject | hardcoded |
| `temperature` | allow iff `TEMPERATURE in sampling_fields` (**matrix-driven**) | allow | allow iff thinking disabled | **inconsistent** |
| `top_p` | allow iff `TOP_P in sampling_fields` (**matrix-driven**) | allow | allow iff thinking disabled | **inconsistent** |

The mechanism asymmetry (→ **S5**):
- OpenAI **mixes** styles in one method: `SamplingField.TEMPERATURE not in self.model_spec.sampling_fields`
  (matrix) sits directly above six hardcoded `if sampling.stop: self._unsupported(...)` lines.
- Cerebras declares `sampling_fields = tuple(SamplingField)` (ALL) then **never reads it** — its only
  sampling checks are the bespoke `len(stop) > 4` and `top_logprobs requires logprobs`.
- DeepSeek declares a 6-field `_DEEPSEEK_SAMPLING_FIELDS` then **never reads it** — it hardcodes the
  rejection of exactly the omitted 3 (`seed`/`frequency_penalty`/`presence_penalty` via a loop, plus
  `logit_bias`). The matrix and the checks encode the same fact twice.

### A.3 Finish reason, index, usage, metadata

| Behavior | openai/responses | cerebras | deepseek | Verdict |
|---|---|---|---|---|
| finish-reason source | status-based (`ResponseStatus` + `incomplete_details.reason`) | `_FINISH_REASONS` dict | `_FINISH_REASONS` dict | **JUSTIFIED** (Responses wire differs) |
| finish-reason map contents | n/a | `{stop,length,content_filter,error}` | `{stop,length,content_filter,insufficient_system_resource→ERROR}` | **JUSTIFIED** (provider strings) |
| finish-reason None guard | n/a | `is None or not in` (field `str \| None`) | `not in` (field `str`) | **JUSTIFIED** (wire optionality) |
| choice index guard | positional `enumerate` | `index is None` (`int \| None`) | `index < 0` (`int`) | **JUSTIFIED** by wire; but note **gap**: cerebras never rejects a *negative* index → S10 nit |
| `metadata` (request labels) | forwarded | reject (`if request.metadata`) | reject | **JUSTIFIED** (only Responses supports it) |
| image input | gated on `model_spec.vision` | image ok; reject `image detail` | reject image (×3) | **JUSTIFIED** values; deepseek redundancy → S9 |
| usage present required? | `usage \| None`→None ok | `Usage \| None`→None ok | **`Usage` required**, returns non-None | **JUSTIFIED** by wire, but different guarantee → S4 |
| usage total check | `total != in+out` (uncond.) | `total != prompt+comp` (only if all present) | `total != prompt+comp` (uncond.) | asymmetric → S4 |
| usage negativity check | none | none | explicit `min(...) < 0` | asymmetric (redundant w/ `TokenUsage ge=0`) → S4 |
| usage cache check | `cached > input` (bound) | `cached > prompt` (bound) | **`cache_hit+cache_miss == prompt` (exact)** + bounds | **fragile/asymmetric** → S4 |
| inconsistency policy | raise `InvalidResponseError` | raise | raise | symmetric *policy*, asymmetric *degree* |
| `by_alias` on request serialize | `True` | `True` | **omitted** | **BUG-IN-WAITING** → S2 |

---

## Matrix B — Cross-ops symmetry

Files: `ops/{text,reason,compare,rank,embed}.py`, `ops/primitives.py`, `operations.py`.

| Property | text | reason | compare | rank | embed | Verdict |
|---|---|---|---|---|---|---|
| I/O op signature `(runtime, src…, *, engine=None)` | ✓ | ✓ | ✓ | ✓ | ✓ (`embed`) | **symmetric ✓** |
| deterministic op drops `runtime`+`engine` | ✓ (`template`) | — | — | — | ✓ (`similarity/distance/mmd/kernel`) | **symmetric ✓** |
| result wrapped in `Symbol(...)` | ✓ | ✓ | ✓ | ✓ | ✓ | **symmetric ✓** |
| execution via `primitives._execute_language` | ✓ | ✓ | ✓ | ✓ | n/a (embed uses `runtime.execute`) | **symmetric ✓** (embed justified — it's an embedding request) |
| `_symbol_value` guard | **defines locally** | **defines locally (dup)** | **defines locally (dup)** | **inlines** `isinstance(source, Symbol)` | bespoke `_text_inputs` | **asymmetric** → S6 |
| `_require_text` guard | **defines locally** | **defines locally (dup)** | **inlines** in `is_instance_of` | **inlines** `"measure must be text"` | n/a | **asymmetric + wording drift** → S6 |
| `from __future__ import annotations` | ✓ | ✓ | ✓ | ✓ | ✓ | present here, **absent** in `primitives`/`__init__` → S11 |

The user-facing surface (top four rows) is **model-consistent**; the maintainer-facing surface
(guards) is not. `ops/primitives.py` already exists as the shared-helper home (`_execute_language`
is imported by all four language ops) but the two guards that belong there are copy-pasted instead.
Wording has already drifted: `"must be a Symbol"` (helper) vs `"source must be a Symbol containing
non-empty text input(s)"` (embed) vs the inlined rank/compare variants.

---

## Matrix C — Validation symmetry (`Runtime.__init__` vs `RuntimeConfig`)

Both paths must yield the same "valid runtime" invariant (direct `Runtime(...)` is used throughout
tests; `RuntimeConfig`→`load_runtime` is the config path). They enforce **different** rules.

| Rule | `Runtime._validate_aliases`/`_validate_default` (runtime.py) | `RuntimeConfig._validate_aliases`/`_validate_default` (config.py) | Divergence |
|---|---|---|---|
| alias is `str` | `if not isinstance(alias, str): raise TypeError` | (Pydantic enforces via `Mapping[str, …]`) | Runtime-only guard (needed — accepts raw `Mapping`) |
| alias non-empty | `if not alias: raise ValueError` | `if not alias: raise ValueError` | same ✓ |
| alias outer whitespace | **not checked** | `if alias != alias.strip(): raise ValueError` | **DIVERGENT**: `Runtime({" chat ": e})` accepted; `RuntimeConfig` rejects |
| default is `str` | `if not isinstance(default, str): raise TypeError` | (Pydantic) | Runtime-only |
| default non-empty/stripped | **not checked** (only membership) | `if not default or default != default.strip(): raise ValueError` | **DIVERGENT** |
| default membership | `if default in engines: return` | `if default in engines: return` | same ✓ |
| engine-identity dedup (`id()`) | `_validate_engine_identities` (Runtime-only) | — (equal specs are legal) | **correctly Runtime-only** ✓ |
| name uniqueness scope | per-map only (language and embedding validated **independently**) | per-map only | **both** per-capability, not global → S3 |

Consequence (→ **S1**): the two "valid alias" contracts disagree in both directions — a caller who
builds a `Runtime` directly gets *weaker* validation than one who goes through `RuntimeConfig`. This
is a silent contract seam, not a style nit.

---

## Detailed findings

### S1 — `Runtime` and `RuntimeConfig` enforce divergent alias/default rules (BUG-IN-WAITING)

**Where.** `Runtime._validate_aliases` (runtime.py) checks `isinstance(alias, str)` + non-empty and
stops there; `RuntimeConfig._validate_aliases` (config.py) checks non-empty + `alias != alias.strip()`.
Symmetric divergence on defaults: `Runtime._validate_default` checks `isinstance` + membership;
`RuntimeConfig._validate_default` checks `not default or default != default.strip()` + membership.

**Why it's a bug-in-waiting.** `Runtime(language_models={" chat ": engine})` constructs a Runtime
whose only engine is reachable only via the whitespace name `" chat "`; the identical config through
`RuntimeConfig` raises. Same invariant, two answers. A user who validates a `RuntimeConfig` and then
trusts the resulting `Runtime` is fine; one who constructs `Runtime` directly (as all the tests do)
gets a laxer contract.

**Make it symmetric.** Factor the shared rules (non-empty, no-outer-whitespace, default membership,
at-least-one-engine) into module-level free functions keyed on `Mapping[str, object]` and call them
from **both** sites. Decide once whether outer whitespace is legal (recommend reject everywhere).
Keep the `str`-type guard on the `Runtime` path only (it alone accepts an untyped `Mapping`); keep
`_validate_engine_identities` Runtime-only. **Feature impact:** keeps-all. Matches r1-05 R5-3.

### S2 — DeepSeek serializes requests without `by_alias=True` (BUG-IN-WAITING)

**Where.**
```
deepseek/client/_client.py:  request.model_dump(mode="json", exclude_none=True)
cerebras/client/_client.py:  request.model_dump(mode="json", by_alias=True, exclude_none=True)
openai/client/_client.py:    body.model_dump(mode="json", by_alias=True, exclude_none=True)
```
I verified DeepSeek's request models (`deepseek/client/chat.py`) contain **zero** `Field(alias=…)` /
`serialization_alias` / `serialize_by_alias`, whereas Cerebras (`JsonSchemaSpec.body = Field(alias="schema")`,
`serialize_by_alias=True`) and OpenAI (`schema_ = Field(alias="schema")`) both alias fields. So today
the DeepSeek omission is a **no-op** — the wire is correct only by the accident that nothing is aliased.

**Why it's a bug-in-waiting.** The moment any DeepSeek request field needs a wire name that differs
from its Python identifier (a keyword-clashing key, a camelCase wire field, a `schema`-style alias),
the missing `by_alias=True` will silently emit the Python name and the request will be malformed —
with no type error to catch it. `by_alias=True` is a **safe uniform default**: it is a no-op when
there are no aliases and correct when there are.

**Make it symmetric.** Add `by_alias=True` to the DeepSeek `model_dump`. (When the client transport
is unified per r1-02 D2, pass the flag as a constant `True` for all three rather than a per-provider
argument — there is no provider for which `by_alias=False` is correct.) **Feature impact:** keeps-all.

### S3 — Engine names are unique per-capability, not globally (BUG-IN-WAITING)

**Where.** `Runtime.__init__` validates `language_snapshot` and `embedding_snapshot` independently;
there is no cross-map name check. `Runtime(language_models={"x": langA}, embeddings={"x": embB})` is
accepted, and `execute(...)` disambiguates by request type. FIXPLAN §2 says "names are globally
unique within one Runtime."

**Why it matters.** Safe today (request type disambiguates), but `engine="x"` then denotes two
different engines depending on request type, and it contradicts the written invariant. Latent hazard
if a future request type is capability-ambiguous.

**Make it symmetric.** Either add a cross-map uniqueness check in the shared validator from S1
(recommended — "a name identifies one engine"), or amend FIXPLAN to "unique within each capability."
Pick one so code and spec agree. **Feature impact:** drops-minimal (same-name-across-capabilities).
Matches r1-05 R5-4.

### S4 — Usage-consistency strictness is asymmetric; DeepSeek's exact cache-sum is fragile (BUG-IN-WAITING)

**Where.** All four engines raise `InvalidResponseError` on token-math inconsistency (uniform policy),
but the checks differ materially:
- OpenAI `_usage`: `total != input+output or cached>input or reasoning>output`.
- Cerebras `_usage`: total check only when all three present; cached/reasoning/prediction **bounds**.
- DeepSeek `_usage`: explicit negativity (`min(...) < 0`), unconditional total, **exact**
  `cache_hit + cache_miss != usage.prompt_tokens`, plus cache/reasoning bounds. This is the strictest.

**Why it's a bug-in-waiting.** `usage` is billing/telemetry, not the answer, yet an inconsistency
discards an already-generated completion. The DeepSeek **exact-sum equality** is precisely the check
that breaks when a provider adds a token category (a new cache bucket) that doesn't sum to the old
`prompt_tokens` — converting a successful call into a user-visible error. Cerebras models the same
relationship as a bound; the two disagree for no capability reason. The negativity check is redundant
with `TokenUsage`'s `ge=0` (verified) and unique to DeepSeek.

**Make it symmetric.** Two options (r1-07 PA-5, endorsed): (1) **degrade** — on inconsistency return
`usage=None` and keep the completion, reserving `InvalidResponseError` for un-normalizable *content*;
or at minimum (2) relax DeepSeek's `==` to `<=` and unify the three engines so "inconsistent" means
the same thing everywhere. **Feature impact:** drops-minimal (loses the guarantee that returned usage
is arithmetically perfect; gains not discarding valid completions).

### S5 — `sampling_fields` matrix is authoritative for no one; enforcement mechanism is inconsistent (JUSTIFIED policy, inconsistent mechanism)

**Where.** See Matrix A.2. OpenAI reads `sampling_fields` for `temperature`/`top_p` and hardcodes six
others in the same method; Cerebras declares `tuple(SamplingField)` and reads none; DeepSeek declares
a 6-tuple and reads none, hardcoding the omitted three.

**Why it matters (symmetry angle).** The *divergence in which fields are supported is justified* — the
three APIs genuinely differ. The defect is that the divergence is **hand-coded three different ways**
when it is already fully described by data (`sampling_fields`). Editing the matrix does nothing;
editing a check silently disagrees with the matrix. The matrix reads as the source of truth and is
not — and OpenAI even mixes both styles in one function, so a reader can't tell which is canonical.

**Make it symmetric.** A single data-driven capability gate in the shared adapter base that consults
`sampling_fields` (and `message_roles`/`content_types`/`response_formats`) as the one source of truth,
with a static `SamplingField → (attr, is-set)` table; each engine keeps only its genuinely
non-boolean rules as a `_validate_provider_specifics` hook (stop-count limits, "temp/top_p ignored
unless thinking disabled", "effort forbidden when thinking disabled", user-id regex, `top_logprobs
requires logprobs`). **Feature impact:** keeps-all (identical rejections, single source). This is
r1-07 PA-2 and r1-06 C2 — I confirm both and frame the fix as *make the three engines enforce
capability the same way*, which also fixes S9.

### S6 — ops input guards defined N×, inlined elsewhere, absent from `primitives` (COSMETIC/DRY)

**Where.** `_symbol_value` verbatim in `text.py`, `reason.py`, `compare.py`; inlined in `rank.py`
(`if not isinstance(source, Symbol)`) and `embed._text_inputs`. `_require_text` in `text.py`,
`reason.py`; inlined in `compare.is_instance_of` and `rank.rank`. None live in `ops/primitives.py`,
which already houses the shared `_execute_language`.

**Why it matters.** Same category of shared helper as `_execute_language`, factored inconsistently;
wording has already drifted across the inlined copies. Pure maintainer-facing asymmetry.

**Make it symmetric.** Move `_symbol_value` and `_require_text` into `ops/primitives.py`; import
everywhere; replace the rank/compare inlines. `embed._text_inputs` stays (richer contract) but its
leading `isinstance(source, Symbol)` can reuse `_symbol_value`. **Feature impact:** keeps-all.
Matches r1-02 D7.

### S7 — engine helper-signature nits (COSMETIC)

**Where.** `_unsupported` returns `None` (openai, cerebras) vs `Never` (deepseek). `_error_metadata`
is non-optional in responses/cerebras/deepseek but `| None` in embedding, and responses alone wraps it
in a redundant `_execution_metadata` (`return self._error_metadata(response.metadata)`).

**Why it matters.** `Never` is strictly more precise (lets pyright mark post-`_unsupported` code
unreachable); the two `None` copies lose that. The `_error_metadata`/`_execution_metadata` split is
gratuitous — one shape would do.

**Make it symmetric.** Standardize `_unsupported` on `-> Never`; pick one `_error_metadata` shape
(the shared base from PA-3 is the natural home); drop `_execution_metadata`. **Feature impact:** keeps-all.

### S8 — naming collisions (COSMETIC)

**Where.** `symai/operations.py` (request *builders*: `language_request`, `image_request`,
`embedding_request`, `data_uri`, `parse_embedding_response`) vs `symai/ops/` (semantic *operations*:
`text.summarize`, …) — two unrelated concepts one letter apart, both imported across the tree
(`function.py` and `ops/embed.py` import from `operations`). Plus `load_runtime` ×2
(`symai.loading.load_runtime` public composer; `symai.runtime.loading.load_runtime` generic, imported
`as _load_runtime`) and `loading.py` ×5 (`symai/`, `runtime/`, and each provider). And the no-op
`cast("ImplementationId", …)` ×4 in `loading.py` with its inverse `cast("str", spec.implementation)`
in `runtime/loading.py` — the branded alias is statically just `str`, so both casts are inert.

**Why it matters.** Grep/goto ambiguity and "which module owns request-building vs operations?"
friction for a new reader. No behavior at stake.

**Make it symmetric.** Rename `operations.py` → `requests.py` (or fold the builders into
`runtime/models.py`'s neighborhood) so "ops" means exactly one thing; rename the generic
`runtime.loading.load_runtime` → `compose_runtime`; drop the branded `ImplementationId` in favor of a
`StrEnum` of builtin implementations (r1-08 API-04) which removes both casts. **Feature impact:**
keeps-all. Confirms r1-08 API-04/API-06.

### S9 — DeepSeek image rejection encoded three times (COSMETIC, drift risk)

**Where.** DeepSeek declares `content_types=(ContentType.TEXT,)` (matrix), rejects `ImageContent` in
`_validate_request` (`if any(isinstance(content, ImageContent) …)`), and rejects it a **third** time
inside `_message` (`if isinstance(part, ImageContent): _unsupported(...)`). Three representations of
one fact.

**Why it matters.** The three agree today; nothing enforces the agreement. Folded into the S5
data-driven gate, the matrix becomes the single source and the two imperative copies disappear.
**Make it symmetric.** Drive the reject from `content_types` (S5); delete the `_message` and
`_validate_request` image checks. **Feature impact:** keeps-all.

### S10 — embedding engine breaks the language-engine shape (COSMETIC)

**Where.** The three language engines follow `execute → _build_request (which calls _validate) → parse`.
`EmbeddingEngine.execute` calls `_validate_request` directly then inlines `CreateEmbeddingRequest(...)`
— **no `_build_request`**. Also its `_parse_response` catches `(TypeError, ValidationError)` while
OpenAI Responses' `_parse_response` catches only `ValidationError`. Minor index nit: Cerebras rejects
`index is None` but never a negative index, while DeepSeek rejects `< 0` — the guards are each tailored
to their wire type but leave asymmetric holes.

**Why it matters.** Embedding is genuinely the simpler path, so this is defensible — but it's the odd
one out in the "execute→build→validate→parse" pattern the reviewer asked about, and the except-set
asymmetry means an unexpected `TypeError` in OpenAI Responses' parse would escape as a raw error
instead of `InvalidResponseError`.

**Make it symmetric.** Give the embedding adapter a `_build_request` for shape parity (or document
embeddings as intentionally simpler), and align the Responses `_parse_response` except-set with the
chat engines. **Feature impact:** keeps-all.

### S11 — `from __future__ import annotations` inconsistency (COSMETIC)

**Where.** Present in `ops/{text,reason,compare,rank,embed}.py`; absent in `ops/primitives.py`,
`ops/__init__.py`, and all four provider engines. No functional effect (none rely on forward refs that
need it), but it's a visible inconsistency in sibling files.

**Make it symmetric.** Either add it everywhere or remove it from the five ops modules (they don't
need it). **Feature impact:** keeps-all.

---

## What is already symmetric and should be kept

- **The 5-arm error ladder and the provider error taxonomy are verbatim-symmetric.** Each provider
  defines the same six classes (`Error`, `APIError`, `AuthError`, `RateLimitError`, `ResponseError`,
  `TransportError`) subclassing the shared `providers/_client/errors.py` base, so all four engines'
  `except` ladders can (and should) collapse to one shared mapper without coupling (r1-07 PA-1). The
  ladder ordering (Auth/RateLimit before APIError) is identical everywhere.
- **`_retry_after` clamp is identical across all four engines** and correctly redundant with the
  `NonNegativeFiniteFloat` model field — one shared copy suffices.
- **Op signature shape and Symbol-wrapping are model-consistent** — every I/O op is
  `(runtime, source, …, *, engine=None)`, every deterministic op drops both, and every op wraps its
  result in `Symbol(...)`. This is the surface a user actually sees, and it is clean.
- **Decoder naming (`TextDecoder`/`ConstructorDecoder`/`TypeAdapterDecoder`/`PydanticDecoder`) and the
  runtime error hierarchy are cleanly symmetric.** `decode_output` is a single well-named free function;
  `DecodeError(ValueError)` is deliberately outside the `SymbolicAIRuntimeError` tree.
- **The ambient-runtime cutover is DONE and now symmetric with the tests** — production
  `runtime.py`/`errors.py`/`__init__.py` no longer carry `_CURRENT_RUNTIME`/`current_runtime`/
  `NoActiveRuntimeError`/root-`__all__` (verified 0 refs). The r1 asymmetry between code and test
  suite is resolved.

---

## Net recommendation (ranked)

1. **Fix the two bugs-in-waiting now** (cheap, real): S2 (`by_alias=True` on DeepSeek) and S1 (one
   shared `Runtime`/`RuntimeConfig` validator). Decide S3 (name-scope) in the same pass.
2. **Unify usage strictness** (S4): degrade to `usage=None` on inconsistency, or at least replace the
   DeepSeek exact cache-sum with a bound and make the three policies identical.
3. **Make capability enforcement data-driven and identical across engines** (S5, which also erases S9)
   — this is the single change that converts the "justified-but-hand-rolled-three-ways" divergence
   into "justified and declared once." Lands naturally with the shared adapter base (r1-07 PA-2/PA-3).
4. **Sweep the cosmetics** (S6–S11) opportunistically as the shared base / naming pass happens; none
   is load-bearing but together they remove the residue that keeps drifting.
