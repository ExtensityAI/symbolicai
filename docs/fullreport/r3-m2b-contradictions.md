# r3-m2b — Contradiction audit & adjudication

**Round 3 · meta/adversarial.** Reads the 14 `r1-*` and 8 `r2-*` reports ONLY (blind to any
`r3-*`), finds every place two reports recommend **conflicting actions** or state **conflicting
facts**, and adjudicates each with one recommendation + code evidence + feature-impact. Every
deciding fact was re-verified against the **live tree at HEAD `84f703b`** ("refactor: remove
legacy runtime and symbol surfaces"). Read-only; anchored by symbol + snippet.

---

## Executive summary

1. Most R1/R2 divergence is **framing, not substance** — the reports overwhelmingly converge on
   the same moves (provider `_engine/`/`_client/` bases, JSON-AST→`pydantic.JsonValue`, ops-helper
   hoist, naming). The genuine *contradictions* are concentrated in six load-bearing decisions.
2. **The sharpest real contradiction is the error-mapper home** (F2): `r2-b2` proposes the
   existing `_client/errors.py`; `r2-d1/d2/r1-07` require a **new `providers/_engine/`**. The
   boundary evidence is decisive — `_client` is runtime-blind (verified) and the mapper imports
   `runtime.errors`, so `_client` is **wrong**. `_engine/` wins, and it is *also* inventory-legal.
3. **Two forks turn on the design doc, and the "cut" camp is wrong on both** (F5, F7): the
   N-output collapse and the `decode_output` `default`/`limit`/`output_index` + `execute_many`
   trims each delete surface that `SYMBOL_REDESIGN.md` §6.1/§7/§11 **explicitly ratifies**. The
   spec-honoring reports (`r2-b2`, `r1-01b`) are correct; the trim reports (`r1-01a`, `r2-b1`)
   contradict the ratified surface. **Keep.**
4. **The spec-matrix fork** (F1: delete-dead-fields vs data-driven-gate) genuinely splits the ×2
   pairs (`b1`=delete vs `b2`=gate; `a1`=xor vs `a2`=gate). Resolution: **gate** (it removes the
   parallel hardcoded enforcement that deletion leaves behind), *conditional on* the `_engine/`
   base landing; delete `context_tokens` in either branch. Never ship the status quo.
5. **Decoder fork** (F6): delete `PydanticDecoder` (not in §7's 3-decoder set; ≡ `TypeAdapterDecoder`)
   and narrow `ConstructorDecoder` to scalar+bool (its container branch directly contradicts §7).
   `r1-08`'s lone "keep `PydanticDecoder`" is overruled. **logprobs** (F4) is a *consensus* product
   gate, not a contradiction — default **cut**.

---

## Adjudication table (the centerpiece)

Feature-impact legend: `keeps-all` / `drops-minimal` / `drops-real`. "Conf" = confidence in the
recommendation.

| # | Fork | Positions (holders) | Deciding evidence (verified live) | **ONE recommendation** | Feat. impact | Conf |
|---|------|---------------------|-----------------------------------|------------------------|--------------|------|
| **F1** | Spec matrix: **delete** dead fields vs **data-driven gate** | delete: `r1-06`C2, `r1-01a`F3, `r1-01b`S2, `r2-b1`W11(A) · gate: `r1-07`PA-2, `r2-c1`SYM-2, `r2-c2`S5, `r2-a2`M2, `r2-b2`#20 · xor: `r2-a1`M4 | `message_roles`/`content_types`/`response_formats`/`context_tokens` = **0 enforcement reads**; each `_validate_request` re-encodes the same facts by hand (DeepSeek rejects images ×3). Deletion removes the *pretense* but leaves the parallel hardcoded checks — the actual drift trap; the gate removes both. | **Build the gate** in `_engine/` (matrix = single source), **conditional on the `_engine/` base (F2/F3)**. Delete `context_tokens` (Language+Embedding) in **both** branches — no gate reads it. If `_engine/` is deferred, delete the dead fields as interim. Prereq: re-audit each provider's true declared roles/formats before trusting the matrix. **Never ship status quo.** | keeps-all (gate) / drops-minimal (delete) | med |
| **F2** | Error→runtime mapper home: `_client/errors.py` vs new `providers/_engine/` | `_client`: `r2-b2`#4; `r1-02`D1 ("`_client/` *or* runtime" — ambiguous) · `_engine/`: `r1-07`PA-1, `r2-d1`M1, `r2-d2`(central insight), `r2-a1/a2`M1 | `grep runtime symai/providers/_client/` → **0 hits** (runtime-blind, the memory's invariant). The mapper raises `runtime.errors.{AuthenticationError,RateLimitError,TransportError,ExecutionError,InvalidResponseError}` (`responses.py:12-18,132-148`). Homing it in `_client/` forces `_client → runtime`, breaking the verified boundary. | **New `providers/_engine/mapping.py`.** `r2-b2`'s `_client` placement is overruled — it optimized for the inventory pin but violates the boundary. No trade-off exists: `_engine/` is *both* boundary-correct and inventory-legal (F3). | keeps-all | high |
| **F3** | New `_engine/` dir vs `test_public_cutover.py:279` pinned inventory | pin-aware: `r2-b2`#3 · pin-silent (add new `_client/*.py` without noting): `r2-d1`, `r2-d2`, `r2-a1`, `r2-a2`, `r2-b1` | The test globs only `providers/_client/*.py` (=`{__init__,errors,headers,models}`), `providers/{openai}/client`, `.../engines`, and cerebras/deepseek `client`+`engines` — **not** top-level `providers/`. So a **new `providers/_engine/` package is unblocked**; but d1/d2's proposed **new `_client/{transport,client,settings}.py` DO break the pinned `_client` set.** | Put the mapper/base/gate/chat-adapter in the **unpinned `providers/_engine/`** (no test edit). For genuine new `_client` modules (`transport.py`, `client.py`, `settings.py`), **deliberately update the `_client` inventory assertion** — it is a greenfield spec, evolving it *with* the intended structure is legitimate (not drift). Only `r2-b2` caught this; the seam-map reports' target trees are compatible **only** after that one test edit. | keeps-all | high |
| **F4** | logprobs: **cut** request side vs **close** the loop | cut-default (all): `r1-06`C1, `r2-b1`W15, `r2-b2`#21, `r2-c1`SYM-8, `r2-a1/a2` | `SamplingConfig.logprobs`/`top_logprobs`/`logit_bias` forwarded + client DTOs parse returned logprobs, but `LanguageModelOutput` has **no** logprobs field. Not a cross-report *contradiction* — unanimous default. Providers may bill for `logprobs=true`. | **Cut the request side** (drop the 3 fields + `LogitBias` + `validate_unique_logit_bias_tokens` + `SamplingField.*LOGPROB*`/`LOGIT_BIAS` + client logprobs DTOs) **unless** product wants logprobs as a first-class result → then **close** (add `logprobs` to `LanguageModelOutput`, map in 3 `_output` builders). **Flag for owner; do not keep the half-state.** | drops-minimal (cut) | high (default) |
| **F5** | N-output **collapse** vs design-ratified `output_index` (don't-cut trap) — *compatible?* | collapse (+drop `output_index`): `r1-06`C3, `r2-b1`W13, `r2-a1`M5, `r2-a2`M8 · keep: `r2-b2`#22, `r1-01b`S6 | `SYMBOL_REDESIGN.md` §7 ratifies `output_index` ("output index selection is deterministic and raises `IndexError` when absent", l.268) and §11 lists "output-index decoding independently covered" (l.342). `output_index` is meaningful **only** if a response can hold >1 output; collapsing `outputs`→single `output` strips it. The dedup/sort also defensively validates untrusted provider payloads (a real, if rarely-hit, guard). | **NOT compatible as an unqualified cut. KEEP** the `outputs` tuple + `output_index` (spec-ratified, cheap, adds payload-validation). Collapse is valid **only** as a deliberate spec amendment (edit §7 + §11 to drop `output_index`), and then collapse `outputs` **and** `output_index` **together** — never one without the other (the incoherent half-state `r2-b2` names). The collapse camp under-weighted the ratification. | drops-real (collapse w/o spec change) → **keep** | high |
| **F6** | Decoders: delete `PydanticDecoder` vs keep the family; `ConstructorDecoder` narrow-vs-keep | delete/fold `PydanticDecoder`: `r1-04`#1, `r1-11`A14/D1, `r2-b1`W16, `r2-b2`#7, `r2-a1/a2`M6 · keep `PydanticDecoder`: `r1-08`("already good") · narrow `ConstructorDecoder`: `r1-04`#2, `r1-11`A15, `r2-b1/b2`, `r2-a1/a2` (no report defends the container branch) | §7 names exactly **3** decoders (Text/Constructor/TypeAdapter); `PydanticDecoder` is a 4th, absent from the spec, and ≡ `TypeAdapterDecoder` for any `BaseModel` (identical result+type). §7 states "nested/container typing uses `TypeAdapter`, not bare runtime classes" — directly condemning `ConstructorDecoder`'s `in (list,tuple,set,dict)` → `ast.literal_eval` branch (`decoding.py:57`). Ops instantiate only `TextDecoder()` + `ConstructorDecoder(bool)` (verified). | **Delete `PydanticDecoder`** by letting `TypeAdapterDecoder` accept `type[T] \| TypeAdapter[T]` (fold; removes the `TypeAdapter(...)` tax). **Narrow `ConstructorDecoder`** to scalar (`int`/`float`/callable) + `bool`; delete the container branch; route containers via `TypeAdapterDecoder`. `r1-08`'s "keep `PydanticDecoder`" is a lone, unargued dissent — overruled. | keeps-all (fold) / drops-minimal (container) | high |
| **F7** | `decode_output` `default`/`limit` + `execute_many`: **trim** (test-only) vs **keep** (spec-ratified) | trim: `r1-01a`F5/F6, `r1-01b`S6(B), `r2-b1`W14/W17 · keep: `r2-b2`T3/T4, `r1-01b`S6(A), `r1-04`#5 | §7 explicitly specs `default`/`limit` with documented rules (ll.247-270); §11 l.342 lists "default, limit, and output-index decoding are independently covered" as acceptance criteria; §6.1 documents `execute_many` as stable-order sequential (ll.202,220). These are **ratified forward-looking surface, not YAGNI**. | **Keep** `default`/`limit`/`output_index` on `decode_output` and **keep** `execute_many`. Only **reconcile** `execute_many`'s signature: §6.1 shows flat `Sequence[object]`; code is nested `Sequence[Sequence[object]]` — update the **doc** to the (better) nested form. The trim recommendations contradict the spec; `r2-b2`/`r1-01b`-A are correct. | keeps-all (keep) / drops-real:spec'd (trim) | high |

**Lesser / near-consensus divergences (not full forks):**

| # | Divergence | Holders | Resolution |
|---|-----------|---------|------------|
| L1 | Public registry rename target: `symai/builtins.py` vs `symai/registry.py` | `builtins.py`: `r2-d1/a1/a2/b*` · `registry.py`: `r2-d2` · both: `r1-03b`B5 | Cosmetic. Prefer **`registry.py`** (names the builtin *registry*); either is keeps-all. Keep the generic-loader rename (`load_runtime`→`build_runtime`/`compose_runtime`) — unanimous intent. |
| L2 | `operations.py` destination: top-level `requests.py` vs `runtime/requests.py` | `runtime/requests.py`: `r1-03b`, `d1`, `d2`, `a1`, `a2` · either: `r1-03a`B4 | **`runtime/requests.py`** — it imports only `runtime.models`. Near-consensus; no real conflict. |
| L3 | Usage-consistency: degrade `usage=None` vs relax-to-bounded vs keep fail-loud | degrade (all): `r1-07`PA-5, `r2-c1`SYM-1, `r2-c2`S4, `r2-b1`W21, `r2-b2`#24 | Consensus default **degrade to `usage=None`** (or at minimum relax DeepSeek's exact `cache_hit+cache_miss==prompt` to `<=` and unify). Design-stance decision, not a contradiction. |
| L4 | `ImplementationId` no-op cast: drop brand vs `StrEnum` of builtins | drop: `a1/a2/b*/d*` · either: `r1-08`API-04 | Drop the no-op `cast("ImplementationId", …)`. `StrEnum` is an optional nicety, not a conflicting claim. |

---

## Per-fork detail (evidence + feature-impact)

### F1 — Spec matrix: delete vs gate (the ×2 pairs genuinely split)

**The disagreement is real, not framing.** The two `b` reports (same goal, run blind) land on
**opposite** recommendations: `r2-b1` W11 "**RECOMMEND delete** … rebuild as the gate later";
`r2-b2` #20 "**Recommend (b)** [gate] for the best end-state." The two `a` reports differ in
posture: `r2-a1` Move 4 frames it as "gate **xor** delete … never both"; `r2-a2` Move 2 takes a
"**definite side**: make it authoritative." R1 splits too: `r1-06`/`r1-01a`/`r1-01b` say delete,
`r1-07` says gate.

**Deciding evidence.** Verified live: `model_spec.{context_tokens,message_roles,content_types,response_formats}`
have **0 enforcement reads** across `symai/`. Enforcement is instead a hand-rolled wall in each
`_validate_request` (e.g. DeepSeek rejects `ImageContent` in the matrix, in `_validate_request`,
**and** in `_message` — three encodings of one fact; `r2-c2` S9). The key asymmetry the delete
camp under-weights: **deleting the dead fields does not remove the drift trap** — it removes the
inert *data* but leaves the parallel hardcoded checks that are the actual second representation.
The gate removes **both** (data becomes the single source; ~60 LOC of hardcoded checks collapse to
a `_validate_provider_specifics` hook).

**Resolution.** Build the **data-driven gate** in the `_engine/` base — but this is *unlocked by*
F2/F3 (the gate needs the shared base to avoid re-duplicating ×4; `r2-a2`/`r2-b2` are right that it
depends on Move 1). `context_tokens` is dead in **both** branches (no gate reads a token budget) —
delete it from `LanguageModelSpec` *and* `EmbeddingModelSpec` unconditionally. If the `_engine/`
base is deferred, fall back to **delete** the dead fields + `MessageRole`/`ContentType`/
`ResponseFormatType` enums as the interim (they discriminate nothing — the message/format models
use `Literal[...]`). **Prerequisite for the gate:** re-audit each provider's declared membership
(openai/cerebras set `message_roles = tuple(MessageRole)` incl. `DEVELOPER` — confirm that is
true before making it authoritative). **Feature impact:** keeps-all (gate) / drops-minimal (delete).
**Conf:** med — the "don't keep status quo" is high-confidence; gate-over-delete is medium and
contingent on the base landing + wanting a `Runtime.capabilities()` introspection surface (none
exists today).

### F2 — Error-mapper home: the one placement that breaks the boundary

**Contradiction.** `r2-b2` item 4 explicitly homes the error→runtime mapper in the **existing
`_client/errors.py`** ("a pinned-but-existing file — no new file, no inventory-test edit"). `r2-d2`'s
central insight and `r2-d1` M1 / `r1-07` PA-1 require a **new `providers/_engine/mapping.py`** and
name `_client` as the wrong home. `r1-02` D1 straddles it ("`providers/_client/` (or `runtime`)").

**Deciding evidence.** `grep` for `runtime` in `symai/providers/_client/` → **0 hits**; the client
tier is deliberately runtime-blind (the memory's "client = faithful API binding that never knows
symai"). The mapper's whole job is to raise `runtime.errors.*` — verified in `responses.py:12-18`
(`from symai.runtime.errors import AuthenticationError, …`) and `:132-148`. Putting that function in
`_client/errors.py` forces `_client → symai.runtime.errors`, breaking the verified invariant and the
`test_public_facades` client-isolation guard. `r2-b2` traded the boundary for the inventory pin —
but that trade is unnecessary (F3): the mapper's correct home, a new `providers/_engine/`, is *also*
inventory-legal. **Recommendation:** `providers/_engine/mapping.py`, catching the shared
`_client.errors.*` base classes (safe — every provider error subclasses them) and raising
`runtime.errors.*`. **Feature impact:** keeps-all (message wording preserved via a `display` arg).
**Conf:** high.

### F3 — Does the seam map break `test_public_cutover.py:279`? Partially — and only `r2-b2` noticed.

**The gap.** `r2-d1` and `r2-d2` present target trees that add **new files to `_client/`**
(`_client/transport.py`, `_client/client.py`, `_client/settings.py`) and a new `_engine/` package.
`r2-a1`/`r2-a2`/`r2-b1` also assume new shared files freely. Only `r2-b2` #3 flagged that
`test_public_cutover.py:279-312` **pins exact file sets**.

**Deciding evidence (verified).** The test asserts
`{p.name for p in (PACKAGE/"providers/_client").glob("*.py")} == {"__init__.py","errors.py","headers.py","models.py"}`
and equivalent exact sets for `openai/client`, `openai/engines`, and cerebras/deepseek
`client`+`engines`. It does **not** glob top-level `providers/`. Therefore:
- A new **`providers/_engine/`** package (mapper, base, gate, chat-adapter) is **unblocked** — no
  test change. This is where F2's mapper and the `BaseHttpEngine`/gate/`ChatCompletionsAdapter` go.
- New **`_client/{transport,client,settings}.py`** files **break the pinned `_client` set** — the
  assertion must be updated.

**Resolution / sequencing.** (1) Land the `_engine/` package first — inventory-legal as-is, and it
is the top-ranked move across every r2 report. (2) For the `_client` base additions, **deliberately
update the pinned `_client` inventory** in the test. It is a greenfield executable spec; extending
it to match the *intended* structure is a legitimate spec edit, not the drift the test guards
against. (Folding transport/settings into existing `_client/{models,headers,errors}.py` to dodge the
pin is possible but a `BaseClient` shell and base transport envelope deserve their own modules.) The
seam-map target trees are correct **modulo** this one test update. **Feature impact:** keeps-all.
**Conf:** high on the facts; the "edit-the-test vs fold-into-existing" is a judgment (recommend edit).

### F4 — logprobs: consensus cut, not a contradiction

Every report that touches it (`r1-06`C1, `r2-b1`W15, `r2-b2`#21, `r2-c1`SYM-8, `r2-a1/a2`) reaches
the **same** default: cut the request side unless logprobs is an intended first-class result; never
keep the half-state. This is a **product gate**, not a cross-report conflict. Verified: request fields
forwarded + client DTOs parse returned logprobs, but `LanguageModelOutput` has no field to hold them.
**Recommendation:** cut (drop `logprobs`/`top_logprobs`/`logit_bias` + `LogitBias` +
`validate_unique_logit_bias_tokens` + `SamplingField` members + client DTOs); close the loop only if
product declares logprobs a feature. **Flag for owner.** **Feature impact:** drops-minimal.
**Conf:** high (default).

### F5 — N-output collapse vs ratified `output_index`: incompatible without a spec change

**Contradiction.** `r1-06`C3 / `r2-b1`W13 / `r2-a1`M5 / `r2-a2`M8 recommend collapsing
`LanguageModelResponse.outputs: tuple[...]` → a single `output` and dropping `output_index`.
`r2-b2` #22 ("leans keep — this is a near-trap") and `r1-01b` S6 keep it.

**Deciding evidence.** `SYMBOL_REDESIGN.md` §7 ratifies `output_index` with an explicit rule
("output index selection is deterministic and raises `IndexError` when absent", l.268) and §11
l.342 makes "output-index decoding" an acceptance criterion. `output_index` is only meaningful when
a response can carry >1 output — so collapsing `outputs` necessarily strips a **ratified** parameter.
Additionally the per-choice dedup/sort defends against malformed provider payloads (`>1` choice
returned unexpectedly), a real guard even though no *request* can ask for N>1.

**Resolution.** **Keep** the tuple + `output_index` (spec-honoring, cheap). The collapse is coherent
*internally* (the collapse camp does drop `output_index` alongside), but it **contradicts the ratified
spec** and is only admissible as a **deliberate amendment** to §7 + §11 — at which point `outputs`
and `output_index` are removed **together**. The one thing no one should do is collapse `outputs`
while leaving `output_index` (an incoherent half-state). The collapse camp under-weighted the
ratification; `r2-b2`/`r1-01b` are correct. **Feature impact:** keep = keeps-all; collapse-without-
spec-change = drops-real:spec'd. **Conf:** high.

### F6 — Decoders: delete `PydanticDecoder`, narrow `ConstructorDecoder`

**Contradiction.** `r1-08` lists `PydanticDecoder(User)` under "**what is already good — keep**"
("a reasonable convenience"). Six other reports (`r1-04`#1, `r1-11`A14, `r2-b1`W16, `r2-b2`#7,
`r2-a1`M6, `r2-a2`M6) delete/fold it. No report defends `ConstructorDecoder`'s container branch.

**Deciding evidence.** §7 enumerates exactly three decoders (Text/Constructor/TypeAdapter) and its
model example routes through `TypeAdapterDecoder(TypeAdapter(list[User]))`; `PydanticDecoder` is
undocumented 4th surface that is functionally identical to `TypeAdapterDecoder` for any `BaseModel`.
§7 also states "nested/container typing uses `TypeAdapter`, not bare runtime classes" — the
`ConstructorDecoder` `in (list,tuple,set,dict)` → `ast.literal_eval` branch (`decoding.py:57`) is a
direct violation, and it has no op consumer (ops use only `TextDecoder()` + `ConstructorDecoder(bool)`,
verified). Its Python-literal grammar (`['a']`) also clashes with `TypeAdapter`'s JSON (`["a"]`).

**Resolution.** **Delete `PydanticDecoder`** by letting `TypeAdapterDecoder` accept
`type[T] | TypeAdapter[T]` and wrap once (removes the `TypeAdapter(...)` boilerplate from the
design's own examples). **Narrow `ConstructorDecoder`** to scalar (`int`/`float`/custom callable) +
`bool`; delete the container branch; route containers through `TypeAdapterDecoder` per §7. `r1-08`'s
"keep" is unargued against the fold and is overruled. (Bonus, uncontested consensus: move
`_normalize_text`'s single-quote strip out of the shared path into the scalar/bool path — it
silently mutates faithful text, `'Twas…'`→`Twas…`.) **Feature impact:** keeps-all (fold) /
drops-minimal (container branch). **Conf:** high.

### F7 — `decode_output` params + `execute_many`: keep (they're ratified), don't trim

**Contradiction.** `r1-01a` F5/F6 and `r2-b1` W14/W17 recommend trimming `decode_output`'s
`default`/`limit`/`output_index` (+`_limit_value`, `Missing`) and removing `execute_many` as
"test-only." `r2-b2` T3/T4 and `r1-01b` S6-A explicitly keep them as ratified; `r1-04`#5 keeps
`execute_many` (only reconcile the signature).

**Deciding evidence.** §7 specs `default`/`limit` with documented semantics (ll.247-270); §11 l.342
makes "default, limit, and output-index decoding" acceptance criteria; §6.1 documents `execute_many`
(ll.202,220). These are **forward-looking infrastructure the design explicitly asked for** — the
YAGNI argument does not apply to ratified surface (and per the project's own rule, "YAGNI applies to
speculative features, not forward-looking infrastructure explicitly asked for").

**Resolution.** **Keep** `default`/`limit`/`output_index` and `execute_many`. The single legitimate
change is **doc/code reconciliation** of `execute_many`: §6.1 shows a flat `Sequence[object]`; the
code takes nested `Sequence[Sequence[object]]` and splats `*values` (the more correct, multi-value
form). Update the **design doc** to the nested form. The trim recommendations contradict the ratified
spec. **Feature impact:** keep = keeps-all; trim = drops-real:spec'd. **Conf:** high. (Overlaps F5 on
`output_index`.)

---

## Cross-report pattern

The `×2` design surfaced the two forks it was meant to (F1, and the F5/F7 spec questions): where two
reports on the *same* goal ran blind, `b1`/`b2` split on the spec matrix and (implicitly, via
different trap-classification) on the decode/N-output trims. The adjudication resolves each **against
the ratified `SYMBOL_REDESIGN.md`** where the doc is explicit (F5, F6, F7) and **against the verified
boundary invariant** where it is architectural (F2, F3). Only F1 remains a genuine judgment call
(gate vs delete), and even there both endpoints beat the status quo.

## What all reports agree on (no contradiction — record for the synthesis)

The `_engine/`/`_client/` shared bases (highest-value move), the JSON-AST→`pydantic.JsonValue`
replacement, ops-helper hoist into `ops/primitives.py`, `operations.py`→`runtime/requests.py`,
the two-`load_runtime` rename, the shared `Runtime`/`RuntimeConfig` validator, DeepSeek `by_alias=True`,
and every Group-3 "trap" (keep multimodal `image_request`/`data_uri`/`ImageContent`/`vision`; keep all
9 `TokenUsage` fields + `RateLimitMetadata`; keep per-provider `MODEL_SPECS`/`_normalized_model_spec`
/wire schemas; keep OpenAI Responses standalone; keep `Symbol`'s operator dunders) are **convergent
across R1 and R2** — those are settled, not forks.
