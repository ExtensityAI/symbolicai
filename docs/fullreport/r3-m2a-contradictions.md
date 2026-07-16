# r3-m2a — Contradiction audit & adjudication

**Role:** META / contradiction auditor. Read only the 14 `r1-*` and 8 `r2-*` reports (no
`r3-*`), found every place two reports recommend conflicting actions or assert conflicting
facts, and resolved each against **live code at HEAD `84f703b`**, the design
(`audit/SYMBOL_REDESIGN.md`), and the test suite. Read-only; anchors are symbol + snippet +
the exact verification command output.

**State confirmed first (so resolved seeds aren't re-litigated):** root `symai/__init__.py`
is 0 bytes; `prompts.py`/`backend/` gone; `_CURRENT_RUNTIME`/`current_runtime`/
`NoActiveRuntimeError`/`static_context`/`dynamic_context` have **0 production refs**;
`jinja2`/`python-box` dead-code-free. Every R1 finding about those is **RESOLVED** and the R2
reports agree — not a contradiction.

The contradictions cluster in six named forks plus three smaller ones. Most cross-report
disagreement is **not** about facts (the reports agree on what the code does) — it is about
*which fix* and *where it lives*. Two forks (5 and 6) are cases where a subset of R1 reports
recommend **cutting design-ratified surface**; the deciding evidence is the design doc itself.

---

## Adjudication table

| # | Fork | Position A (holders) | Position B (holders) | **Verdict** | Conf |
|---|------|----------------------|----------------------|-------------|------|
| 1 | `LanguageModelSpec` dead matrix | **Delete** 4 fields + 3 enums (r1-01a F3, r1-01b S2, r1-06 C2, r2-b1 W11) | **Data-driven gate** makes matrix authoritative (r1-07 PA-2, r2-a2 Move2, r2-b2 item20, r2-c1 SYM-2, r2-c2 S5) | **Gate is the target end-state, done inside the `_engine/` base; Delete is the fallback if that base is deferred. Never ship the populated-but-unread status quo.** | high |
| 2 | Error→runtime mapper home | **`_client/errors.py`** (existing file) (r2-b2 item4); "`_client/` or runtime" (r1-02 D1) | **New `providers/_engine/`** — `_client` must stay runtime-blind (r1-07 PA-1, r2-a1/a2, r2-d1 M1, r2-d2 §central) | **`providers/_engine/mapping.py`. Homing it in `_client/` is a real boundary violation.** r2-b2 is wrong on this point. | high |
| 3 | New shared files vs pinned inventory | Add new `_client/*.py` files: `transport.py`,`client.py`,`settings.py` (r2-d1/d2 target trees) | Fold into **existing** `_client` files; new classes go to **unpinned** `_engine/` (r2-b2 item3-5) | **r2-d1/d2's new `_client/*.py` files DO break `test_deleted_production_tree_and_adapter_inventory`. Runtime-aware shared code → new `_engine/` (unpinned, safe). Runtime-blind client code → fold into existing `_client/{models,headers,errors}.py`, or update the inventory pin deliberately.** | high |
| 4 | logprobs | Cut request side (r2-b2 item21) | Cut **or** close loop, product-gated (r1-06 C1, r2-a2/b1/c1) | **Not a real contradiction — both default to cut.** Cut `logprobs`/`top_logprobs`/`logit_bias` unless the owner declares logprobs a first-class result; then close the loop. | high |
| 5 | N-output collapse vs `output_index`/`decode_output` params | **Collapse** to single `output`, trim `decode_output` (r1-01a F5, r1-06 C3, r2-a1 M5, r2-a2 M8, r2-b1 W13/W14) | **Keep** — `output_index`/`default`/`limit`/`execute_many` are design-ratified (r2-b2 items22/T3/T4) | **KEEP.** `decode_output(output_index/default/limit)` and `execute_many` are ratified (SYMBOL_REDESIGN §6.1/§7). Collapsing `outputs`→`output` while keeping `output_index` is an incoherent half-state. r2-b2 is correct; the cut recommendations conflict with the spec. | high |
| 6 | Decoder layer | **Delete `PydanticDecoder`** / narrow `ConstructorDecoder` (r1-04 #1/#2, r1-11 A14, all r2) | **Keep the decoder family** incl. `PydanticDecoder` as convenience (r1-08 "already good") | **Fold `PydanticDecoder` into `TypeAdapterDecoder(bare type[T])`** (preserves the single-model ergonomics r1-08 wants AND collapses to one-decoder-per-concept per §7). **Narrow `ConstructorDecoder`** — its container branch contradicts §7 line 272. | high |
| 7 | `execute_many` | Remove (test-only) (r1-01a F6) | Keep, ratified (r2-b2 T4) | **Keep; reconcile the signature** (design flat `Sequence[object]` vs live nested `Sequence[Sequence[object]]`). Same class as Fork 5. | high |
| 8 | Structural `ModelSpec`/`ReasoningSpec` shape hoist | Hoist shape → `_client/models.py` (r1-02 D9, r2-b1 W9, r2-b2 item13, r2-d1 M10) | **Leave per-provider** — payoff below noise floor, shapes differ (r2-d2 optional §) | **Low-stakes; lean r2-d2 (leave or defer).** The shapes genuinely differ (cerebras has no `vision`; openai defaults `True`; deepseek requires it), so a shared base needs `vision` optional for ~15-20 LOC. Do it only opportunistically inside the Fork-1/2 pass. | med |
| 9 | Public registry module rename | `symai/builtins.py` (r1-03b, r2-a1) | `symai/registry.py` (r2-d2) | **Cosmetic; pick `registry.py`** (names it for what it is). Keep the public entry `load_runtime`; rename only the *generic* `runtime/loading.py::load_runtime`→`compose_runtime`/`build_runtime`. | low |

---

## Fork 1 — `LanguageModelSpec`: delete dead fields vs data-driven gate

**Positions.** All reports agree on the *facts* (see verification below). They split on the *fix*:
- **Delete** (r1-01a F3, r1-01b S2, r1-06 C2, r2-b1 W11): drop `context_tokens`,
  `message_roles`, `content_types`, `response_formats` + the `MessageRole`/`ContentType`/
  `ResponseFormatType` enums; keep the hardcoded per-engine `_validate_request` checks.
- **Gate** (r1-07 PA-2, r2-a2 Move 2, r2-b2 item 20, r2-c1 SYM-2, r2-c2 S5): make the matrix
  the single enforcement source via a shared `_gate_capabilities`, deleting ~60 lines of
  parallel hardcoded checks; each engine keeps only a `_validate_provider_specifics` hook.
- The two synthesis-of-synthesis reports openly disagree: **r2-b1 recommends "A (delete) now"**;
  **r2-b2 recommends "B (gate)"**; r2-a1 says "gate xor delete, never both"; r2-a2 sides with gate.

**Deciding evidence (verified live).** The dead-field claim is real:
```
$ grep -rn "\.message_roles\|\.content_types\|\.response_formats" symai/ | grep -v test
   (no output — zero enforcement reads)
$ grep -rn "model_spec.context_tokens" symai/ | grep -v test
   (none; the only context_tokens hits read the *client* spec while populating)
```
`MessageRole`/`ContentType`/`ResponseFormatType` appear only in the three engine builders
(populating the dead fields) + `runtime/models.py`. So both camps' shared premise holds.

The distinguishing facts: (a) the gate's payoff is **not** conditional on a
`Runtime.capabilities()` introspection API — r2-b1 undersells it here. `Runtime` never exposes
`model_spec`, true, but the gate's value is removing the **parallel-representation drift trap**
(deepseek encodes "no images" in *three* places; declares `sampling_fields` then hardcodes the
same rejections) — that value is real independent of introspection, and r1-07/r2-c1/r2-c2 verify
it. (b) The gate has a genuine prerequisite the delete-camp is right to flag: it needs the shared
`_engine/` base to live in (else it re-duplicates ×4 — r2-a2 Move 2, r2-b2 item 20), and the
matrix must be *accurate* before it becomes authoritative. I spot-checked the sharpest risk —
OpenAI declares `message_roles=_ALL_MESSAGE_ROLES` (all roles incl. `DEVELOPER`) and does **not**
hardcode-reject developer messages (`responses.py:250 role=message.role`), so for roles the
matrix is accurate; a per-provider capability re-audit is still owed before flipping the switch.

**Recommendation.** The **gate** is the better end-state and is unlocked by the `_engine/` base
that every synthesis report already ranks #1 — so build it there, after a per-provider capability
re-audit. If the `_engine/` base is deferred, **delete the dead fields now** (high-confidence,
nothing reads them). The reports are not truly irreconcilable once sequenced: delete is the
fallback, gate is the target; the only forbidden option is shipping the status quo. `context_tokens`
dies in both branches (no reader in either). **Feature impact: keeps-all** (gate) / **drops-minimal**
(delete — loses only never-read introspection data). **Conf: high.**

---

## Fork 2 — Error→runtime mapper: `_client` vs `_engine` (is `_client` a boundary violation?)

**Positions.**
- **`_client/errors.py`** (r2-b2 item 4, explicitly): "Home it in the existing `_client/errors.py`
  … no new file, no inventory-test edit." r1-02 D1 is ambiguous — "`providers/_client/` (or runtime)".
- **`providers/_engine/`** (r1-07 PA-1, r2-a1 Move 1, r2-a2 Move 1, r2-d1 M1, r2-d2 §central):
  a new runtime-aware tier. **r2-d2 explicitly rebuts r2-b2**, calling the `_client/` home "the
  trap — `_client/` is wrong," because the mapper translates `client_errors.*` → `runtime.errors.*`
  and `_client` must never import `runtime`.

**Deciding evidence (verified live).**
```
$ grep -rn "from symai.runtime\|import symai.runtime" symai/providers/_client/
   (no output — _client/ is runtime-blind by construction)
```
The mapper constructs `AuthenticationError`/`RateLimitError`/`TransportError`/`ExecutionError`/
`InvalidResponseError` — these are `symai.runtime.errors` classes; the mapper **must** import
`runtime.errors` to raise them. Homing it in `_client/errors.py` therefore introduces the *first*
`runtime` import into `_client/`, destroying the property that r1-03a/03b/r2-d2 all verify and that
the user's own architectural principle names ("client = faithful API binding that never knows
symai; engine = the only crossing point").

Nuance that must be stated precisely: the existing guard
`tests/providers/test_public_facades.py::test_provider_client_packages_do_not_import_symbolic_runtime_layers`
walks each **provider** `client/` package but **not** `providers/_client/` — so r2-b2's placement
would *pass current tests*. It is a boundary violation the test net happens not to cover, not a
test failure. That makes r2-b2's argument ("no test churn") technically true and architecturally
wrong: it optimizes for test-avoidance over the ratified layering, which the greenfield stage
explicitly de-prioritizes.

**Recommendation.** **`providers/_engine/mapping.py`.** Every provider error already subclasses the
shared `_client/errors.py` bases, so one function catching the base classes is zero-schema-coupling
(the reason r2-b2 wanted it "shared" is fully satisfied). `_engine/` is the correct runtime-aware
tier and is unpinned, so it also sidesteps Fork 3. Extend the boundary test to also walk
`_client/` (it should have all along). **Feature impact: keeps-all. Conf: high.**

---

## Fork 3 — New `providers/_engine/` dir vs the pinned inventory test

**Positions.** r2-d1/d2's target trees add **new files to `_client/`**: `_client/transport.py`,
`_client/client.py`, `_client/settings.py`. r2-b2 (items 3-5, and its exec-summary §3) warns that
`test_public_cutover.py` pins the exact `_client` file set, so new shared files must land in a
**new unpinned `_engine/`** package or the inventory test must be edited.

**Deciding evidence (verified live).** `test_deleted_production_tree_and_adapter_inventory`:
```python
assert {path.name for path in (PACKAGE / "providers/_client").glob("*.py")} == {
    "__init__.py", "errors.py", "headers.py", "models.py",
}
```
It also pins each provider's `client/` and `engines/` dirs to exact file sets. It does **not** glob
`providers/` top-level or any `providers/_engine/`.

Therefore:
- **r2-d1/d2's proposal to add `_client/transport.py`, `_client/client.py`, `_client/settings.py`
  WOULD break this test** — an unflagged consequence in both d-reports.
- **A new `providers/_engine/` package does NOT break it** — the glob never looks there. r2-b2 is
  correct.
- Folding shared code into the **existing** `_client/{models,headers,errors}.py` also does not
  break it (filename set unchanged).

**Recommendation (right sequencing).**
1. Runtime-aware shared code (error mapper, `BaseHttpEngine`, capability gate, `ChatCompletionsAdapter`)
   → **new `providers/_engine/`** (unpinned + correct boundary; resolves Forks 2 & 3 together).
2. Runtime-blind shared *client* code (base `ResponseMetadata`/`APIResponse`, header constants,
   error `__init__` bodies) → **fold into the existing `_client/{models,headers,errors}.py`**
   (r2-b2's route) to avoid inventory churn.
3. If a genuinely new `_client` file is warranted (e.g. a substantial `BaseClient` that doesn't fit
   `models.py`), **update the inventory assertion deliberately** as part of that change — the test is
   an executable spec, and this is greenfield; editing it consciously is fine. What is *not* fine is
   r2-b2's specific dodge of putting the runtime-aware mapper in `_client` to avoid the edit (Fork 2).

**Feature impact: keeps-all. Conf: high.**

---

## Fork 4 — logprobs: cut vs close-loop-or-cut

**Positions.** r2-b2 (item 21) "recommend cut." r1-06 (C1) "cutting is the lower-risk default unless
the product wants logprobs as a first-class result." r2-a2/b1/c1 all "default cut, flag owner."

**Deciding evidence.** This is a **near-non-contradiction**: every report defaults to cut and gates
the alternative on explicit product intent. Verified: `SamplingConfig.logprobs`/`top_logprobs`/
`logit_bias` are forwarded and the client DTOs parse returned logprobs, but `LanguageModelOutput`
has no field to hold them — a genuine request/response coherence hole, not in the design doc.

**Recommendation.** **Cut the request side** (`logprobs`/`top_logprobs`/`logit_bias` + `LogitBias`
+ `validate_unique_logit_bias_tokens` + the `SamplingField.*LOGPROB*`/`LOGIT_BIAS` members + client
logprobs DTOs) **by default**; the *only* thing that flips it to "close the loop" is an owner decision
to make logprobs a first-class result. Do not ship the half-state. **Feature impact: drops-minimal**
(cut). **Conf: high** (that it's an owner call, not an inter-report conflict).

---

## Fork 5 — N-output collapse vs the ratified `output_index`/`decode_output`/`execute_many` surface

**Positions.**
- **Collapse/trim** (r1-01a F5, r1-01b S6, r1-06 C3, r2-a1 Move 5, r2-a2 Move 8, r2-b1 W13/W14):
  `LanguageModelResponse.outputs: tuple` → single `output`; drop `index`/dedup/sort and
  `decode_output`'s `output_index`/`default`/`limit`/`_limit_value`; "no `n` request field exists,
  so it's all dead."
- **Keep** (r2-b2 items 22, T3, T4): `output_index`/`default`/`limit` and `execute_many` are
  **explicitly ratified** forward-looking surface; cutting them drops spec'd capability.

**Deciding evidence (the design doc settles it).** `audit/SYMBOL_REDESIGN.md` §7:
```python
def decode_output(response, decoder, *, output_index: int = 0,
                  default: T | Missing = MISSING, limit: int | None = None) -> T: ...
```
plus explicit "Decoder rules": *"default catches only the documented decode failure"*, *"output
index selection is deterministic and raises `IndexError` when absent"*, *"collection limiting is
post-decode and deterministic"*, *"sets pass through"* — and §11: *"scalar, boolean, Pydantic model,
nested container, default, limit, and output-index decoding are independently covered."* Live code
matches (`decoding.py:97-107`). So these are ratified, not YAGNI.

The internal-coherence clincher: `_output_text(response, output_index)` scans
`LanguageModelResponse.outputs` by `LanguageModelOutput.index` (live: `outputs: tuple[...] =
Field(min_length=1)`, `index: int = Field(ge=0)`). **Collapsing `outputs`→single `output` while
keeping the ratified `output_index` is self-contradictory** — exactly r2-b2's "incoherent half-state"
point. And per Fork 7, `execute_many` is ratified in §6.1 too.

**Recommendation.** **KEEP** the outputs-tuple + `index` + `output_index`/`default`/`limit` +
`execute_many`. The engine-side choice dedup/sort is defensive machinery for an N>1 case the request
layer can't currently produce, but it is cheap and guards untrusted provider payloads; removing the
*response* modeling is a **deliberate spec change** that must simultaneously drop `output_index` — do
that only if the owner ratifies dropping multi-output from the design, not as a "dead code" cut. The
r1-01/r1-06/r2-a1/r2-a2/r2-b1 collapse recommendations **conflict with the ratified surface** and
should be declined. **Feature impact: keeps-all** (keep) / **drops-ratified** (collapse). **Conf: high.**

---

## Fork 6 — Decoder layer: delete `PydanticDecoder` / narrow `ConstructorDecoder` vs keep the family

**Positions.**
- **Delete/narrow** (r1-04 #1/#2, r1-11 A14/A15, r2-a1 Move 6, r2-a2 Move 6, r2-b1 W16, r2-b2 items
  7/8): `PydanticDecoder` is a strict subset of `TypeAdapterDecoder`; `ConstructorDecoder`'s
  `list/tuple/set/dict` branch has no op consumer and clashes with `TypeAdapter`.
- **Keep the family** (r1-08 "What is already good"): the `*Decoder` set incl. `PydanticDecoder(User)`
  "as sugar" is a reasonable, consistent public surface.

**Deciding evidence (design doc).** SYMBOL_REDESIGN §7 enumerates exactly **three** standard decoders
— its examples use `TextDecoder()`, `ConstructorDecoder(int)`, `TypeAdapterDecoder(TypeAdapter(list[User]))`
— and §7 line 272 states *"nested/container typing uses `TypeAdapter`, not bare runtime classes."*
Live `decoding.py` ships a **fourth**, `PydanticDecoder` (class at line 87), unsanctioned by the spec
(r1-11 A14 confirms). And `ConstructorDecoder.decode`'s container branch (`ast.literal_eval` for
`list/tuple/set/dict`) **directly contradicts** §7 line 272. Verified: ops instantiate only
`TextDecoder()` and `ConstructorDecoder(bool)`; `PydanticDecoder` and the container branch have zero
non-test consumers.

**Recommendation.** These are reconcilable — r1-08's concern is *ergonomics of the single-model case*,
which the fold preserves:
- **Fold `PydanticDecoder` → `TypeAdapterDecoder`** by letting `TypeAdapterDecoder` accept a bare
  `type[T] | TypeAdapter[T]`. This keeps `TypeAdapterDecoder(User)` one-liner sugar (r1-08's want) AND
  collapses to one-decoder-per-concept (design §7). r1-08's "keep" and r1-04's "delete" both get what
  they actually care about.
- **Narrow `ConstructorDecoder`** to scalar + `bool`; route containers through `TypeAdapterDecoder`
  (design-aligned per §7 line 272, not merely a preference).
- Independent bonus both camps miss consensus on but all endorse: move `_normalize_text`'s single-quote
  stripping out of the shared path into the scalar/bool decode path (it silently mutates faithful
  `TextDecoder` output, e.g. `'Twas…'`→`Twas…`).

**Feature impact: keeps-all** (Pydantic + scalar/bool decoding preserved; containers still decode via
TypeAdapter's JSON grammar). **Conf: high.**

---

## Forks 7-9 (smaller, for completeness)

**Fork 7 — `execute_many`.** r1-01a F6 "remove (test-only)" vs r2-b2 T4 "keep, ratified." Design §6.1
documents `execute_many` as *"stable-order sequential execution."* → **Keep.** But the design signature
is `inputs: Sequence[object]` (flat) while live `function.py:52` is `Sequence[Sequence[object]]`
(nested) — r1-04 #5. **Reconcile the signature** (keep the nested impl, amend the doc, or vice-versa);
do not delete. **Conf: high.**

**Fork 8 — structural `ModelSpec`/`ReasoningSpec` shape hoist.** r1-02 D9 / r2-b1 W9 / r2-b2 item 13 /
r2-d1 M10 say hoist the frozen dataclass *shape* to `_client/models.py` (keeps-all, ~15-20 LOC). r2-d2
says **leave per-provider** — the shapes genuinely differ (cerebras `ModelSpec` has no `vision`; openai
defaults `True`; deepseek requires it), so a shared base needs `vision` optional for a payoff "below the
noise floor," and it introduces a type spanning three provider catalogs. **Lean r2-d2** (leave/defer);
if done, only opportunistically inside the Fork-1/2 provider pass, and it does **not** break the
inventory test (`models.py` is existing). Low stakes. **Conf: med.**

**Fork 9 — public registry module name.** `symai/builtins.py` (r1-03b, r2-a1) vs `symai/registry.py`
(r2-d2). Cosmetic; both keep the public entry `load_runtime` and rename only the *generic*
`runtime/loading.py::load_runtime`. Pick `registry.py`. **Conf: low.** (The `operations.py` →
`runtime/requests.py` move and the generic-loader rename are otherwise consensus, not contradictions;
r1-03a's alternative top-level `requests.py` is the lone minor divergence and `runtime/requests.py`
wins on cohesion — its only import is `runtime.models`.)

---

## What was NOT a contradiction (agreement, recorded to prevent re-litigation)

These appear in many reports and **agree** — do not mistake overlap for conflict: usage-consistency
→ degrade to `usage=None` (r1-07 PA-5 = r2-c1 SYM-1 = r2-b1 W21); ops `_symbol_value`/`_require_text`
→ `ops/primitives.py` (F8=S5=D7); JSON `JsonObject`/`JsonArray`/`JsonEntry` AST → `pydantic.JsonValue`
(F4=S1=C4, gated only on "is deep-freeze an invariant?" — verified nothing consumes it); `Runtime`
vs `RuntimeConfig` validator divergence → one shared validator, reject outer whitespace (R5-3=SYM-3=S1);
DeepSeek `by_alias=True` (SYM-2 c1 vs S2 c2 — same fix); engine-name global-uniqueness (R5-4=SYM-4=S3);
keep `Symbol` dunders / strict-tolerant boundary / `_execute_language` / multimodal path /
per-provider `MODEL_SPECS` + `_normalized_model_spec` + wire schemas + OpenAI Responses standalone
(unanimous "don't touch" — the coupling wall). The multimodal-delete trap (deleting `image_request`/
`data_uri`/`ImageContent` alongside the dead `content_types` field) is called out by r1-09 I, r2-b1,
r2-b2 T1 — all agree it is wired end-to-end and must be kept; that agreement is important precisely
because Fork 1's "delete `content_types`" invites the mistake.

---

## Method / caveats

Verified live at HEAD `84f703b` via grep/sed/reads of: `tests/test_public_cutover.py`
(inventory + AST guards), `tests/providers/test_public_facades.py` (client-runtime boundary),
`symai/providers/_client/*`, the three engines, `symai/decoding.py`, `symai/function.py`,
`symai/runtime/models.py`, and `audit/SYMBOL_REDESIGN.md` §§6-7,11. The two decisive, non-obvious facts
that flip published R2 recommendations: (a) `_client/` has **zero** runtime imports and the mapper
*must* import `runtime.errors` → Fork 2 resolves against r2-b2; (b) the inventory test pins
`providers/_client` to exactly four filenames → Fork 3 confirms r2-d1/d2's new `_client/*.py` files
break it. Tree is a moving target; treat line numbers as approximate, anchors (symbol + snippet) as
load-bearing.
