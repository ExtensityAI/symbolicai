# R7 · Contracts are a required pillar (correcting FP-08)

**Status: MUST-KEEP feature.** The user states contracts are the most-used capability of SymbolicAI
and are explicitly required. The engine-redesign **deleted** them; the audit recorded this as an
*intentional* reduction (`FP-08 — structured-output validation with self-healing retry removed`) and
the design's non-goals list "no self-healing." **That decision is wrong for this product.** This note
corrects the record and specifies how contracts must return, faithful to the real implementation.

Grounding: read from `dev` — `symai/strategy.py` (1217 LOC), `docs/source/FEATURES/contracts.md`
(699 lines), `tests/contract/*`. This supersedes the `r1-04`/earlier "collapse the decoders" framing:
the decoder layer was a parse-once shadow of contracts, not a replacement.

## 1. What a contract is (from the real code)

Design-by-Contract applied to probabilistic LLM calls. `@contract` is a **class decorator** that wraps
an `Expression`'s `forward` with a validate → remediate → execute pipeline.

- **Decorator knobs** (`strategy.py:505`): `pre_remedy=False`, `post_remedy=True`,
  `accumulate_errors=False`, `verbose=False`, `remedy_retry_params={tries, delay, max_delay, jitter,
  backoff, graceful, dynamic_engine}`.
- **Typed I/O**: input is an `LLMDataModel` (Pydantic `BaseModel` subclass; `Field(description=...)` text
  is fed into prompts); output is the `forward` **return annotation**, also an `LLMDataModel`. Native
  Python types (`str`, `list[int]`, …) auto-wrap into a single-`value` model and unwrap on return.
- **User-defined terms**: a `prompt` property (fixed task description), `pre(input)` (pre-conditions),
  optional `act(...)` (intermediate transform), `post(output)` (semantic post-conditions), and
  `forward(...)` (the main logic / fallback).
- **Pipeline** (`_run_contract_pipeline` → `_validate_input` → `_act` → `_validate_output`):
  1. validate input against its type + `pre`; if `pre_remedy`, LLM-correct failures;
  2. run `act`;
  3. validate output: **Pydantic type validation** + **semantic `post` checks (an LLM validator,
     `_check_semantic_conditions`)**; if `post_remedy`, run the **remediation loop** — turn the
     validation errors into a corrective prompt and re-ask, `tries` times with exponential backoff
     (`_run_validation_attempts` / `retry` params), optionally accumulating prior errors;
  4. **still call the user's `forward`** with the validated input (never fully bypassed — the fallback
     guarantee), then type-check the final object.
- **Guarantee**: a returned instance of the output model that passed type + semantic validation (or, in
  `graceful` mode, best-effort without raising). Validation failures are *learning signals*, not just
  errors.
- **State/telemetry** on the instance: `contract_successful`, `contract_result`, `contract_exception`,
  `contract_perf_stats()` (per-stage timing), plus `TypeValidationFunction`/`ValidationFunction` for
  building remedy prompts and simplifying `ValidationError`s.

## 2. Why the redesign's removal is unacceptable

- It is the reliability backbone: it's how probabilistic output becomes *dependable* typed output with
  semantic guarantees. Removing it (FP-08) guts the library's main value for its heaviest users.
- The redesign's `decoding.py` (`decode_output(..., default=...)`) offers **none** of it — no `pre`/`post`,
  no semantic validation, no remediation, no retry. It parses once and swallows failure into a default.
- Therefore **contracts are a first-class pillar of the target, not an optional add-on**, and the
  redesign is *not* feature-complete until they return.

## 3. Porting contracts onto the explicit-runtime architecture

The old contract leaned on things the redesign removed (`Expression`, ambient engine discovery, the
mutable prompt-context stack). The port must honor the redesign's principles — **explicit runtime/handle,
immutable values, no ambient globals** — while preserving contract behavior 1:1.

Target shape (new style — a contracted unit is a plain class, not an `Expression`; the engine is passed
explicitly as an `r5` handle):

```python
from symai.contract import contract, LLMDataModel        # ported: symai/contract.py + models
from pydantic import Field

class Review(LLMDataModel):
    text: str = Field(description="Raw customer review text.")

class Verdict(LLMDataModel):
    sentiment: str = Field(description="one of: positive, negative, neutral")
    confidence: float = Field(ge=0, le=1, description="model confidence in [0,1]")

@contract(post_remedy=True, remedy_retry_params={"tries": 5, "backoff": 2, "graceful": False})
class Classify:
    @property
    def prompt(self) -> str:
        return "Classify the sentiment of the review."
    def pre(self, input: Review) -> bool: ...             # pre-conditions
    def post(self, output: Verdict) -> bool: ...          # semantic post-conditions
    def forward(self, input: Review) -> Verdict: ...      # fallback / main logic

with load_runtime(config) as rt:
    classify = Classify(rt.language_model("smart"),       # explicit engine handle (no ambient)
                        remedy=rt.language_model("smart"))# dynamic_engine -> an explicit remedy handle
    verdict: Verdict = classify(Review(text="..."))       # validated, self-healed, typed
    stats = classify.contract_perf_stats()
```

Design decisions this forces (each preserves a contract behavior on the new base):

| Old mechanism | New home | Note |
|---|---|---|
| ambient engine for generation + remedies | **explicit handle(s) at construction** (`Classify(handle, remedy=…)`) | `dynamic_engine` becomes a second handle — cleaner than today. |
| `TypeValidationFunction` (type validation) | Pydantic validate over `Function(...).text`; use `JsonSchemaResponseFormat` (already in the request contract) so the provider is *asked* for the schema | the redesign already supports structured-output requests — reuse it. |
| semantic `post` checks | an internal validator `Function` run on the same/`remedy` handle | keeps `_check_semantic_conditions` behavior. |
| remediation loop (retry+backoff, accumulate_errors) | a bounded loop around `Function` re-calls, errors → corrective prompt | port `_run_validation_attempts` verbatim in spirit. |
| `LLMDataModel` + `Field(description=…)` → prompt | keep `LLMDataModel` (Pydantic base); descriptions drive generation/remedy prompts | the one legacy type worth carrying forward as-is. |
| `contract_perf_stats()` timing | fold into the **`r6` observer seam** — each attempt is one `execute`, so its usage/cost/latency is captured for free | remediation cost becomes *visible* (a 3-try heal = 3 billed calls). |

## 4. Interactions with the rest of the plan

- **Subsumes the decoder question.** Simple throwaway calls: `response.text` / `TypeAdapter(T).validate_json`.
  The *important* typed-output path is a contract. So `r1-04`'s decoder-tuning and my "collapse decoders"
  note apply only to the throwaway path; contracts own reliable typed output. Do **not** delete
  structured-output request support (`JsonSchemaResponseFormat`, the `JsonObject`→`pydantic.JsonValue`
  change is fine but the *feature* stays) — contracts depend on it.
- **Needs the `r6` observer seam** to account remediation-loop cost.
- **Uses the `r5` handle** as the explicit engine (and a second handle for `dynamic_engine` remedies).
- **Reverses an audit "non-goal":** update `FINDINGS.md` FP-08 and `SYMBOL_REDESIGN.md` non-goals — "no
  self-healing / no structured-output validation with retry" must become **required scope**.

## 5. Effort & sequencing

This is a **subsystem port (effort L)**, not a cleanup — `strategy.py` is ~1200 LOC plus `LLMDataModel`
and the validation-function machinery, and it must be re-expressed without `Expression`/ambient engines.
It should be **added to the roadmap as a pillar**, sequenced after the runtime/handle and structured-output
request paths are stable (it builds on both), and before any "feature parity vs main" or release claim —
the redesign cannot be called a SymbolicAI successor without it.

Recommended order relative to the existing plan: Release blockers (P1–P4) → runtime/handle (`r5`) +
observer seam (`r6`) → **Contracts port (this)** → the simplification tail (Groups A–D). Simplifications
must not delete anything contracts rely on (structured-output requests, typed models, the validation path).

## 6. Open questions to confirm with the owner

1. Keep the `@contract` **class-decorator** ergonomics (familiar to current users), or move to a
   `Contract[In, Out]` object? (Recommend: keep the decorator for muscle-memory; it can wrap a plain class.)
2. `pre`/`post`: pure predicates, or may they **normalize/mutate** input? (The real `act` mutates input;
   confirm `pre`/`post` stay validation-only.)
3. Carry `LLMDataModel` forward verbatim, or re-base it on the redesign's `FrozenModel` conventions?
4. Is `graceful` mode + the always-run `forward` fallback still desired, or should a hard-fail contract be
   the default in the new stricter world?
