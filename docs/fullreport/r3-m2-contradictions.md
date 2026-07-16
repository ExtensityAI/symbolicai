# R3 · Contradiction auditor — adjudicating the forks

Authored in the main loop (Round-3 subagents died on the session API limit). Each fork is
resolved with a single recommendation + the deciding evidence. **Headline: none of the six
apparent disagreements is a deep conflict.** Each dissolves once you separate two things the
reports conflated. That itself is the finding — the audit converges.

| # | Fork | Positions | Resolution |
|---|---|---|---|
| 1 | `LanguageModelSpec` matrix | **delete dead fields** (r1-06, r1-01a/b) vs **make it a data-driven gate** (r1-07, r2-a1/a2, r2-c1/c2) | **Both — they address different fields.** Delete the 4 fields that are read *nowhere* (`message_roles`, `content_types`, `response_formats`, `context_tokens`) + the 3 orphan enums (`MessageRole`/`ContentType`/`ResponseFormatType`). For the fields that ARE read (`response_tokens`, `reasoning_*`, `sampling_fields`, `vision`, `dimensions`), keep them and route the per-field `_unsupported` checks through one shared gate that reads them — killing the "spec says X, hardcode says X twice" divergence. Not a conflict once dead-vs-live fields are separated. |
| 2 | Error→runtime mapper placement | **fold into `_client/errors.py`** (r1-02, r2-b2) vs **new `providers/_engine/`** (r2-d1, r2-d2) | **`_engine/`.** The mapper translates provider `ClientError` → `symai.runtime.errors`, so it must import `runtime` — which `_client/` must never do (verified: 0 client→runtime imports). Folding it into `_client/` is a boundary violation. The shared error *base classes* stay in `_client/`; the *mapping to runtime errors* lives in `_engine/`. r2-d2 caught the trap r1-02 set. |
| 3 | New `_engine/` dir vs pinned inventory | seam-map wants new files (r2-d1/d2) vs `test_public_cutover.py:279` pins file inventory + r2-b2's caution | **Not a real blocker.** The pin asserts `providers/_client/*.py == {fixed set}`; a *new* `providers/_engine/` package doesn't alter `_client/`'s contents, so it doesn't fail that assertion. Any per-provider dir enumeration that does trip should be updated **in the same commit** as the extraction — the inventory test is a spec to co-evolve, not a wall. Sequencing: land the safe in-file folds (~300 LOC, no inventory change) first; introduce `_engine/` as a deliberate, test-updating step. |
| 4 | logprobs | **cut** (r2-b2) vs **cut or close the loop** (r1-06, r2-a1) | **Cut now.** `logprobs`/`top_logprobs`/`logit_bias` are forwarded and even parsed by clients, but no output field holds them and no ops/decoder consumes them. Closing the loop = new response fields + per-provider parsing for a feature nobody exercises (YAGNI). Cut both request field + forwarding; if a real consumer appears, add both halves together. `drops-minimal`: a knob that produced nothing usable. |
| 5 | N-output collapse vs `output_index` | **collapse** (r1-06, r2-a1/a2) vs **don't cut the design-blessed `output_index`** (r2-b2 trap) | **Compatible — collapse the response, keep the param thin.** No request field can produce >1 output (verified), so the response tuple + per-output `index` + engine dedup/sort is machinery for an unreachable case: collapse `LanguageModelResponse` to a single output. `decode_output(output_index=0)` can stay as a thin forward-compat affordance (or drop — minor). The "trap" was only about deleting the param wholesale, not about collapsing the dead response machinery. |
| 6 | Decoder layer | **delete `PydanticDecoder`** (r1-04) vs **keep the decoder family as public API** (r1-08) | **Both.** `PydanticDecoder` is provably identical to `TypeAdapterDecoder` for a `BaseModel` (r3-m3 confirmed); deleting the redundant member makes the public family *more* coherent, which is exactly r1-08's goal. Also narrow `ConstructorDecoder` to scalars+bool (its container branch has no consumer and clashes on grammar with TypeAdapter). "Keep the family" ≠ "keep every member." |

## Meta-observation

The ×2 redundancy the user asked for paid off precisely here: where a single agent stated a
one-sided recommendation (r1-06 "delete", r1-02 "fold into `_client`"), a sibling or cross-cutting
agent supplied the missing half (r1-07 "gate", r2-d2 "boundary — use `_engine`"). The synthesis
inherits the *reconciled* position, not either partial one.
