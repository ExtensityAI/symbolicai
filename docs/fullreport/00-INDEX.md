# engine-redesign — Full Audit (simplicity / elegance / boundaries / dedup)

Multi-round, multi-lens audit of the `refactor/engine-redesign` worktree. **Read-only**:
the only artifacts produced are report files in this directory. No code is edited.

Goal: is this **simple enough** (can it be simpler *while keeping features*, dropping only
minimal/irrelevant?), **more elegant**, **better**, with **cleaner boundaries** and **less
duplication** — plus many more lenses. Snapshot commit `09bab6a` (moving target).

Shared map for all agents: [`_CONTEXT.md`](./_CONTEXT.md).

## Strategy

Three rounds of subagents (no workflows; plain parallel agents), each agent writing its own
report file, then a human-authored synthesis:

- **Round 1 — single-lens deep dives** (broad coverage, one concern each).
- **Round 2 — cross-cutting synthesis** (combine concerns, system-level altitude), reads R1.
- **Round 3 — meta / adversarial** (what did we miss? where do reports conflict? are the top
  claims actually true?), reads R1+R2.
- **Synthesis** — `00-SUMMARY.md`, authored after the rounds.

## Report roster

Goals marked **×2** run two independent agents blind to each other; a synthesis step
merges them. Rounds may be extended dynamically if findings reveal new lenses.

### Round 1 — lenses (14 agents)
| File | Lens |
|---|---|
| `r1-01a-simplicity.md`, `r1-01b-simplicity.md` | **×2** Accidental complexity / over-engineering / YAGNI (simpler while keeping features) |
| `r1-03a-boundaries.md`, `r1-03b-boundaries.md` | **×2** Layering & module boundaries; import direction; upward deps; vestiges |
| `r1-02-duplication.md` | Duplication & DRY (cross-provider engines/clients, ops helpers, validation) |
| `r1-04-symbol-function-decoding.md` | Value layer elegance: Symbol DSL, Function, decoding two-stage |
| `r1-05-runtime-ownership.md` | Runtime lifecycle, selection, thread-ownership, ambient discovery |
| `r1-06-contracts-models.md` | Normalized contract/type model (`runtime/models.py`): dead fields, JSON AST, N-output |
| `r1-07-provider-adapters.md` | Engine adapter design: validate/parse symmetry, capability-vs-enforcement, error mapping |
| `r1-08-api-surface-naming.md` | Public surface, re-exports vs "no facade", naming, ergonomics, discoverability |
| `r1-09-dead-code-deps.md` | Dead code & legacy residue: prompts.py, unused deps/exports, backend/ |
| `r1-10-tests.md` | Test structure, coverage vs acceptance criteria, fixture realism, typecheck tests |
| `r1-11-design-impl-drift.md` | SYMBOL_REDESIGN/FIXPLAN claims vs actual code; prior-finding fixed/open status |
| `r1-12-packaging-docs.md` | Packaging, version/migration, README/docs, deps/extras coherence |

### Round 2 — cross-cutting (8 agents, each goal ×2)
| File | Focus |
|---|---|
| `r2-a1-architecture-synthesis.md`, `r2-a2-architecture-synthesis.md` | Is the whole decomposition right? Do abstractions earn their keep system-wide? |
| `r2-b1-simpler-keeping-features.md`, `r2-b2-simpler-keeping-features.md` | Ranked, feature-preserving simplification plan (pressure-tested) |
| `r2-c1-consistency-symmetry.md`, `r2-c2-consistency-symmetry.md` | Cross-provider / cross-ops / error-handling / naming symmetry |
| `r2-d1-boundary-coupling.md`, `r2-d2-boundary-coupling.md` | Ideal seam map: where consolidation risks coupling provider schemas |

### Round 3 — meta / adversarial (6 agents, each goal ×2)
| File | Focus |
|---|---|
| `r3-m1a-completeness.md`, `r3-m1b-completeness.md` | Coverage critic: unapplied lenses, unread code, unverified claims |
| `r3-m2a-contradictions.md`, `r3-m2b-contradictions.md` | Where reports disagree / proposals conflict; adjudication |
| `r3-m3a-adversarial-verify.md`, `r3-m3b-adversarial-verify.md` | Try to refute the top-impact claims against the real code |

### Synthesis
| File | |
|---|---|
| `00-SUMMARY.md` | Final ranked synthesis: keep / simplify / fix, with feature-impact and effort |

## Status

- **Round 1 — COMPLETE** (14/14 reports written). During Round 1 the live sibling agent
  landed the legacy cutover (`84f703b`): empty root, `prompts.py`/`backend/` deleted, ambient
  runtime + `static/dynamic_context` removed, 620 tests green. Surviving opportunities are
  summarized in `_CONTEXT.md`'s STATE UPDATE.
- **Round 2 — COMPLETE** (8/8). Convergent: provider `_engine`/`_client` base is a *missing class*; `runtime/models.py` carries ~4 unexercisable "aspirational" surfaces; net ~500–650 LOC removable at keeps-all/drops-minimal; traps identified (multimodal path, `TokenUsage` fields); decision forks surfaced (spec-matrix gate-vs-delete, error-mapper placement, logprobs, new-dir-vs-pinned-inventory).
- **Round 3 — COMPLETE.** First dispatch died on a session API limit; re-run after it cleared.
  Files: `r3-m1a/b-completeness`, `r3-m2a/b-contradictions`, `r3-m3a/b-adversarial-verify` (the ×2
  independent agents) plus main-loop consolidations `r3-m1-completeness`, `r3-m2-contradictions`,
  `r3-m3-adversarial-verify`. Result: 10/11 load-bearing claims confirmed (1 phrasing-corrected, 0
  refuted); both contradiction auditors agreed on every fork; completeness critics added a new
  category (toolchain/packaging + prompt-fidelity blockers). The ×2 re-run corrected the provisional
  pass on N-output (spec-ratified — keep) and surfaced the Python-floor lie.
- **Round 4 — COMPLETE** (2 targeted follow-ups): `r4-symbol-variance` (fix: PEP 695 `class Symbol[T]`
  → covariance, clears the embed pyright errors), `r4-prompt-fidelity` (data migrated faithfully;
  2 query-builder regressions in `rank`/`is_instance_of`).
- **Round 5 — COMPLETE** (1 follow-up from a usage review): `r5-engine-handle-ergonomics` — replace
  the two-arg `(runtime, …, engine="name")` selection with a bound `rt.language_model("name")` handle;
  `keeps-all`, also closes the per-capability name-ambiguity finding. Missed by the `r1-08` API lens.
- **Round 6 — COMPLETE** (forward-looking infra): `r6-observability-usage-cost` — a `Runtime.execute`
  observer seam for usage/cost/logging (home for SYMBOLICAI-17). Additive, `keeps-all`. Also recorded:
  `EngineSpec` → **`EngineConfig`** rename (Group D).
- **Round 7 — COMPLETE (top-priority scope correction):** [`r7-contracts`](./r7-contracts.md) — the
  redesign **deleted the `@contract` Design-by-Contract system** (typed I/O + `pre`/`act`/`post` +
  self-healing remediation), the user's most-used capability. Read from `dev` (`symai/strategy.py` ~1217
  LOC). It is a **required pillar** and must be re-built on the explicit runtime; reverses audit FP-08.
- **Round 8 — COMPLETE (implementation plan):** [`r8-contracts-plan`](./r8-contracts-plan.md) — port
  Contracts as a primary `Contract[In, Out]` object with `@contract` as a backcompat shim + `LLMDataModel`
  kept; lives in a new `symai/contract/` package (altitude of `ops`, above function/runtime). Phased plan,
  module map, load-bearing signatures, tests, and 4 open decisions.
- **Synthesis — COMPLETE → [`00-SUMMARY.md`](./00-SUMMARY.md)** (folds in Rounds 4–7).

## Read order for a human
0. `09-INTEGRATION.md` — **how to land it all** (re-baseline snapshot + phased, gated execution playbook).
1. `00-SUMMARY.md` — the answer + ranked plan.
2. `r3-m2*` (adjudicated forks) and `r3-m3*` (which claims are proven).
3. Any `r1-*`/`r2-*` lens for the detail behind a specific finding.
