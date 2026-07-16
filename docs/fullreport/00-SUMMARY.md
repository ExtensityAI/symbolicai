# engine-redesign — Full Audit Synthesis

Read-only audit of the `refactor/engine-redesign` worktree through simplicity / elegance /
boundaries / duplication + ~20 further lenses. **28 agent-reports across 3 rounds** (Round 1:
14 single-lens ×, incl. simplicity ×2 and boundaries ×2; Round 2: 8 cross-cutting, each goal ×2;
Round 3: 6 meta/adversarial ×2 + 3 main-loop consolidations). Every load-bearing claim was
re-verified against live code; the tree moved during the audit (see below). Source reports:
`r1-*.md`, `r2-*.md`, `r3-*.md`; shared map + state in `_CONTEXT.md`.

Snapshot at synthesis: HEAD `84f703b`. **No code was modified — only these reports.**

---

## 0. What happened during the audit (important)

The legacy cutover **landed mid-audit**. When Round 1 started (`09bab6a`) the tree still carried
ambient-runtime discovery, `Function.static_context/dynamic_context`, a 1042-LOC `prompts.py`, an
empty `backend/`, and a ~90-name root facade. By `84f703b` a sibling agent had deleted all of it:
empty root `__init__.py`, `prompts.py`/`backend/` gone, `jinja2`/`python-box`/`tomllib` dropped,
620 tests green, the two cutover suites 6-failing → 66/66. **So roughly half of Round 1's findings
were resolved before this synthesis** — correctly recorded as "already addressed," not open.

That reframes the whole audit: **the redesign's core is sound and now realized; what remains is a
well-scoped quality tail plus a set of release blockers.**

---

## 1. Bottom line — answering the question directly

| Your lens | Verdict |
|---|---|
| **Simple enough?** | Core (Symbol/Function/decoding/runtime) yes. The two *ends* are not: the provider tier is **under-factored** (3× hand-synced scaffolding) and the contract tier is **over-modeled** (~4 surfaces the system cannot exercise). |
| **Simpler while keeping features?** | Yes. **~500–650 LOC and ~6–8 public types** are removable at `keeps-all` or `drops-minimal`, with **no real feature lost** — provided 4 named traps are respected. |
| **More elegant?** | Yes, via three moves: a data-driven capability **gate** (kill spec-vs-code drift), a shared provider **`_engine`/`_client` base**, and decoder consolidation + the `_normalize_text` fix. |
| **Better / cleaner boundaries?** | The layering shape/direction/count are **already right** (acyclic DAG, `Symbol` contained, client↔engine seam holds). Improvements are: **two** shared provider tiers (not one), and three name-collision renames. |
| **Deduplicate?** | ~350–450 LOC of low/no-risk provider-infra dedup + ops helpers, +~120 medium (chat adapter). Provider layer 3472 → ~2800–2950 LOC. |
| **Feature-complete?** | **No — the redesign deleted Contracts** (`@contract` / Design-by-Contract self-healing typed I/O), the user's most-used capability. The audit had logged this as intentional (`FP-08`); that non-goal is **wrong**. See the pillar callout below and [`r7-contracts`](./r7-contracts.md). |
| **Publishable today?** | **No** — missing Contracts (above), plus reasons *beyond* simplicity: a wrong Python floor, broken docs/migration, dirty+unwired static gates, and corrupted migrated few-shot data. |

> ## ⛏ Missing pillar — Contracts (highest-priority scope correction)
> The redesign removed the `@contract` system (real impl on `dev`: `symai/strategy.py`, ~1217 LOC):
> typed `LLMDataModel` input/output, `pre`/`act`/`post` conditions, **semantic validation**, and a
> **self-healing remediation loop** (validation errors → corrective re-prompts, retry+backoff). The
> current `decode_output(default=…)` is a parse-once shadow of it. **Contracts are a required pillar and
> must be re-built on the new explicit-runtime architecture** (engine passed as an `r5` handle; remediation
> cost captured by the `r6` observer seam; structured-output request support kept). This outranks every
> simplification here and means `FINDINGS.md` FP-08 + the design non-goals must be reversed. Effort **L**
> (a subsystem port). Full design + open questions in [`r7-contracts`](./r7-contracts.md).

**Confidence is high.** Two independent adversarial verifiers confirmed **10 of 11** load-bearing
claims (1 phrasing-corrected, 0 refuted); two independent contradiction auditors reached identical
verdicts on every fork; two independent completeness critics converged on the same new gaps.

---

## 2. Release blockers (correctness/publishability — fix regardless of simplicity)

These surfaced *alongside* the quality work and outrank it.

| ID | Blocker | Evidence | Fix |
|---|---|---|---|
| **P1** | **`requires-python = ">=3.11"` is false.** Code uses PEP 695 (`type X =`, `def f[T]`) at 28 sites → 3.12+. `import symai` `SyntaxError`s on 3.11; ruff `target-version="py311"` can't even parse the tree (28 "phantom" errors). | verified: `pyproject.toml:18`, `ruff.toml:35`, 28 grep hits; `r3-m1a`,`r3-m1b` | Floor → `>=3.12`; ruff `target-version="py312"`. |
| **P2** | **Docs/migration are fully broken.** README + all `docs/source/*.md` import a dead API (`Provider`, `create_runtime`, `TransportConfig`); no migration guide (promised by design §3.1/§10); version still `1.18.0`; `CMDS.md` points at deleted paths; the empty root now makes `from symai import Symbol` fail. | `r1-12`, `r3-m1b` | Rewrite docs to the final `RuntimeConfig`/`EngineConfig`/`Function`/`ops.*` surface; add migration guide; bump to `2.0.0`. |
| **P3** | **Static gates dirty + unwired.** `pyright` = 7 prod errors with **no `[tool.pyright]` config**; `ruff` ≈ 60 real errors incl. **PEP 695 / legacy `TypeVar`-`Generic` mixing** (UP046/UP047) in 8 files; `tests/typecheck/function_decoding.py` runs under no runner; no CI. The 7 pyright errors break down (per `r4-symbol-variance`): 2 × `Symbol[T]` invariance in `ops/embed.py:206/221`; 2 × `Runtime.execute` union re-widening at `runtime.py:210`; 2 × `Mapping`-key invariance at `runtime.py:60/61`; 1 × intentional `symbol.py:15 __hash__=None`. | verified pyright/ruff runs; `r1-10`,`r3-m1a/b`,`r4-symbol-variance` | Add configs; exempt PLC0415 for lazy-import modules; **migrate `Symbol` to PEP 695 `class Symbol[T]`** (pyright infers covariance → clears both embed errors, `keeps-all`, effort S — see Round-4 below); type the `Mapping` key `str`; narrow/cast the `execute` dispatch; scoped ignore for `__hash__`; unify generic syntax; gate pyright+ruff+pytest in CI. |
| **P4** | **Prompt query-builder regressions (the data is fine — the query strings aren't).** The Round-4 fidelity audit recovered the original few-shot lists from `git show a220d6f:symai/prompts.py` and diffed element-wise: the inlined example **data is byte-faithful (13/13 identical)** — the migration did NOT corrupt it. But two **migration-introduced regressions** live in the rewritten query builders: `ops/rank.py` dropped the `order:` field that leads all examples and hardcodes descending, so **`asc` ranking is broken** (6/13 examples are ascending); `ops/compare.py` `is_instance_of` emits a token (`is instance of`) matching **neither** example form (`isinstanceof`/`instanceof`). Plus pre-existing defects copied verbatim (the `[33, 'a', , 'help', …]` stray comma, merged Format/Contains examples carrying a stale `Symbol(...)` repr, `logic`'s colon-wrapped operands). 11 defects total: 6 degrade-output, 5 cosmetic, 0 hard-break. | `r4-prompt-fidelity` (git-diff verified) | Fix the two query-builder regressions first; then clean the carried-over data defects. |

---

### Round-4 mini-lenses (follow-ups you approved)

- **`r4-symbol-variance`** — `Symbol` is `Generic[T]` (invariant), which is the *only* cause of the 2
  `ops/embed.py` pyright errors. `T` appears solely in output positions (read-only `value` getter;
  constructors are exempt; operators take `object`), so **covariance is provably sound**. Fix:
  migrate to PEP 695 `class Symbol[T]` — pyright infers covariance, both embed errors clear, the
  module-level `TypeVar` (a `python.md` violation) disappears, and the dormant ruff UP046 is
  pre-empted. `keeps-all`, effort **S**. Feeds P3 and Group A.
- **`r4-prompt-fidelity`** — see **P4**. Headline: example *data* migrated faithfully; the real
  regressions are in two rewritten query-builders (`rank` order, `is_instance_of` token).

## 3. The quality tail — ranked, feature-preserving plan

Grouped by risk/decision. Feature impact tagged; all verified against `84f703b`.

### Group A — safe now (keeps-all, low risk), route into EXISTING files (~300 LOC)
| # | Change | Notes |
|---|---|---|
| A1 | Fold the 4× error→runtime mapping into one parametrized mapper | **Place in a new `providers/_engine/mapping.py`, NOT `_client/`** — it imports `runtime.errors`, and `_client/` must stay symai-blind (verified 0 client→runtime imports). |
| A2 | Hoist `_symbol_value`/`_require_text` → `ops/primitives.py` | 3×/2× duplication + drifted inline copies in rank/compare. |
| A3 | Fold `PydanticDecoder` into `TypeAdapterDecoder(bare type)`; narrow `ConstructorDecoder` to scalar+bool | `PydanticDecoder ≡ TypeAdapterDecoder` proven; container branch violates design §7 ("containers use TypeAdapter"). |
| A4 | Move `_normalize_text` single-quote stripping into the scalar/bool path only | Today it silently rewrites `TextDecoder` output (`'Twas'`→`Twas`) and mangles text before JSON validation. |
| A5 | Unify `Runtime.__init__` vs `RuntimeConfig` validation | Divergence is real: `Runtime({" chat ": e})` accepted, same via `RuntimeConfig` rejected. |
| A6 | Add `by_alias=True` to DeepSeek request serialization | Latent bug: harmless only until an aliased field is added. |
| A7 | Usage-consistency → degrade to `usage=None` instead of raising `InvalidResponseError` | Don't discard a valid completion over accounting metadata; DeepSeek's exact `cache_hit+cache_miss==prompt_tokens` is especially fragile. |
| A8 | Remove `cast("ImplementationId", ...)` no-op casts; consider typed builtin-implementation constants | `Annotated[str, ...]` is statically just `str`. |
| A9 | Collapse the 4 identical `settings.py`; drop redundant loader model-recheck; `FrozenModel ≡ StrictModel` dedup | Mechanical. |
| A10 | Remove dead OpenAI client endpoints (`retrieve/delete/cancel/list_input_items` + GET/DELETE path + `quote` import) | 0 prod callers (`r3-m1b`); confirm before deleting. |

### Group B — contract slimming (needs a one-line decision each; mostly settled)
| # | Change | Feature impact | Recommendation |
|---|---|---|---|
| B1 | Delete dead spec fields (`message_roles`/`content_types`/`response_formats`/`context_tokens`) + orphan enums (`MessageRole`/`ContentType`/`ResponseFormatType`) | drops-minimal (introspection nobody reads) | Do it — **and** make the *surviving* fields authoritative via a data-driven **gate** (B4/Group C), which also removes the parallel hardcoded `_unsupported` checks. Delete-only is the fallback if the gate is deferred. **Never ship the status quo (declared-symmetric, enforced-asymmetric).** |
| B2 | `JsonObject`/`JsonArray`/`JsonEntry` AST → `pydantic.JsonValue` | drops-minimal (keeps the structured-output feature) | Do it (~75–150 LOC, ~4 public types). Never constructed in production; round-trips to `pydantic.JsonValue` anyway. |
| B3 | `logprobs`/`top_logprobs`/`logit_bias` — **cut** | drops-minimal (request knob with no response home) | Cut both request field + forwarding; re-add both halves together if a real consumer appears. Owner may override to "close the loop." |

### Group C — structural extraction (keeps-all, larger; do LAST on a settled foundation)
| # | Change | Coupling risk | Notes |
|---|---|---|---|
| C1 | New `providers/_engine/`: `BaseHttpEngine` (construct+cleanup, `close`, `_retry_after`, `_unsupported`), the error mapper (A1), and the capability **gate** | none/low | New top-level dir is **inventory-pin-legal** (`test_public_cutover.py:279` pins `_client`, not `providers/`). |
| C2 | Grow `providers/_client/` base (BaseClient/transport/headers/base `ResponseMetadata`) | low | **Adding files *into* `_client/` breaks the pinned inventory test** → do it as a deliberate, test-updating commit. Cerebras is a **superset** (adds `RateLimitState` + 6 rate-limit headers) → base treats rate-limit as an *optional extension*, not assumed symmetry. |
| C3 | `ChatCompletionsAdapter` shared by **cerebras + deepseek only** | **medium** | The one place to stop. If the generic hooks cost more than the ~120 saved lines, leave them duplicated. |
| — | **Line not to cross** | — | OpenAI **Responses** ≠ chat-completions (subclasses only the base). Per-provider wire schemas, `_normalized_model_spec`, `MODEL_SPECS` catalogs, `ReasoningEffort` enums stay **per-provider** — the gate *reads* each matrix, it does not centralize it. |

### Group D — naming/placement (keeps-all, small)
| # | Change |
|---|---|
| D1 | `symai/operations.py` → `symai/runtime/requests.py` (kills the `operations`/`ops` collision; co-locate builders with the `*Request` models). |
| D2 | Rename generic `runtime/loading.py::load_runtime` → `compose_runtime`/`build_runtime`; public `symai/loading.py` → `symai/registry.py` (kills the double `loading.py`/`load_runtime`). |
| D3 | **Engine-handle selection API** (see `r5-engine-handle-ergonomics`): replace the two-arg `op(runtime, …, engine="name")` with a bound `rt.language_model("name")` / `rt.embedding("name")` handle passed as one argument — typos fail at acquisition (next to config) not at execute; drops the `engine=` kwarg from ~19 ops + `Function`; keeps `runtime.execute(req, engine=name)` as the low-level escape hatch; also closes the per-capability name-ambiguity finding. `keeps-all`, effort S–M. (Missed by the `r1-08` API lens.) |
| D4 | Rename the public `EngineSpec` type → **`EngineConfig`** (consistent with `RuntimeConfig`; `RuntimeConfig{ language_models: {name: EngineConfig} }`). Mechanical `pyseam` rename. |

**Forward-looking infra (additive, not simplification — separate from the plan above):**
[`r6-observability-usage-cost`](./r6-observability-usage-cost.md) proposes a single `Runtime.execute`
**observer seam** (`observers=`, an immutable `ExecutionRecord`) that captures usage/cost/logging
uniformly across `ops.*`, `Function`, and raw `execute` — the home for flexible usage+cost tracking
and the planned stdlib-logging rework (SYMBOLICAI-17). `keeps-all`, effort M. Prices stay app-owned.

---

## 4. Traps — changes that LOOK like simplification but drop a real feature

| Trap | Why it's a trap |
|---|---|
| **T1** Deleting `image_request`/`data_uri`/`ImageContent`/`vision` when removing the dead `content_types` field | The multimodal request path is **wired end-to-end** (OpenAI `InputImage`, Cerebras `ImageContentPart`, `vision` gating) and publicly importable — just unexposed through `ops.*`. Deleting it drops real vision support. |
| **T2** Pruning "single-provider" `TokenUsage`/`RateLimitMetadata` fields | Every field has ≥1 real producer (DeepSeek cache tokens, Cerebras image/prediction tokens + all rate-limit). Not dead. |
| **T3** Collapsing N-output / cutting `decode_output(output_index/default/limit)` / `execute_many` | **Spec-ratified** (SYMBOL_REDESIGN §6.1/§7/§11). Collapsing `outputs` while keeping `output_index` is *incoherent*. Only touch these as a **deliberate spec amendment**, dropping both halves together. |
| **T4** Homing the error→runtime mapper in `_client/errors.py` | It imports `runtime.errors`; `_client/` must stay symai-blind. This is a boundary violation the current test guard doesn't catch. Use `_engine/`. |

---

## 5. What's already right — keep

Acyclic layer DAG + correct direction; `Symbol` contained to `symbol.py`+`ops.*`; the **client↔engine
seam** (clients never import `runtime`); the strict(internal)/tolerant(inbound) model boundary;
single-level discriminated unions; `_execute_language`'s two-stage design (the design's own predicted
composition point); the Runtime selection/lifecycle core (at-most-once teardown, reverse-order
`BaseExceptionGroup` cleanup, single-owner-thread); lazy inert imports; construction-cleanup discipline
(`client.close()` + `add_note`); the **cutover tests as an executable spec**; `Symbol`'s ~46 operator
dunders (spec-mandated §4.3, uniform, I/O-free); `ops/embed.py` numpy math (verified correct);
credential handling (secret-safe by construction).

---

## 6. Open decisions for the owner (not the audit's to make)

1. **logprobs**: cut (recommended) vs close-the-loop (add response fields + per-provider parsing).
2. **Spec matrix**: data-driven gate (recommended target) vs delete-only (fallback if `_engine/` base is deferred).
3. **N-output**: keep as-is (recommended) vs deliberately amend §7/§11 to drop `output_index` + collapse `outputs`.

---

## 7. Suggested sequencing

1. **Release blockers P1–P4** (Python floor, docs/migration/version, static-gate config + the 7 holes, prompt-fidelity) — independent of everything else, gate the release.
2. **Pillars first** — the redesign isn't a SymbolicAI successor without these: the `r5` engine-handle + `r6` observer seam (small), then the **Contracts port (`r7`, effort L)** which builds on both. Nothing in the simplification tail may delete what contracts rely on (structured-output requests, typed models, the validation path).
3. **Group A** safe dedup into existing files (~300 LOC), bottom-up.
4. **Group B** data-contract decisions (settle the normalized model; do B1 with the capability gate).
5. **Group C** structural `_engine`/`_client` extraction (largest, on the now-settled foundation; C2 co-edits the inventory test; C3 stops before Responses).
6. **Group D** naming polish (incl. `EngineSpec`→`EngineConfig`).

Net effect: ~500–650 LOC and ~6–8 public types removed, the provider tier de-duplicated behind a
real base, the contract tier reduced to what the system can exercise, static/packaging health
restored — with every current feature preserved.
