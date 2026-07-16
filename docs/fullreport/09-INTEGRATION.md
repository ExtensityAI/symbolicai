# 09 · Integration playbook

How to land the audit + design work (reports `r1`–`r8`, summarized in `00-SUMMARY.md`) **without going
stale and without shooting ourselves in the foot.** Two mechanisms carry the whole plan:

1. **Design docs are the source of truth; reports are rationale; re-baseline before every slice.**
   Line-anchored findings rot — the *decisions* don't.
2. **Dependency-ordered phases, each gated green, with the traps encoded as executable tests.** A wrong
   change fails CI instead of shipping.

---

## Re-baseline snapshot (as of working tree `203902f`)

Re-run this delta at the start of each work session and before every phase:

- Invariant baseline **green**: full suite **630 passed**.
- Static gates still block the next phase: **pyright 5 errors**, **ruff 7 errors**.
- Python floor and Ruff target are already `3.12` in the working tree.
- The worktree contains pre-existing, uncommitted Phase-1 changes. Preserve and reconcile them;
  never reset or overwrite them while applying the roadmap.
- Phase-0 integration invariants remain present and passing.

Re-baseline command (local, no API): `uv run ruff check symai`, `uv run pyright symai`, then
`uv run pytest -q`. Diff the result against the finding list and mark each slice FIXED or
STILL-OPEN before editing.

---

## Phase 0 — Stabilize the ground (no behavior change)

| # | Task | Why it's first |
|---|---|---|
| **T0.1** | Re-baseline pass (above); record the FIXED/OPEN delta. | Never work from a stale anchor. |
| **T0.2** | **Amend the design docs to match our decisions**: reverse `FINDINGS.md` **FP-08** (contracts are *required*), update `SYMBOL_REDESIGN.md` non-goals + record `EngineConfig`, the engine-handle API (`r5`), the observer seam (`r6`), the decoder→`response.text`/`Callable` collapse, **N-output/`output_index` KEPT**, and defaults dropped. | **Spec before code.** Skip this and the next agent re-derives "contracts were intentionally removed" and undoes the work. |
| **T0.3** | **Encode the invariants as tests** (the anti-foot-gun): multimodal path is wired end-to-end; no `providers/*/client/*` imports `runtime`; every `TokenUsage`/`RateLimitMetadata` field has a producer; the `_client` inventory pin stays (edited only deliberately); N-output/`output_index` present. | Turns "don't do this" from a memo into a failing test. |

Phase 0 changes only docs + tests — zero runtime behavior, safe to land immediately.

> **Phase 0 complete.**
> - **T0.1** — the source is restructured into `symai/providers/{openai,cerebras,deepseek}/{client,engines}`
>   with a shared `providers/_client/` base. (`audit/FINDINGS.md`'s `symai/clients/` + `symai/backend/`
>   paths are the older `refactor/cleanup` layout; the `docs/fullreport/` reports track `providers/`.)
> - **T0.2** — `audit/FINDINGS.md`, `SYMBOL_REDESIGN.md`, and `FIXPLAN.md` state the decisions directly:
>   contracts are a required pillar (FP-08), operations and `Function` take a bound engine handle, the
>   decoder path is a `Callable[[str], T]`, config defaults are dropped (sole-engine auto-resolution
>   kept), and N-output/`output_index` is retained. (The `EngineConfig` rename and observer seam are
>   recorded in the reports as Phase 2–3 code changes.)
> - **T0.3** — `tests/test_integration_invariants.py` (9 tests) pins the traps: multimodal wired, the
>   client layer never imports `runtime`, `TokenUsage`/`RateLimitMetadata` fields pinned + producer-backed,
>   N-output/`output_index` retained.
> - **Green:** full suite **630 passed** (621 + 9); ruff + pyright clean on the new file.

## Phase 1 — Safety gates before behavior changes

Do these first **because CI becomes the guardrail for everything after.**

| # | Task | Report |
|---|---|---|
| **T1.1** | Python floor → `>=3.12`; ruff `target-version = "py312"`. Already present in the working tree; verify and retain. | P1 |
| **T1.2** | Add `[tool.pyright]` + ruff config (exempt `PLC0415` in lazy-import modules); **wire CI: pyright + ruff + pytest**. | P3, `r3-m1a/b` |
| **T1.3** | Clear the remaining static holes: type `Runtime` mappings and execute dispatch, preserve intentionally unhashable `Symbol`, and remove unused provider-package imports without adding facade exports. | `r4-symbol-variance`, current re-baseline |
| **T1.4** | Prompt-fidelity: verify and fix `rank` ordering, the `is_instance_of` token, and carried-over malformed examples. | P4, `r4-prompt-fidelity` |

**Gate:** CI green (pyright + ruff + pytest) before Phase 2. From here, every change is protected.

## Phase 2 — Pillars, BEFORE any simplification

This ordering is the single most important foot-gun guard: **contracts depend on structured-output
requests, typed models, and the schema path — build them before the tail can delete anything they need.**

| # | Task | Report |
|---|---|---|
| **T2.1** | Final runtime/config cutover as one atomic API change: rename `EngineSpec` → `EngineConfig`; remove configured `default_*`; add `rt.language_model("name")` / `rt.embedding("name")` handles; ops + `Function` take a handle; keep sole-engine unnamed resolution and the low-level `runtime.execute(req, engine=…)` escape hatch. Migrate every caller with no compatibility alias. | `r5`, decoders-&-defaults decision, Group D4 |
| **T2.2** | Observer **seam**: `observers=` on the Runtime, immutable `ExecutionRecord`, built-in `log_executions` + module logger, optional `Price`/`cost`/`UsageMeter` (prices app-owned). | `r6` |
| **T2.3** | **Contracts port** (effort L): new `symai/contract/` (`models`, `contract`, `validation`, `remedy`, `decorator`); native `Contract[In, Out]` (raises via `__call__`, reports through a discriminated success/failure result from `.run()`) + `@contract` backcompat shim; `LLMDataModel` kept; boundary + ported parity tests. | `r7`, `r8` |

**Gate:** the redesign is not a SymbolicAI successor until T2.3 lands. Contract parity tests + CI green.

## Phase 3 — Simplification tail, on the settled foundation

| # | Task | Notes | Report |
|---|---|---|---|
| **T3.A** | Safe dedup **into existing files** (~300 LOC): ops helpers → `primitives`; `settings.py` ×4; `PydanticDecoder` fold into `TypeAdapterDecoder`; narrow `ConstructorDecoder`; `_normalize_text` quote-strip → scalar/bool path only; unify `Runtime`/`RuntimeConfig` validation; DeepSeek `by_alias`; usage-consistency → degrade to `usage=None`; remove dead OpenAI client endpoints; drop `cast("ImplementationId", …)`. | keeps-all; no new files → doesn't touch the inventory pin. | Groups A, `r2-b1/b2` |
| **T3.B** | Data-model decisions: delete dead spec fields + orphan enums **behind a data-driven capability gate**; `JsonObject`→`pydantic.JsonValue`; **cut** logprobs. **N-output KEPT.** | Settle the contract model before the structural pass. | Group B, `r3-m2a/b` |
| **T3.C** | Structural extraction: new `providers/_engine/` (`BaseHttpEngine`, the error→runtime mapper, the capability gate); grow `providers/_client/` base (**edit the inventory pin deliberately**); `ChatCompletionsAdapter` for **cerebras+deepseek only** (stop before OpenAI Responses). | Largest; medium-risk step is the chat adapter. | Group C, `r2-d1/d2` |
| **T3.D** | Remaining naming: `operations.py`→`runtime/requests.py`; generic loader→`compose_runtime`; public `loading.py`→`registry.py`. | `pyseam` mechanical renames; Group D |

**Gate:** full suite + pyright + ruff green after each task.

## Phase 4 — Publish the settled API

| # | Task | Report |
|---|---|---|
| **T4.1** | Rewrite README + `docs/source/*` against the final handles, `EngineConfig`, observers, and native Contracts; add the major-version migration guide. | P2, `r1-12`, `r5`–`r8` |
| **T4.2** | Remove stale dependencies and commands, set version `2.0.0`, build the package, install the built artifact into a clean environment, and run end-to-end fake-engine smoke examples. | P2, packaging audit |

**Gate:** ruff + pyright + full pytest + package build/install smoke all green.

---

## Rules that keep it from going sideways

- **Every phase ends green** (pytest + pyright + ruff) before the next begins — no building on sand.
- **Small, one-finding PRs**, each **rebased on the current redesign tip first**; the cutover suite is the
  invariant baseline (must stay green). No big-bang refactor.
- Check worktree ownership before every slice and preserve unrelated changes.
- Branch from `dev` / the redesign tip; never merge to `main` without the owner's explicit action.
- Reports remain the rationale and this playbook remains the execution order.
- **Don't delete across a phase boundary.** The traps (multimodal, single-provider usage fields,
  N-output/`output_index`, mapper-not-in-`_client`) are Phase-0 tests — respect them in every later PR.

## One-line dependency map

```
P0 (docs+tests) → P1 (static gates + CI) → P2 (EngineConfig/handles → observer → CONTRACTS) → P3 (A dedup → B data-model → C structure → D naming) → P4 (docs + package proof)
                                                                 ▲ pillars before tail: never delete what contracts need
```
