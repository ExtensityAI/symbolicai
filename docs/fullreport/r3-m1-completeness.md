# R3 · Completeness critic — what the 22 reports missed

Authored in the main loop (Round-3 subagents died on the session API limit). Local checks
only, live HEAD `84f703b`.

## Executive summary

The lens coverage was broad; three real gaps survived it. The biggest is that the
**static-quality gates are unwired and not currently clean** — a publishability signal no
lens ran end-to-end. Two smaller gaps: a genuine **pyright type-hole in `Runtime.execute`**,
and thin direct coverage of a few leaf files (now spot-checked here). The functional suite
is green, so none of this is a correctness emergency — it is release-readiness debt.

## Gap 1 (HIGH) — static-quality gates are unwired and dirty

- **pyright: 7 errors, and there is no pyright config at all** (`pyproject.toml` has no
  `[tool.pyright]`; no `pyrightconfig.json`). Running `uv run pyright symai` (default basic mode):
  - `runtime/runtime.py:210` — `execute()` passes the `LanguageModelRequest | EmbeddingRequest`
    union into the branch-selected engine whose `execute` wants the narrowed type →
    `reportArgumentType`. A **genuine type hole** in the core dispatch; the overloads promise
    narrowing the body doesn't statically prove. Needs a `cast` or a per-branch restructure.
  - `symbol.py:15` — `__hash__ = None` trips `reportAssignmentType`. Standard unhashable idiom;
    wants `__hash__: ClassVar[None] = None` or a scoped `# pyright: ignore`.
  - (5 further errors in the same run.)
- **ruff: 78 issues under `uv run ruff check symai`**, breaking down as:
  - **24 × PLC0415** (import-not-at-top-level) — these are the **intentional lazy imports** the
    design relies on for inert `import symai` (loaders in `loading.py`, provider `loading.py`, ops).
    The design *fights the linter*: `ruff.toml` never exempts them. Fix is config, not code
    (per-file-ignore PLC0415 in the lazy-import modules), but today they read as 24 errors.
  - **~19 genuine residue** from the cutover: 7 × F401 (unused imports left after deletions),
    6 × I001 (unsorted imports), 5 × TC001 / 2 × TC003 (type-checking-block candidates),
    4 × RUF022 (unsorted `__all__` in ops), 2 × A001 (builtin shadowing — `ops` `filter`/`map`/`format`).
- **No CI wires any of it.** `.github/` contains only `FUNDING.yml` (per r1-12). Combined with r1-10's
  finding that `tests/typecheck/function_decoding.py` runs under no runner, **all three static gates
  (pyright, ruff, the type-assertion file) exist as tools but are ungated and not green.**
- Caveat honored: the tree is mid-refactor and the sibling agent just landed a large cutover, so some
  residue is transient; and a project-specific pyright invocation may differ from default mode. But
  "make full type check clean" is an existing project commit goal (`095e8ad`), so the intent is real
  and currently unmet. **Recommend:** add `[tool.pyright]`, exempt PLC0415 for lazy-import modules,
  clear F401/I001, fix the two type holes, and gate pyright+ruff+pytest in CI before the major release.

## Gap 2 (MED) — leaf files under-examined by the lenses (spot-checked here)

- `providers/_client/headers.py::authorization_header` — credential validation *is* sound: rejects
  empty, leading/trailing whitespace, and any control char (< 0x20 or 0x7F) before building the Bearer
  header; raises bare `TypeError`/`ValueError` **without** echoing the key (SEC-01 stays fixed). Minor:
  the bare raises lose a message.
- `ops/embed.py` numeric core — `similarity(metric="cosine")` does max-abs scaling *then* L2
  normalization; the double step is **scale-invariant and correct** (not a bug), just verbose. `mmd`
  has an explicit pairwise bound; `distance(minkowski)` validates `p>=1`; `kernel` validates per-kind.
  No numerical defect found.
- `ops/rank.py`, `ops/compare.py`, `ops/reason.py` — small, follow the standard op shape; the only
  note is the `_symbol_value`/`_require_text` duplication already logged.

## Gap 3 (MED) — un-run whole lenses, now assessed

- **Observability/logging:** there is essentially none in the execution path (no logger, no request
  ids surfaced to logs). For a provider-calling library that is a real operability gap (matches the
  standing "logging audit/rework" project intent) — but it is *additive*, out of scope for a
  simplification pass, and correctly absent rather than half-built. Flag, don't fix now.
- **Extensibility (4th provider):** adding one today means editing ≥4 sites (settings, loading
  registry, client package, engine package) with ~350 LOC of hand-copied scaffolding — the same
  duplication the seam-map addresses. The `_engine`/`_client` base is the extensibility fix too.
- **Concurrency beyond the runtime lock:** the single-owner-thread contract is enforced; no shared
  mutable global remains after the ambient-runtime removal. No new issue.

## What was already well-covered (no gap)

Symbol/Function/decoding, the contract model, provider adapters, duplication, boundaries, the
design↔impl drift, packaging/docs, and tests all got deep, corroborated treatment (several ×2). The
adversarial pass (`r3-m3`) confirmed 9/11 load-bearing claims. Coverage breadth is not the weak point.
