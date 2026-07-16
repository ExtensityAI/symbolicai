# r3-m1a — Completeness / Coverage Critic

**Round 3 meta-lens.** Goal: find what the 14 R1 + 8 R2 lens/synthesis reports **missed** —
unexamined code, un-run perspectives, and unverified claims. Read-only; verified against live
code at HEAD `84f703b`. Line numbers are approximate (moving target); findings anchored by
symbol + snippet.

## Executive summary

1. **Whole perspectives were never run.** The 22 reports are all *design* lenses (simplicity,
   duplication, boundaries, contracts, naming, drift, packaging). **No report actually ran
   `pyright`/`ruff` over the tree, exercised numeric correctness, checked credential-safety of
   the current client code, or audited the migrated few-shot data.** I ran them; three surface
   real, previously-unreported defects.
2. **Static analysis does NOT pass.** `uv run pyright` = **68 errors** (7 in `symai/` prod code,
   ~61 in tests); `uv run ruff check` = **109 errors** (78 inside `symai/`). Only r1-10 (F2)
   caught *one* corner (the orphaned `tests/typecheck/` dir). The 7 production pyright errors and
   the systemic `Symbol[T]`-invariance friction were unreported.
3. **`requires-python = ">=3.11"` is factually wrong.** The code uses PEP 695 (`type X = …`,
   `def f[T]`, `class C[T]`) which is **3.12+ syntax** — it raises `SyntaxError` on 3.11. r1-12
   discussed `requires-python` only re: the numpy cap; nobody caught the version-floor lie.
4. **Migrated few-shot example data is corrupted and unaudited.** `ops/rank.py` and
   `ops/compare.py` contain garbled example tuples (a stray `, ,`, two examples merged into one
   string, a leftover old-`Symbol` repr, `is instance of` prompt vs `isinstanceof`/`instanceof`
   examples). No lens checked prompt/example fidelity after the `prompts.py`→inline-tuple move.
5. **Zero observability.** Not one `logging`/`logger`/`warnings.warn` call exists in `symai/`.
   For a networked LLM/HTTP library this is a real gap (rate-limit metadata is parsed then
   discarded). No report raised it. Credential handling itself, by contrast, is genuinely good.

---

## Gap register (ranked)

| # | Gap | Type | Evidence | Conf | Impact | Follow-up lens? |
|---|---|---|---|---|---|---|
| G1 | `pyright` not clean: 7 errors in `symai/`, ~61 in tests | un-run lens | `uv run pyright symai/` → 7 errors | high | high | **yes** — static-health |
| G2 | `ruff` not clean: 78 errors in `symai/` (24 PLC0415, invalid-syntax ×~28, F401, I001, A001…) | un-run lens | `uv run ruff check symai/` → 78 | high | med | **yes** — static-health |
| G3 | `requires-python=">=3.11"` but code is PEP-695 (3.12+) → SyntaxError on 3.11 | unexamined + wrong claim | `pyproject.toml:18`; `type _OwnedEngine` @ runtime.py:26 | high | high | fold into packaging |
| G4 | `Symbol[T]` invariance makes `ops.embed` / numeric ops uncallable pyright-clean | unexamined | embed.py:206/221; test errors @ 593/598/621/623 | high | high | **yes** — type-ergonomics |
| G5 | Corrupted/mismatched migrated few-shot examples | unexamined content | rank.py `_RANK_EXAMPLES`; compare.py `_CONTAINS`/`_IS_INSTANCE_OF` | high | med | **yes** — prompt-fidelity |
| G6 | No logging/observability anywhere in `symai/` | un-run lens | `grep -rE 'logging\|getLogger' symai/` → 0 | high | med | **yes** — observability |
| G7 | No lint/type config at all (`[tool.ruff]`/`[tool.pyright]` absent) → ruff infers 3.11, no CI gate | un-run lens | `pyproject.toml` has only `[tool.uv*]` | high | med | fold into G1/G2 |
| G8 | Embedding execution path least-examined (engine + client) | unexamined code | `openai/client/embeddings.py` = 4 report mentions; `engines/embedding.py` = 6 | med | med | spot-covered here |
| G9 | "byte-identical transport/headers" claim overstated | unverified claim | diffs: openai≡deepseek *modulo docstring/import*; cerebras genuinely differs | high | med | note only |
| G10 | Numeric correctness (cosine/minkowski/MMD) never verified — actually sound | un-run lens (cleared) | embed.py; verified below | high | low | none (positive) |
| G11 | Credential-safety never audited — actually good, one nit | un-run lens (mostly cleared) | `_client/headers.py`, `_client.py` | high | low | note only |

---

## A. Unexamined code (blind spots)

Method: for every `.py` under `symai/`, counted how many of the 22 r1/r2 reports mention its
basename, then read the low-coverage + task-flagged files and verified against live code. Raw
mention-count is noisy (shared basenames like `errors.py`/`settings.py`/`transport.py`), so I
verified the substance, not just the string.

**Genuinely thin coverage (verified by reading):**

- **The whole embedding path** — `providers/openai/client/embeddings.py` (4 mentions, the lowest
  of any non-`__init__` file) and `providers/openai/engines/embedding.py` (6). Reports covered the
  *chat/responses* adapters in depth (r1-07, r2-c*) but treated embeddings as an afterthought.
  Things only reachable here and never called out:
  - `EmbeddingEngine._parse_response` (embedding.py) hard-rejects on
    `if usage.total_tokens != usage.prompt_tokens: raise InvalidResponseError(...)`. This is the
    same brittleness class as the seeded "over-strict usage-consistency" but on the embedding
    side — a benign provider-side accounting change breaks all embeddings. Not in any report.
  - `_validate_request` gates `dimensions` on a hardcoded `_DIMENSIONALITY_MODELS` frozenset —
    the same "hardcoded capability check vs data-driven spec" smell the adapter reports raised for
    chat, but unreported for embeddings.
- **`ops/rank.py`, `ops/reason.py`, `ops/compare.py`** (9–10 mentions each, mostly in dedup
  tables for `_symbol_value`/`_require_text`). The *example payloads* — the bulk of these files —
  were never read for content (see G5 below).
- **`runtime/engines.py`** (5 mentions) — trivially small `Protocol` pair, no issue; noting only
  that it was effectively unexamined.
- **`providers/_client/headers.py`** — the credential validator (`authorization_header`) — 9
  mentions but none about its *security* behavior (see §B-security).

**Adequately covered** (verified the mentions are substantive): `symbol.py`, `function.py`,
`decoding.py`, `runtime/models.py`, `runtime/runtime.py`, `runtime/config.py`, the three
`chat_completions.py`/`responses.py` engines, `_client.py`, provider `errors.py`. The seeds and
r1-02/r1-07/r2-* cover these well.

**The 7 production-code pyright errors nobody reported** (all real, verified):

| Site | Error | Root cause |
|---|---|---|
| `symbol.py:15` `__hash__ = None` | `None` not assignable to `(self)->int` | idiom to make `Symbol` unhashable trips pyright; needs `__hash__: ClassVar[None]` or ignore |
| `ops/embed.py:206,221` `_numeric_array(symbol, field)` | `Symbol[Sequence[float]\|ndarray]` not assignable to `Symbol[object]` | **Symbol invariance** (G4) |
| `runtime/runtime.py:60,61` `self._validate_aliases("…", snapshot)` | `dict[str, LanguageModelEngine]` not assignable to `Mapping[object,object]` | `Mapping` key is invariant; param should be `Mapping[str, object]` |
| `runtime/runtime.py:210` `selected.execute(request)` | union `LanguageModelRequest\|EmbeddingRequest` not narrowed | dispatch branch doesn't narrow the request type for the engine |

---

## B. Un-run lenses (I ran the ones that matter)

The R1/R2 roster (per `00-INDEX.md`) is entirely design-oriented. These perspectives were never
applied; results of my quick passes below.

### B1. Static-analysis health — **FAILS** (G1, G2, G7)
- `uv run pyright` → **68 errors, 0 warnings**. 7 in `symai/` (table above); the rest in
  `tests/test_symbol_runtime_cutover.py` (Symbol-invariance friction on `mmd`/`distance`/
  `similarity`, and `metric="bad"` negative cases with no `# pyright: ignore`) and
  `tests/typecheck/function_decoding.py:35` (r1-10 F2 caught this one).
- `uv run ruff check` → **109 errors** (78 in `symai/`). Breakdown for `symai/`: ~28
  `invalid-syntax` (ruff parsing PEP-695 as 3.11), 24 `PLC0415` (import-outside-top-level — the
  lazy provider imports; likely *intended* but unsuppressed), 7 `F401`, 6 `I001`, 5 `TC001`, 4
  `RUF022`, 2 `TC003`, 2 `A001` (`filter`/`map` params shadow builtins in `ops/text.py:272,294`).
- **Root cause of the noise:** there is **no `[tool.ruff]` or `[tool.pyright]` config** in
  `pyproject.toml` (only `[tool.uv*]`). Ruff therefore infers `target-version` from
  `requires-python = ">=3.11"` and flags every PEP-695 construct as invalid syntax, which also
  *degrades its lint coverage* on `ops/compare.py`, `ops/rank.py`, `ops/reason.py`,
  `ops/primitives.py`, `runtime/runtime.py`. Nobody ran either tool tree-wide; the intent
  (commit `test(runtime): make full type check clean`) is unmet and ungated.

### B2. Numeric correctness of `ops/embed.py` — **SOUND** (G10, positive)
I verified the three flagged concerns; all are correct — worth recording so they aren't
"fixed" into bugs later:
- **Cosine "double normalization" is not a bug.** `left_scaled = lhs/max|lhs|` then
  `left_scaled/‖left_scaled‖` reduces algebraically to `lhs/‖lhs‖` — the max-abs step is a
  pure overflow-avoidance preconditioner that cancels exactly. **One real nit:** the result is
  not clamped to `[-1, 1]`, so floating-point can yield `|cos| > 1` (a downstream `1 - cos`
  could go slightly negative).
- **Minkowski** `sum(|Δ|**p) ** (1/p)` with the `p≥1 & finite` guard is correct.
- **MMD bound** uses `(n_x+n_y)² ≤ 1e6`, which upper-bounds the true pairwise work
  `n_x²+n_y²+n_x·n_y` — a conservative (safe) over-estimate, never an under-bound. `_rbf_matrix`
  correctly clamps negative squared-distances (`np.maximum(…, 0, out=…)`). The estimator is the
  *biased* MMD² (includes the diagonal) — a legitimate choice, worth one doc sentence.
- `_numeric_array` validation (`dtype.kind in "iuf"`, finite check, `astype(float, copy=False)`)
  is careful and allocation-lean. **Keep.**

### B3. Credential-safety / secret-leakage — **GOOD**, one nit (G11)
- `_client/headers.py::authorization_header` validates `SecretStr`, rejects empty / outer-space /
  control chars, returns `Bearer …`. The api key is stored only in `Client._headers`; **every
  provider error message is a static string** — no path interpolates the key or the Authorization
  header into an exception. `TransportError` re-raises httpx errors whose repr carries the URL but
  not headers. No leak found. **Nit:** `authorization_header` raises **bare** `TypeError`/
  `ValueError` (no `msg`) — good for not echoing the secret, but inconsistent with the repo's
  `raise X(msg)` convention and gives zero diagnostic.
- The openai `client/errors.py` stores `response.text` on `APIError`/`AuthError`; for a 401 that
  body is provider-controlled, not the request — low risk, but the only place a response body is
  retained. Not a leak of *our* secret.

### B4. Observability / logging — **ABSENT** (G6)
`grep -rE "import logging|getLogger|logger\.|warnings\.warn" symai/` → **0 hits**. A networked
client library emits nothing: no request/latency/retry/model-selection signal. Notably,
`cerebras` parses a full `RateLimitState` (`x-ratelimit-*` headers) in
`client/headers.py`/`transport.py` and then **discards it** — it is never logged, surfaced on the
normalized response, or used for backoff. (Aligns with the user's planned SYMBOLICAI-17 logging
pass, but the reports never flagged that it is currently *entirely* absent.)

### B5. Concurrency beyond the runtime lock — **low risk, briefly checked**
`runtime.py` uses `threading.Lock` + `get_ident` for thread-ownership (seeded as "over-applied").
Beyond that: `httpx.Client` is thread-safe for requests; the only unguarded shared mutable state
is the `_closed: bool` flag on each `Client`/engine (`close()` does check-then-set without a
lock) — a benign double-close race at worst. Not worth a dedicated lens.

### B6. Extensibility (cost of a 4th provider) — **covered obliquely, never costed**
A new provider requires ~10–12 files (`client/{__init__,_client,transport,headers,errors,chat}`,
`engines/{__init__,chat_completions}`, `loading.py`, `settings.py`, `__init__.py`) plus loader
registration — most near-duplicated. The duplication reports (r1-02, r1-07, r2-a/d) cover the
*mechanism*; none states the *marginal cost* explicitly, which is the number that motivates the
missing `_engine`/`_client` base class they all recommend.

### B7. Docs content-correctness applied to **prompt/example data** — **NOT run** (G5)
The migration from `prompts.py` classes to inline example tuples was verified for *deletion*
(r1-09) but never for *fidelity of the copied content*. Concrete corruptions:
- `ops/rank.py` `_RANK_EXAMPLES`: `list: [33, 'a', , 'help', 1234567890]` — a stray `, ,`
  (empty element) in two examples; several entries have unbalanced quotes from auto-wrapping
  (`'1234567890]`, `'33]'`, `'2, 1]'`).
- `ops/compare.py` `_CONTAINS_EXAMPLES`: two examples are **merged into one string** with no
  separator — `…what it is']",))],)' =>True'Apple Inc.' in 'Microsoft…' =>False` — and it embeds a
  stale `<class 'symai.symbol.Symbol'>(value=…)` repr from the *old* Symbol.
- `ops/compare.py` `_IS_INSTANCE_OF_EXAMPLES`: mixes operators `isinstanceof` (lines ~77–99) and
  `instanceof` (lines ~100–103); meanwhile `is_instance_of()` emits the prompt with the token
  **`is instance of`** — the prompt and its own few-shot examples use three different tokens,
  degrading in-context conditioning.

These are cheap to fix but silently reduce output quality; a targeted fidelity pass is warranted.

---

## C. Unverified claims repeated across reports

- **"transport.py / headers.py are byte-identical" (r1-02 D3, echoed by r2-a1).** Verified by
  diff: openai vs deepseek `transport.py` differ **only in the docstring**; `headers.py` differ
  **only in the import path** — "identical modulo docstring/import," not literally byte-identical
  (r2-a1 hedged correctly; r1-02's "byte-identical" is overstated). More importantly **cerebras is
  genuinely different** — its `transport.py` adds a `RateLimitState` model + `rate_limit` field
  and its `headers.py` parses six `x-ratelimit-*` headers. So a blind extract-to-`_client/` would
  either drop cerebras's rate-limit capability or force it on the others; the consolidation is a
  design decision, not the mechanical move "byte-identical" implies. (G9)
- **"620 passed / 66/66 green" (r1-10, r1-11).** `uv run pytest --collect-only` now reports
  **621 tests collected, 0 collection errors** — consistent (tree moved by one test). Claim
  stands; not a gap.
- **"pyright: 0 errors" for the no-op `cast` removal (r1-08:22,211).** That local claim is about
  one file; it does *not* imply the tree is pyright-clean — and it isn't (68 errors). The reports
  never asserted a tree-wide clean pass, but a reader could infer one; worth disambiguating.
- **"`_client.py` is 83% identical" (r1-02 D2).** Spot-verified: cerebras vs deepseek `_client.py`
  diff is ~58 lines of a ~115-line file changed/context — the *structure* (`_raise_for_status`,
  `_parse_response`, ctor, `close`, `_request`) is indeed shared; the figure is plausible and I
  did not find it overstated.

---

## Recommended follow-up mini-lenses

1. **Static-health gate (G1/G2/G3/G7)** — highest ROI. Add `[tool.ruff]` (`target-version` =
   py312) + `[tool.pyright]`, bump `requires-python` to `>=3.12`, resolve the 7 prod pyright
   errors, and wire both into CI. Small, mechanical, unblocks every other quality signal.
2. **Type-ergonomics / `Symbol[T]` variance (G4)** — decide whether `Symbol` should be covariant
   (`class Symbol[T_co]`) or whether numeric ops should accept `Symbol[object]`/protocols, so the
   library's own tests and users can call `embed.*` pyright-clean. Design decision, medium effort.
3. **Prompt/example fidelity pass (G5)** — re-derive the inline example tuples from the original
   `prompts.py` (git history) and assert prompt-token ↔ example-token consistency. Low effort,
   real output-quality impact.
4. **Observability seam (G6)** — a thin `logging` pass (request lifecycle, rate-limit metadata,
   model selection); the parsed-but-discarded `RateLimitState` is the obvious first consumer.

## Already good (keep)
- `ops/embed.py` numeric implementation (preconditioned cosine, conservative MMD bound, strict
  finite/dtype validation) — carefully written; only add a `[-1,1]` clamp + estimator doc note.
- Credential handling and static error messages across every provider client/engine — secret-safe
  by construction.
- `providers/_client/{models,errors}.py` (`StrictModel`/`TolerantModel`, the client exception
  hierarchy) — small, correct, the right shared base to build the recommended `_client` on.
