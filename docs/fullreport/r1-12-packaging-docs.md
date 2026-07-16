# r1-12 — Packaging, migration & docs coherence

> **Historical snapshot terminology.** This report evaluates documentation against the API that
> existed during Round 1. Its `EngineSpec` and configured `default_*` references are superseded by
> `EngineConfig`, no configured defaults, and sole-engine-only unnamed selection.

Lens: packaging (`pyproject.toml`), migration guide, README, `docs/source/**`, `CMDS.md`,
`.gitbook.yaml`. Read-only audit of worktree `engine-redesign` snapshot (live-verified;
tree is a moving target). All prior-audit IDs re-verified against current code and labeled.

---

## Executive summary

- **The published docs describe a THIRD, non-existent API.** README + all five
  `docs/source/*.md` pages import `Provider`, `ProviderEngineConfig`, `create_runtime`,
  `TransportConfig` and call `runtime.execute(request)` — **none of those names exist**
  (`create_runtime`/`Provider`/`ProviderEngineConfig`/`TransportConfig` all resolve `False`
  on live `import symai`). Every code snippet in the shipping docs raises `ImportError`.
  The prior audit (DOC-03) verified these pages *accurate* at `a220d6f`; the engine-redesign
  cutover replaced that API and left the docs behind. **This is the headline blocker.**
- **The promised major-version migration guide does not exist.** `SYMBOL_REDESIGN.md` §3.1/§10
  and `FIXPLAN.md` L (line 423) require it as a release gate; there is no `MIGRATION*`,
  `CHANGELOG*`, or "migration" doc anywhere in the tree. Release gap.
- **Version frozen at `1.18.0`** for a breaking whole-API rewrite (PKG-01 STILL-OPEN). Nothing
  gates a bump (no CI publish workflow — `.github/` holds only `FUNDING.yml`). Must go to `2.0.0`.
- **`CMDS.md` is entirely dead** (DOC-01/02/03 STILL-OPEN): points at `symai/backend/mixin/…`
  and `symai/backend/engines/neurosymbolic/…` (deleted), `tests/engines` (only a stale `.pyc`),
  `--engine-api` (no such pytest option), and a `symai.config.json` the library never reads.
- **`python-box` is a dead dependency** (zero imports); `numpy`'s `<=2.1.3` cap is now
  unmotivated (torch fully removed). Description/keywords, `source-exclude`, and extras
  structure are **correct and should be kept**.

Overall read: **NOT publishable.** The code cutover ran ahead of docs, version, and migration
artifacts. None of these are deep design problems — they are release-hygiene gaps — but they
are blocking and mutually reinforcing (docs point at a dead API *and* there's no migration map
to the real one).

---

## Findings table

| ID | Finding | Sev | Confidence | Prior-audit label |
|----|---------|-----|-----------|-------------------|
| DP-1 | README + 5 docs pages import a non-existent API (`create_runtime`/`Provider`/`ProviderEngineConfig`/`TransportConfig`) | High | high | regression vs DOC-03 |
| DP-2 | No major-version migration guide (promised by §3.1/§10 + FIXPLAN gate) | High | high | PKG-02 STILL-OPEN |
| DP-3 | Version pinned `1.18.0` for a breaking rewrite; no version symbol; nothing gates bump | High | high | PKG-01/03 STILL-OPEN |
| DP-4 | `CMDS.md` references deleted paths/tests/config and a dead engine flow | Med | high | DOC-01/02/03 STILL-OPEN |
| DP-5 | `python-box` dead dep; `jinja2`/`tomllib` used only by `prompts.py` (slated for deletion) | Med | high | new (partial FP overlap) |
| DP-6 | `numpy<=2.1.3` cap unmotivated after torch removal; blocks newer CPython | Med | high | PKG-04 STILL-OPEN |
| DP-7 | Docs cover neither the real `RuntimeConfig/EngineSpec` API nor the Symbol/ops surface | Med | high | new |
| DP-8 | Public root (`__all__` + 74 names) contradicts design §3.1 and `test_public_cutover.py` | Med | high | seed #3; PARTIALLY-FIXED |
| — | description/keywords, `source-exclude`, extras, `.gitbook.yaml` structure | PASS | high | keep |

---

## Detailed findings

### DP-1 — README + every `docs/source` page uses an API that does not exist (High)

**What.** The shipping docs' code examples import and use `Provider`, `ProviderEngineConfig`,
`create_runtime`, and `TransportConfig`, and construct
`RuntimeConfig(language_model=ProviderEngineConfig(provider=…, model=…, api_key=…))`.
The live package exposes none of these.

**Where.** Live verification:

```
$ python -c "import symai; print(hasattr(symai,'create_runtime'), hasattr(symai,'Provider'), \
    hasattr(symai,'ProviderEngineConfig'), hasattr(symai,'TransportConfig'))"
False False False False
```

The **actual** API (`symai/runtime/config.py`, `symai/loading.py`):

```python
class RuntimeConfig(FrozenModel):
    language_models: Mapping[str, EngineSpec] = Field(default_factory=dict)
    embeddings: Mapping[str, EngineSpec] = Field(default_factory=dict)
    default_language_model: str | None = None
    default_embedding: str | None = None
# entrypoint is load_runtime(config) → Runtime;  no create_runtime, no ProviderEngineConfig,
# no Provider enum, no api_key field (engine credentials live in EngineSpec.settings).
```

Offending doc lines (representative):

- `README.md:45-54` — `from symai import (… Provider, ProviderEngineConfig, RuntimeConfig, … create_runtime)`
  and `RuntimeConfig(language_model=ProviderEngineConfig(provider=Provider.OPENAI, model="gpt-5.4", api_key=…))`.
- `docs/source/INTRODUCTION.md:21` — `from symai import Provider, ProviderEngineConfig, RuntimeConfig`.
- `docs/source/QUICKSTART.md:14-22` and `docs/source/QUICKSTART.md:53` — same imports, `with create_runtime(config) as runtime`.
- `docs/source/RUNTIME.md:42` and `:74-80` — `from symai import … create_runtime`; `ProviderEngineConfig.provider`.
- `docs/source/EMBEDDINGS.md:24-30` — `from symai import (… Provider, ProviderEngineConfig, … create_runtime)`; `Provider.OPENAI`.
- `docs/source/INSTALLATION.md:38-51` — `from symai import Provider, ProviderEngineConfig, RuntimeConfig, TransportConfig`
  plus a `TransportConfig(request_timeout=…, connect_timeout=…, connect_retries=…)` block.

**Why it matters.** These are not stale prose — they are copy-paste examples that fail at the
first import line. `tests/test_public_cutover.py` (`FORBIDDEN_PUBLIC_NAMES`, lines 43-60, and
`test_runtime_configuration_has_a_clean_module_cutover`, line 128) *asserts* `Provider`,
`ProviderEngineConfig`, `create_runtime`, `TransportConfig`, `NamedEngineConfig` are absent —
so the docs are documenting names the test suite deliberately forbids. The prior audit's DOC-03
note ("the five `docs/source/*` runtime pages + README were verified accurate — every snippet
executes") was true at `a220d6f` and is now **false**: the engine-redesign cutover changed the
runtime surface and the docs were not migrated.

**Proposed change.** Rewrite all six documents against the real surface:
`EngineSpec(implementation="openai:responses", settings={...})` inside
`RuntimeConfig(language_models={"main": …}, embeddings={"main": …})`, obtained via
`load_runtime(config)`, executed with `runtime.execute(request, engine="main")`. Add the
Symbol/ops surface (see DP-7). Do NOT hand-wave — the config shape is entirely different
(named mappings of `EngineSpec`, not a scalar `ProviderEngineConfig`).

**Feature impact:** keeps-all (docs only). **Impact:** high. **Effort:** M.

---

### DP-2 — No major-version migration guide (High)

**What.** `SYMBOL_REDESIGN.md:81` ("the major-version migration guide documents them together
with the Runtime model/configuration imports"), `SYMBOL_REDESIGN.md:318` ("migration
documentation maps each retained semantic method to its `ops.*` function"), and
`FIXPLAN.md:423` ("Publish as a new major version with a migration guide") all promise one.

**Where.** No such file exists:

```
$ find . -path ./.venv -prune -o -type f \( -iname "*migrat*" -o -iname "CHANGELOG*" \
    -o -iname "*UPGRADE*" \) -print       # → (nothing)
```

`docs/source/SUMMARY.md` lists only Introduction / Installation / Quickstart / Runtime /
Embeddings — no migration entry. The only "migration" strings in the repo are the audit docs
themselves flagging its absence (`audit/FINDINGS.md:412` PKG-02; `audit/README.md:46`
"Not publishable … no settled final major surface").

**Why it matters.** This is a hard release gate per FIXPLAN. `from symai import Symbol`,
`Expression`, `sym_return_type`, `.save()/.load()`, `adapt()` all now fail with no map to
`ops.*`/`Function`/`decode_output` replacements. §8 of the design has the exact removed→reason
table that a migration guide should be built from; it just hasn't been turned into user docs.

**Proposed change.** Author `docs/source/MIGRATION.md` (add to `SUMMARY.md`): a table mapping
each removed/moved concept (`Symbol.<method>` → `ops.<ns>.<fn>`; `Expression`/`Result` →
`Function` + `decode_output`; `sym_return_type` → decoders; `save/load/adapt/clear` → removed)
plus the canonical-import list from §3.1. **Feature impact:** keeps-all. **Impact:** high. **Effort:** M.

---

### DP-3 — Version `1.18.0` for a breaking rewrite; no gate, no version symbol (High)

**What / Where.** `pyproject.toml:11` → `version = "1.18.0"` — identical to the pre-rewrite
line, yet the public surface is wholly incompatible. Live check: `symai.__version__` and
`symai.SYMAI_VERSION` both absent (`False`). No `[project.scripts]`. `.github/` contains only
`FUNDING.yml` — there is **no CI/publish workflow gating the version**, so nothing mechanical
stops a `1.18.0` wheel from shipping.

**Why it matters.** `pip install -U symbolicai` would deliver a silent breaking change under a
minor-looking bump (PKG-01). `FIXPLAN.md:423` explicitly requires: "Complete the Runtime,
Function/decoder, and Symbol cutovers before setting the release version… Do not ship an
intermediate major that immediately requires another major for the Symbol redesign." So the
version bump is intentionally *gated on* the cutover completing (see DP-8) — which is not yet done.

**Proposed change.** Bump to `2.0.0` **after** the root/public cutover and Symbol surface land;
optionally restore an introspectable `symai.__version__` (was in `__all__` on `main`, PKG-03).
**Feature impact:** keeps-all. **Impact:** high. **Effort:** S (but sequenced after DP-8).

---

### DP-4 — `CMDS.md` is a fully dead workflow doc (Medium)

**What.** Every operational instruction in `CMDS.md` points at deleted or non-existent things.

**Where.**
- `CMDS.md:87-89` — `ruff … symai/backend/mixin/deepseek.py` and
  `symai/backend/engines/neurosymbolic/engine_deepseekX_reasoning.py`. Neither exists;
  `symai/backend/` contains only an empty `__init__.py` (+ stale `__pycache__`). The DeepSeek
  engine is now `symai/providers/deepseek/engines/chat_completions.py`.
- `CMDS.md:14,46,49,58,64,70` — `uv run pytest tests/engines --engine-api=mock|live`.
  `tests/engines/` holds only a stale `test_engine_handle.cpython-314-pytest-9.1.1.pyc` (no
  `.py` source), and `--engine-api` is not a registered option (no `conftest.py`/`pytest_addoption`
  — DOC-01, reproduced by prior audit).
- `CMDS.md:9,49` — configure `.venv/.symai/symai.config.json` and pick a key from `api_keys.log`.
  The library reads no config file and no env (grep confirms zero `environ/getenv/dotenv` in `symai/`);
  directly contradicts `docs/source/INSTALLATION.md` (DOC-03).
- `CMDS.md:77-82` — engine flow `prepare → build_request → call_request → parse_response`. The
  real engine protocol (`symai/runtime/engines.py`) is exactly `{close, execute}` — asserted by
  `test_public_cutover.py:160-163`.

**Why it matters.** `CMDS.md` is at repo root and reads as authoritative workflow; it is 100% wrong.

**Proposed change.** Rewrite against `tests/providers/**` + `tests/runtime/**` and the real
`load_runtime`/`execute` flow, or delete it if superseded by `docs/source`. **Feature impact:**
keeps-all. **Impact:** med. **Effort:** S.

---

### DP-5 — `python-box` dead dependency; `jinja2`/`tomllib` transitively dead (Medium)

**What / Where.** `pyproject.toml:26-32` `dependencies`:

```toml
"numpy>=1.26.4,<=2.1.3",
"pydantic>=2.8.2",
"jinja2>=3.1.0",
"python-box>=7.1.1",
"httpx>=0.28.1",
```

Verified imports across `symai/`:
- `python-box` / `box` → **zero** imports anywhere. Dead, remove.
- `jinja2` → only `symai/prompts.py:9,80-83`.
- `tomllib` → only `symai/prompts.py:3,130` (and `tomllib` is stdlib since 3.11 — never a
  declared dep anyway).
- `numpy` → only `symai/ops/embed.py`. `pydantic` / `httpx` → widely used (keep).

`prompts.py` is on the deletion list (`test_public_cutover.py` `DELETED_FILES` includes
`prompts.py`; `test_deleted_modules_have_no_import_spec` asserts `find_spec("symai.prompts") is
None`). It still exists and is still imported by `function.py:5` and `ops/{text,reason,rank,compare}.py`,
so those cutover tests currently fail. **Once `prompts.py` is removed, `jinja2` also becomes a
dead dependency.**

**Why it matters.** Ships install weight and supply-chain surface for code that either isn't
used (`python-box`) or is scheduled for removal (`jinja2`).

**Proposed change.** Drop `python-box` now. Drop `jinja2` as part of the `prompts.py` removal
(coordinate with the dead-code lens — that lens owns whether `prompts.py`'s example classes get
relocated into `ops/`). **Feature impact:** `python-box` keeps-all; `jinja2` drops-minimal
(only if prompt templating is genuinely retired). **Impact:** med. **Effort:** S.

---

### DP-6 — `numpy<=2.1.3` upper cap is unmotivated post-torch (Medium)

**What / Where.** `pyproject.toml:27` → `numpy>=1.26.4,<=2.1.3`. `torch` has **zero** hits in
`symai/` and `pyproject.toml`. The cap was inherited from the torch-constrained era; with torch
gone and `requires-python = ">=3.11"` (no upper bound), the `<=2.1.3` ceiling blocks newer
CPython (2.1.3 predates 3.13/3.14 wheel coverage) while `requires-python` advertises support.
numpy is a hard dependency used only by the embedding-math helpers in `ops/embed.py` (PKG-04).

**Proposed change.** Relax to a lower-bound-only or `<3` constraint. Optionally reconsider making
numpy an extra given its single-module use, but that's a larger call (embed similarity/distance
math is a stated feature). **Feature impact:** keeps-all. **Impact:** med. **Effort:** S.

---

### DP-7 — Docs describe neither the real runtime API nor the Symbol/ops surface (Medium)

**What.** Beyond the dead-name problem (DP-1), the docs' *conceptual* model is also wrong two
ways. They present `runtime.execute(request)` with a scalar `ProviderEngineConfig` — but the
real config is a **named mapping** of `EngineSpec` (multiple engines per runtime, `engine=`
selection is the whole point per `SYMBOL_REDESIGN.md:305`). And the entire product-goal surface —
`Symbol`, `ops.text/reason/compare/rank/embed`, `Function`, `decode_output` (`SYMBOL_REDESIGN.md`
§1 "Preserve the recognizable experience of composing operations with Symbols") — appears in
**zero** documentation pages. A reader of the docs would never learn Symbol/ops exists.

**Why it matters.** The docs undersell and misrepresent the library. The redesign's raison
d'être (the Symbol value DSL + ops) is undocumented.

**Proposed change.** Add Symbol/ops/Function/decoding pages to `docs/source` + `SUMMARY.md`;
show `engine=`-keyed named engines. **Feature impact:** keeps-all. **Impact:** med. **Effort:** M.

---

### DP-8 — Public root contradicts the design and the cutover tests (Medium; in-progress)

**What.** `SYMBOL_REDESIGN.md:81` — "The package root is empty rather than a compatibility
facade." `tests/test_public_cutover.py` encodes this: `test_old_root_names_are_absent…` asserts
`not hasattr(symai, "__all__")` and `test_import_symai_is_subprocess_isolated_and_inert` asserts
`"public_names": []`. **Live, the opposite holds:** `import symai` has `__all__` and **74**
public names (`symai/__init__.py:1-142` re-exports decoders, runtime models, errors, `Runtime`,
`current_runtime`, `load_runtime`, …). So the facade is a superseded design (seed #3):
**PARTIALLY-FIXED / IN-PROGRESS** — the redesign trimmed some names (`Symbol` not re-exported)
but the root is still a ~74-name facade.

Consequently several cutover tests currently **fail** against this snapshot (moving target):
`test_old_root_names_are_absent…`, `test_import_symai_is_subprocess_isolated_and_inert`
(root not empty), `test_deleted_modules_have_no_import_spec` / `test_deleted_production_tree…`
(`prompts.py` still present), and `test_production_ast_has_no_legacy_graph_references`
(`current_runtime`/`NoActiveRuntimeError`/`Prompt` still referenced — the FORBIDDEN_IDENTIFIERS
set). `__init__.py` even imports `current_runtime` and `NoActiveRuntimeError`, which
`test_runtime_module_exposes_no_ambient_registry…` forbids.

**Why it matters for packaging.** What `import symai` exposes is the public contract the docs and
migration guide must document. The version bump (DP-3) is explicitly gated on this cutover
completing (FIXPLAN L). This overlaps the runtime/public-surface lens — I flag it here only as a
**publishability gate**, not to re-litigate the module design.

**Proposed change.** Land the root cutover (empty root or a small, deliberate `__all__`), remove
`prompts.py` / ambient-runtime references, then reconcile with the design's "canonical imports
from owning modules" statement — and only then bump the version and write the migration guide
against the *final* surface. **Feature impact:** keeps-all. **Impact:** med. **Effort:** M (owned elsewhere).

---

## What is already good (keep)

- **`description` + `keywords` are accurate.** `pyproject.toml:16,19` name OpenAI/Cerebras/DeepSeek
  language models + OpenAI embeddings — exactly the four implemented builtins in
  `symai/loading.py` (`openai:responses`, `cerebras:chat-completions`, `deepseek:chat-completions`,
  `openai:embeddings`). No stale Anthropic/Groq/etc. claims. Keep.
- **`source-exclude` is correct** (`pyproject.toml:37-52`): excludes `/docs`, `/tests`,
  `/examples`, `/artifacts`, `/build`, `/dist`, `/.github`, `uv.lock`, `ruff.toml`, `pytest.ini`,
  `*.egg-info`, `__pycache__`, `*.pyc`, `.DS_Store`. Docs and tests will not ship in the wheel. Keep.
  (Minor: `/examples` is excluded but no `examples/` dir exists — harmless.)
- **Build backend** (`uv_build`, `module-name = "symai"`, `module-root = ""`) is coherent. Keep.
- **Extras structure is consistent with the stated design** — no `[project.optional-dependencies]`;
  only `[dependency-groups] dev = [ruff, pytest, pytest-xdist]`. README/INSTALLATION both say "no
  capability extras," matching reality. (A future extras tidy-up is a separate planned pass, not a
  gap for this release.)
- **`.gitbook.yaml` structure is internally valid**: `root: symbolicai/docs/source/`,
  `readme: INTRODUCTION.md`, `summary: SUMMARY.md`; `SUMMARY.md` links resolve to the five existing
  pages. (Its *content* is broken per DP-1, but the wiring is fine.)

---

## Publishability checklist

| Item | Status | Note |
|------|--------|------|
| Version reflects breaking change | **FAIL** | `1.18.0`; must be `2.0.0` (gate: DP-8 cutover) |
| Version bump gated / CI publish guard | **FAIL** | No publish workflow; `.github/` only `FUNDING.yml` |
| Introspectable version symbol | **FAIL** | `symai.__version__`/`SYMAI_VERSION` absent |
| Migration guide exists | **FAIL** | No `MIGRATION*`/`CHANGELOG*`; promised by §3.1/§10 + FIXPLAN |
| README examples run | **FAIL** | Imports `Provider`/`ProviderEngineConfig`/`create_runtime` → ImportError |
| `docs/source/**` examples run | **FAIL** | All 5 pages use the same dead API (+ `TransportConfig`) |
| Docs cover the real API (RuntimeConfig/EngineSpec/load_runtime) | **FAIL** | Docs use scalar `ProviderEngineConfig` that doesn't exist |
| Docs cover Symbol/ops/Function/decoding | **FAIL** | Entirely absent from docs |
| `CMDS.md` accurate | **FAIL** | Dead paths, tests, config, engine flow |
| `dependencies` match imports | **FAIL** | `python-box` unused; `jinja2` only via to-be-deleted `prompts.py` |
| `numpy` constraint sane | **FAIL** | `<=2.1.3` cap unmotivated post-torch; blocks newer CPython |
| `description`/`keywords` accurate | **PASS** | Matches implemented providers |
| `source-exclude` correct | **PASS** | Docs/tests/caches excluded from wheel |
| Extras structure | **PASS** | No capability extras, consistent with docs |
| `.gitbook.yaml` wiring | **PASS** | Structure valid (content broken via DP-1) |
| Build backend config | **PASS** | `uv_build`, module wiring coherent |

**Verdict: NOT publishable.** Blocking: DP-1 (docs run against a dead API), DP-2 (no migration
guide), DP-3 (version), DP-4 (CMDS.md), plus the DP-8 cutover the version bump depends on.
Packaging metadata itself (description, exclude, backend, extras) is sound.
