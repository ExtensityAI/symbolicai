# r1-09 — Dead Code & Legacy Residue

**Lens:** Dead code, vestigial modules, and dead dependencies. Read-only audit; nothing deleted.
**Snapshot:** worktree `refactor/cleanup` @ live edit; anchors are symbol + snippet (line numbers approximate).

## Executive summary

1. **`symai/prompts.py` (1043 LOC) is the single largest pile of legacy residue and is slated for outright deletion.** The repo's own `tests/test_public_cutover.py` lists `prompts.py` in `DELETED_FILES`, forbids the identifiers `Prompt`/`PromptRegistry`, and its three cutover tests **currently FAIL** because the file still exists. So this is not a judgment call — the intended end-state deletes it; the work is unfinished.
2. Of **34 `Prompt` subclasses, 21 are fully dead** (zero references anywhere) and **13 are "used"** only in that ops modules read their `.value` example-string lists at import time via `tuple(X().value)`. The `Prompt` class machinery itself (jinja templating, `format_kwargs`, `__call__`, `__len__`, the `Callable` branch) is dead; `PromptRegistry` is 100% dead.
3. **Two dependencies are dead: `python-box` is imported nowhere in the entire repo, and `jinja2` is used only by the dead `PromptRegistry`.** `tomllib` (stdlib) is likewise only used by `PromptRegistry`. `numpy`, `pydantic`, `httpx` are all genuinely used.
4. **`symai/backend/` is a 0-byte empty-`__init__` vestige** that `test_public_cutover.py` asserts must not exist (test currently FAILS on it).
5. **`operations.image_request` / `data_uri` have no ops-layer consumer** (tested-only), but the underlying multimodal path (`ImageContent`) is wired through all three engines — so they are a partially-exposed real feature, not vestige. `parse_embedding_response` is used but its index check duplicates `ops.embed.embed`'s stricter inline check.

Overall read: the dead surface is large but unusually well-bounded — almost all of it lives in one file (`prompts.py`) plus the `backend/` stub, and the test suite already encodes the target end-state. This is finishable cleanup, not archaeology.

---

## Findings table

| # | Symbol / file | Verdict | Safe to remove? | Feature impact |
|---|---|---|---|---|
| A | `symai/prompts.py` whole module (1043 LOC) | DEAD-as-module (intended delete) | Yes, after relocating 13 example lists | keeps-all (data relocated) |
| B | `PromptRegistry` + jinja2 + tomllib machinery | DEAD | Yes | drops-minimal (unused template engine) |
| C | 21 `Prompt` subclasses (see table) | DEAD | Yes | keeps-all |
| D | 13 `Prompt` subclasses (data-only used) | PARTIAL (class dead, data live) | Class: yes; data: relocate | keeps-all |
| E | `Prompt` base class + `function.py` `isinstance(examples, Prompt)` branch | PARTIAL (only as str-tuple source) | Yes, if examples become plain tuples | keeps-all |
| F | dep `python-box` (`box`) | DEAD | Yes | keeps-all |
| G | dep `jinja2` | DEAD (only PromptRegistry) | Yes, with B | keeps-all |
| H | `symai/backend/__init__.py` (0 bytes) | DEAD vestige | Yes | keeps-all |
| I | `operations.image_request` / `data_uri` | PARTIAL (no op consumer; tested; runtime-wired) | No (drops multimodal builders) | drops-real-feature if removed |
| J | `operations.parse_embedding_response` | USED (redundant index check) | No (dedup, don't delete) | keeps-all |
| K | root `__init__.__all__` (~69 re-exports) | candidate (public-surface) | Defer to API lens | n/a |

---

## Detailed findings

### A + B + C + D + E — `symai/prompts.py` is legacy residue slated for deletion

**What.** The entire 1043-LOC module is a legacy few-shot-example container (`Prompt`) plus an unused jinja/toml template registry (`PromptRegistry`) plus 34 example classes. Only the raw example strings of 13 classes are actually consumed.

**Where — the intent is encoded in the test suite.** `tests/test_public_cutover.py`:

```python
DELETED_FILES = { ... "prompts.py", ... }          # line 70
FORBIDDEN_IDENTIFIERS = { ... "Prompt", "PromptRegistry", ... }   # lines 113-114
```

and three tests assert it's gone. **They currently FAIL** (verified by running `uv run pytest tests/test_public_cutover.py -q`):

```
test_deleted_modules_have_no_import_spec        AssertionError: symai.backend
test_deleted_production_tree_and_adapter_inventory  AssertionError: prompts.py
test_production_ast_has_no_legacy_graph_references  75 violations, first: 'prompts.py:15: definition Prompt'
6 failed, 4 passed
```

**Who still imports it** (grep `from symai.prompts` across `symai/**`):

```
symai/function.py:5:      from symai.prompts import Prompt
symai/ops/reason.py:6:    from symai.prompts import LogicExpression, SimpleSymbolicExpression
symai/ops/text.py:6:      from symai.prompts import (CombineText, ExtractPattern, Format,
                                                     IncludeText, MapExpression, Modify, ReplaceText)
symai/ops/rank.py:6:      from symai.prompts import RankList
symai/ops/compare.py:6:   from symai.prompts import ContainsValue, FuzzyEquals, IsInstanceOf
```

The 13 imported subclasses are used **only** as data:

```python
# ops/text.py
_MODIFY_EXAMPLES = tuple(Modify().value)   # a tuple[str, ...]
...
function = Function("Modify the text ...\n", examples=_MODIFY_EXAMPLES)
```

Every `examples=` callsite in `symai/` passes a pre-materialized `_XXX_EXAMPLES` string tuple — **never a `Prompt` instance** (verified: `grep -rn "examples=[A-Z]" symai/ tests/` returns nothing). Therefore the `Prompt` **class** and its methods (`__str__`/`format_kwargs` templating, `__call__`, `__len__`, `__repr__`, the `Callable` input branch) are all dead; only the literal example strings matter. Consequently `function.py`'s `isinstance(examples, Prompt)` branch (`_normalize_examples`, lines 92-93) is dead in practice.

`PromptRegistry` (lines 76-171) — jinja2 `Environment`, `load_from_folder`, `manifest`/`set_manifest`, `register_template`/`render`/`has_template`, `tojson_filter` — has **zero references** anywhere outside its own definition and the forbidden-name string in the test:

```
$ grep -rnw PromptRegistry symai/ tests/ | grep -v prompts.py
tests/test_public_cutover.py:114:    "PromptRegistry",   # <- it's in the FORBIDDEN list, not a consumer
```

**Proposed change.** Relocate the 13 live example lists as plain `tuple[str, ...]` module constants next to each op (or a tiny `ops/_examples.py` data module that defines **no** `Prompt` class), drop `Prompt`/`PromptRegistry` and the 21 dead classes, delete `prompts.py`, and drop the `isinstance(examples, Prompt)` branch + the `Prompt` import/type in `function.py` (so `examples: Sequence[str] | str | None`). This satisfies the already-failing cutover tests.

**Feature impact:** keeps-all — the few-shot example DATA is preserved; only the unused class wrapper and template engine are dropped.
**Confidence:** high · **Impact:** high (1043 LOC + a failing release gate) · **Effort:** M.

#### Per-class verdict table (`Prompt` subclasses)

`grep -rnw <ClassName> symai/ tests/ | grep -v prompts.py` was run for each. "USED (data)" = imported by an ops module purely for `.value`.

| Prompt subclass | Used-by evidence | Verdict |
|---|---|---|
| `FuzzyEquals` | `ops/compare.py:6` (`_EQUALS_EXAMPLES`) | USED (data) |
| `Modify` | `ops/text.py:6` (`_MODIFY_EXAMPLES`) | USED (data) |
| `MapExpression` | `ops/text.py:6` (`_MAP_EXAMPLES`) | USED (data) |
| `Format` | `ops/text.py:6` (`_FORMAT_EXAMPLES`) | USED (data) |
| `RankList` | `ops/rank.py:6` (`_RANK_EXAMPLES`) | USED (data) |
| `ContainsValue` | `ops/compare.py:6` (`_CONTAINS_EXAMPLES`) | USED (data) |
| `IsInstanceOf` | `ops/compare.py:6` (`_IS_INSTANCE_OF_EXAMPLES`) | USED (data) |
| `ExtractPattern` | `ops/text.py:6` (`_EXTRACT_EXAMPLES`) | USED (data) |
| `SimpleSymbolicExpression` | `ops/reason.py:6` (`_INTERPRET_EXAMPLES`) | USED (data) |
| `LogicExpression` | `ops/reason.py:6` (`_LOGIC_EXAMPLES`) | USED (data) |
| `ReplaceText` | `ops/text.py:6` (`_REPLACE_EXAMPLES`) | USED (data) |
| `IncludeText` | `ops/text.py:6` (`_INCLUDE_EXAMPLES`) | USED (data) |
| `CombineText` | `ops/text.py:6` (`_COMBINE_EXAMPLES`) | USED (data) |
| `SufficientInformation` | none | **DEAD** |
| `Filter` | none¹ | **DEAD** |
| `SemanticMapping` | none | **DEAD** |
| `Transcription` | none | **DEAD** |
| `ExceptionMapping` | none | **DEAD** |
| `CompareValues` | none² | **DEAD** |
| `StartsWith` | none | **DEAD** |
| `EndsWith` | none | **DEAD** |
| `InvertExpression` | none | **DEAD** |
| `NegateStatement` | none | **DEAD** |
| `CleanText` | none | **DEAD** |
| `ListObjects` | none | **DEAD** |
| `ForEach` | none | **DEAD** |
| `MapContent` | none | **DEAD** |
| `Index` | none | **DEAD** |
| `SetIndex` | none | **DEAD** |
| `RemoveIndex` | none | **DEAD** |
| `SimulateCode` | none | **DEAD** |
| `GenerateCode` | none | **DEAD** |
| `TextToOutline` | none | **DEAD** |
| `UniqueKey` | none | **DEAD** |
| `ProbabilisticBooleanMode{Strict,Medium,Tolerant}` (module constants) | none | **DEAD** |

¹ `grep Filter` returns only string-literal hits (`"Filter the text..."` in `ops/text.py:112` and `test_symbol_runtime_cutover.py:146`) — the `filter` op does **not** use the `Filter` prompt class; it has no `examples=`. The class is dead.
² Note `ops/compare.py` uses `FuzzyEquals`, not `CompareValues`; the ordering-comparison examples in `CompareValues` are orphaned.

**Tally: 13 used-for-data, 21 dead subclasses, `Prompt` base + `PromptRegistry` + 3 constants dead.** No `tests/test_prompts.py` exists; the only test touching the module asserts its non-existence.

---

### F + G — Dead dependencies in `pyproject.toml`

**What.** `dependencies` (pyproject.toml lines 26-32) lists `numpy`, `pydantic`, `jinja2`, `python-box`, `httpx`.

**Where / evidence:**

```
$ grep -rn "from box\|import box\|Box(" symai/ tests/ .    →   (no box imports anywhere)
$ grep -rn "jinja" symai/ tests/    →   only symai/prompts.py (PromptRegistry)
$ grep -rn "tomllib" symai/ tests/  →   only symai/prompts.py (PromptRegistry.load_from_folder)
```

- **`python-box` (`box`): DEAD** — imported in zero files across the whole repo (including `.venv`-excluded search). Remove outright.
- **`jinja2`: DEAD** — sole consumer is `PromptRegistry` (`jinja2.Environment`, `jinja2.BaseLoader`, `jinja2.StrictUndefined`, `from_string`). Removable together with finding B.
- **`tomllib`** (stdlib, not a pyproject dep): sole consumer is `PromptRegistry.load_from_folder`. No action on deps, but confirms PromptRegistry is the only thing keeping toml parsing alive.
- **`numpy`** — USED (`ops/embed.py` math). **`pydantic`** — USED (models, decoding, providers). **`httpx`** — USED (all provider clients). Keep.

**Proposed change.** Delete `python-box>=7.1.1` now; delete `jinja2>=3.1.0` when `prompts.py` goes.
**Feature impact:** keeps-all (box) / drops-minimal (jinja — the unused template registry).
**Confidence:** high · **Impact:** med · **Effort:** S.

---

### H — `symai/backend/` empty vestige

**What / Where.** `symai/backend/__init__.py` is 0 bytes (`wc -c` = 0); `backend/` contains only that file (+ `__pycache__`).

**Evidence it's intended gone.** `test_public_cutover.py` `DELETED_TREES = {"backend", ...}` (line 85); `test_deleted_modules_have_no_import_spec` asserts `find_spec("symai.backend") is None` and **currently FAILS**:

```
AssertionError: symai.backend  (ModuleSpec ... origin='.../symai/backend/__init__.py')
```

The AST guard also treats `symai.backend.*` as a forbidden import prefix. Nothing imports `symai.backend` (the old `backend/*.py` files are all in `DELETED_FILES` and already gone; only the empty package dir remains).

**Proposed change.** Delete the `symai/backend/` directory.
**Feature impact:** keeps-all. **Confidence:** high · **Impact:** low · **Effort:** S.

---

### I — `operations.image_request` / `data_uri`: no ops consumer, but a real runtime-wired feature

**What.** `image_request` (operations.py:45) and `data_uri` (operations.py:71) build multimodal requests. No ops-layer function calls them.

**Where / evidence:**

```
$ grep -rn "image_request\|data_uri" symai/ tests/
symai/operations.py         (definitions only)
tests/test_operations.py    (unit tests only — no op, no e2e)
```

But the multimodal path **is** wired end-to-end below the ops layer — `ImageContent` is consumed by every engine:

```
providers/deepseek/engines/chat_completions.py:220,285  isinstance(part, ImageContent)
providers/cerebras/engines/chat_completions.py:195,262-266  -> chat_api.ImageContentPart(...)
providers/openai/engines/responses.py:172  has_image = any(isinstance(content, ImageContent) ...)
```

**Verdict: PARTIAL, not dead.** These are builder helpers for a genuinely-plumbed capability that simply has no ergonomic ops entry point yet. Deleting them would drop the only convenient way to construct an image request (`data_uri` + `image_request`).

**Proposed change.** Keep. If the intent is a lean surface, either (a) expose an `ops` image operation that uses them, or (b) consciously drop multimodal (engines included) — but do not silently delete just the builders. Do **not** classify as dead code.
**Feature impact:** drops-real-feature (multimodal request building) **if removed**; keeps-all if left.
**Confidence:** high · **Impact:** med · **Effort:** S (decision, not code).

---

### J — `operations.parse_embedding_response`: used, but duplicate index validation

**What.** `ops.embed.embed` validates response indices inline, then calls `parse_embedding_response`, which re-derives indices and re-validates uniqueness.

**Where:**

```python
# ops/embed.py — embed()
indices = tuple(vector.index for vector in response.vectors)
expected_indices = set(range(len(inputs)))
if len(indices) != len(inputs) or set(indices) != expected_indices:   # STRICTER check
    raise ValueError("Embedding response indices must exactly match input indices")
return Symbol(parse_embedding_response(response))

# operations.py — parse_embedding_response()
indices = tuple(vector.index for vector in response.vectors)
if len(indices) != len(set(indices)):                                 # REDUNDANT (subset of above)
    raise ValueError("Embedding response indices must be unique")
return [ ... sorted(...) ... ]
```

`embed`'s "exact match to `range(len(inputs))`" already implies uniqueness, so `parse_embedding_response`'s uniqueness `if` is dead on the `embed` path. But `parse_embedding_response` is **USED** (embed relies on it for the sort + float cast) and is independently unit-tested, so it is **not** a deletion candidate — it's a dedup opportunity (fold the sort/cast into `embed`, or have `embed` not pre-check and rely on `parse_embedding_response`).

**Feature impact:** keeps-all. **Confidence:** high · **Impact:** low · **Effort:** S.

---

### K — Root `__init__.__all__` re-exports (public-surface candidate)

**What.** `symai/__init__.py` re-exports **69** names in `__all__` while (per seed #3) omitting `Symbol` and the `ops` namespaces. The design doc says the root "is not a compatibility facade."

**Evidence / caveat.** Every `__all__` name resolves to a real defined type; most are consumed by tests (e.g. `ResponseMetadata` 13 test files, `FinishReason` 9, `ReasoningFormat` 6). Being "unused by tests" does **not** make a public export dead — so per _CONTEXT conventions I flag this only as a **candidate**, deferring the keep/trim call to the public-API lens. One concrete data point: `ProviderId` has **zero** consumers importing it from the root (it's used internally in `runtime/models.py`, `config.py`, `errors.py`, and provider engines) — a plausible over-export. The bespoke JSON AST exports (`JsonObject`/`JsonArray`/`JsonEntry`, seed #6) are lightly test-touched (1-4 files) and may be internal detail rather than public surface.

**Proposed change.** None from this lens beyond flagging. Public-API lens should decide whether the root facade should shrink (design says yes) and whether internal types like `ProviderId`/the Json AST belong in the public `__all__`.
**Confidence:** med · **Impact:** med · **Effort:** M (owned by another lens).

---

## What is already good (keep)

- **The dead surface is bounded to one file + one stub.** Almost all legacy residue is `prompts.py` and the `backend/` stub — not scattered across the tree. The `providers/`, `runtime/`, and `ops/` layers show no orphaned modules.
- **The cutover is test-driven.** `test_public_cutover.py` already encodes the target end-state (deleted files, forbidden identifiers, an AST guard). It's failing today, which is exactly the signal an auditor wants — the finish line is written down, not guessed.
- **The example DATA is worth keeping.** The 13 used few-shot lists are carefully curated; the recommendation is to relocate them as plain tuples, not discard them.
- **Dependency hygiene is close.** Only 2 of 5 runtime deps are dead, and both are traceable to a single dead module.
- **Multimodal is genuinely wired** through all three engines — `image_request`/`data_uri` are a real (if unexposed) feature, not cruft; classifying them correctly avoids a feature-losing deletion.

---

## Verification commands run (read-only)

- `grep -rnw <Symbol> symai/ tests/ | grep -v prompts.py` for every `Prompt` subclass, `PromptRegistry`, and the `ProbabilisticBooleanMode*` constants.
- `grep -rn "jinja\|box\|tomllib\|image_request\|data_uri\|parse_embedding_response\|examples=" symai/ tests/ .`
- `uv run pytest tests/test_public_cutover.py -q` → 6 failed / 4 passed (prompts.py + backend still present; 75 AST violations).
- `uv run python` AST-violation dump grouped by file (43 `prompts.py`, 12 `function.py`, 13 across `ops/*`, 6 `runtime/runtime.py`, 1 `runtime/errors.py`).
- `wc -c symai/backend/__init__.py` → 0.
