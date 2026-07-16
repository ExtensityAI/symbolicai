# R4 — Prompt / Few-Shot Fidelity Audit

Scope: every inlined few-shot example tuple and instruction string in
`symai/ops/{text,reason,compare,rank,embed}.py`, audited against the deleted
`symai/prompts.py` recovered from git history.

---

## Executive summary

1. **The `prompts.py` → inline migration is byte-for-byte faithful.** All **13**
   example sets were recovered from `git show a220d6f:symai/prompts.py` (the last
   revision before deletion) and diffed element-by-element (AST `literal_eval`,
   Python 3.12) against the current tuples. Every one is **IDENTICAL** after
   implicit string-concatenation is resolved. The migration introduced **zero**
   new example-content defects.
2. **Every seed defect is real but PRE-EXISTING** — it was already baked into
   `prompts.py` and copied over verbatim: the malformed rank list
   `[33, 'a', , 'help', …]`, the merged Format triple, the merged Contains pair
   with the stale `Symbol(...)` repr, and the `isinstanceof`/`instanceof` token
   split all exist unchanged in the original.
3. **Two defects ARE migration-introduced — and they are the two most damaging.**
   The op *query builders* were rewritten and drifted away from the examples:
   `rank` **dropped the `order:` field** that leads all 13 examples (and the
   instruction now hardcodes "highest to lowest" while 6/13 examples are `asc`);
   `is_instance_of` **changed the operator token from `isinstanceof` to
   `is instance of`**, so the query now matches *neither* example token.
4. One long-standing cross-layer mismatch survives faithfully: `logic` wraps
   operands in colons (`expr :X: op :Y:`) but its examples never use colons
   (`expr X op Y`). This was already inconsistent in the old preprocessor.
5. `embed.py` has no few-shot examples (pure numpy) and `filter/style/summarize/
   translate/query/template` pass no `examples=` — all fidelity-clean by
   construction. Net: the raw data was migrated perfectly; the defects are old
   content bugs plus two query-builder regressions, all fixable in-place.

---

## Method

- Recovered the original example lists: `git show a220d6f:symai/prompts.py`
  (1042 LOC, the commit immediately before `84f703b` deleted it). Cross-checked
  against `28ede02^` (pre-cleanup, 1327 LOC) — the relevant classes are identical.
- Recovered the original **query templates** (not in `prompts.py`) from
  `git show 28ede02^:symai/pre_processors.py` and `…:symai/core.py`, so that
  instruction/query-vs-example mismatches could be classified as
  *pre-existing* vs *migration-introduced*.
- Diffed each original `super().__init__([...])` list against the current
  `_*_EXAMPLES` tuple with a Python-3.12 AST script (`literal_eval` both sides,
  element-wise compare). Result: 13/13 IDENTICAL, matching counts.

Prompt-class → inlined-tuple map (all IDENTICAL):

| Original class | Inlined tuple | count |
|---|---|---|
| `FuzzyEquals` | `compare._EQUALS_EXAMPLES` | 35 |
| `ContainsValue` | `compare._CONTAINS_EXAMPLES` | 25 |
| `IsInstanceOf` | `compare._IS_INSTANCE_OF_EXAMPLES` | 24 |
| `Modify` | `text._MODIFY_EXAMPLES` | 7 |
| `MapExpression` | `text._MAP_EXAMPLES` | 7 |
| `Format` | `text._FORMAT_EXAMPLES` | 9 |
| `ReplaceText` | `text._REPLACE_EXAMPLES` | 15 |
| `IncludeText` | `text._INCLUDE_EXAMPLES` | 9 |
| `CombineText` | `text._COMBINE_EXAMPLES` | 24 |
| `ExtractPattern` | `text._EXTRACT_EXAMPLES` | 17 |
| `SimpleSymbolicExpression` | `reason._INTERPRET_EXAMPLES` | 39 |
| `LogicExpression` | `reason._LOGIC_EXAMPLES` | 20 |
| `RankList` | `rank._RANK_EXAMPLES` | 13 |

(The "count" is the *literal* element count. Where it looks low — Format shows 9
not 11, Contains 25 not 26 — that undercount is itself a defect: examples were
merged by missing commas *in the original*. See D4/D6.)

---

## Defect table

| # | File → symbol / example | Defect | Origin | Severity | Fix |
|---|---|---|---|---|---|
| D1 | `rank._RANK_EXAMPLES` [idx 4 & 5] | Malformed list `[33, 'a', , 'help', 1234567890]` — stray comma → not valid Python; teaches the model to emit broken lists | pre-existing | degrades-output | Remove the empty slot: `[33, 'a', 'help', 1234567890]` and fix both RHS answers accordingly |
| D2 | `rank.rank()` query vs `_RANK_EXAMPLES` + instruction | Query builds `measure: '…' list: … =>` but **every** example is prefixed `order: 'desc'` / `order: 'asc'`; instruction hardcodes "highest to lowest" yet 6/13 examples are `asc`. Orphaned field + contradictory examples | **migration-introduced** | degrades-output | Restore `order` param + emit `order: '{order}' measure: '{measure}' list: {value} =>`, OR strip `order:` from all examples and drop the `asc` ones |
| D3 | `compare.is_instance_of()` query vs `_IS_INSTANCE_OF_EXAMPLES` | Query token is `is instance of` (spaces); examples use `isinstanceof` (×20) and `instanceof` (×4). Query matches **neither** | query change **migration-introduced**; intra-example split pre-existing | degrades-output | Set query token to `isinstanceof` (original) and normalize the 4 `instanceof` examples to `isinstanceof` |
| D4 | `compare._CONTAINS_EXAMPLES` [idx 3] | Two examples merged into one string (missing comma): `…=>True'Apple Inc.' in 'Microsoft…' =>False` — one string, two `=>` | pre-existing | degrades-output | Split into two tuple elements at the `=>True` boundary |
| D5 | `compare._CONTAINS_EXAMPLES` [idx 3] | Same string embeds a stale **old-Symbol repr**: `<class 'symai.symbol.Symbol'>(value=("[…]",))` — obsolete internal format under the redesign | pre-existing | cosmetic→degrades | Replace with a plain string/list literal example; drop the internal repr |
| D6 | `text._FORMAT_EXAMPLES` [idx 5] | **Three** examples merged into one (two missing commas): `…'japanese' =>…。text '…' format 'japanese romanji' =>…text 'April 1, 2020' format 'EU date' =>01.04.2020` | pre-existing | degrades-output | Split into three tuple elements |
| D7 | `reason.logic()` query vs `_LOGIC_EXAMPLES` | Query wraps operands in colons `expr :X: op :Y: =>`; all 20 examples use `expr X op Y =>` (no colons) | pre-existing (old preprocessor already mismatched) | degrades-output | Drop the colons in the query → `expr {left} {operator} {right} =>` to match examples |
| D8 | `reason._INTERPRET_EXAMPLES` [idx 22] | Dangling/postfix example `"x ≠ 5"  not  =>x = 5.` — operator trails the operand, double spaces, no second operand | pre-existing | cosmetic | Rewrite as `not("x = 5") =>x ≠ 5.` (matching the `not(...)` form used elsewhere in the same list) |
| D9 | `text._COMBINE_EXAMPLES` [idx 6 & 17] | Questionable answers + inconsistent spacing: `True + 0 => False` (only example with `=> ` spacing; dubious result) and `'One' + 'Two' =>3` (One+Two→3, not "Three") | pre-existing | cosmetic | Normalize spacing to `=>`; reconsider the two answers (low-confidence — may be intentional fuzzy semantics) |
| D10 | `compare._EQUALS_EXAMPLES` | Contains `<=` / `>=` comparison examples (e.g. `'eleven' <= 8 =>False`, `'eleven' <= 11 =>True`) that the `==`-only `equals` op never emits | pre-existing | cosmetic | Drop the `<=`/`>=` rows, or leave (harmless but off-instruction) |
| D11 | `text.modify()` / `text.replace()` queries | Query ends `'{arg}'=>` (no space before `=>`) while examples use ` =>` (with space). Faithful to the old preprocessors, but examples disagree | pre-existing | cosmetic | Add a space: `… '{new}' =>` for consistency with examples |

---

## Detailed findings

### D1 — `rank._RANK_EXAMPLES`: malformed list literal (stray comma)

`symai/ops/rank.py` (lines ~21 and ~23), both the `desc` and `asc` "length"
examples:

```
"order: 'desc' measure: 'length' list: [33, 'a', , 'help', 1234567890] =>['a', 33, 'help', 1234567890]"
"order: 'asc'  measure: 'length' list: [33, 'a', , 'help', 1234567890] =>[1234567890, 'help', 'a', 33]"
```

`[33, 'a', , 'help', 1234567890]` has an empty element between `'a'` and
`'help'`. It is invalid Python and it demonstrates malformed list syntax to the
model in a *ranking* op whose output is parsed as a list. Verified identical in
`prompts.py@a220d6f` (`RankList`), so this is an original bug carried over, not a
migration artifact. **Fix:** `[33, 'a', 'help', 1234567890]` (both examples).

### D2 — `rank`: query dropped the `order:` field the examples depend on (migration-introduced)

Original (`pre_processors.py::RankPreProcessor`):
```
f"order: '{order!s}' measure: '{measure!s}' list: {list_} =>"
```
Original `core.rank` took `order: str = "desc"` and the prompt read
*"Order the list … based on their quality measure and order literal"*.

Current `symai/ops/rank.py`:
```python
function = Function("Rank the objects from highest to lowest by the requested measure:\n",
                    examples=_RANK_EXAMPLES)
...
(f"measure: '{measure}' list: {source.value!s} =>",)
```
The `order:` field is gone from the query and from the signature, but **all 13
examples still lead with `order: 'desc'` / `order: 'asc'`**, and 6 of them show
ascending output. The model is shown a leading token it never receives, and half
the exemplars contradict the hardcoded "highest to lowest". This is the single
most consequential defect and it was introduced by the query rewrite. **Fix:**
either reinstate an `order` parameter and emit the `order:` prefix, or strip
`order:` from every example and delete the `asc` rows.

### D3 — `is_instance_of`: operator token changed to a form no example uses (migration-introduced)

Original (`IsInstanceOfPreProcessor`): `f"{a} isinstanceof {b} =>"` — matched the
20 `isinstanceof` examples. Current `symai/ops/compare.py`:
```python
(f"{value!s} is instance of {type_description} =>",)
```
Now the query says `is instance of` (three space-separated words), while examples
say `isinstanceof` (×20) or `instanceof` (×4). The query matches **neither**, and
the examples already disagree among themselves. **Fix:** query token →
`isinstanceof`; normalize the four trailing `instanceof` examples (currently
`'https://*.com' instanceof 'url' =>True`, `'€12.50' …`, `'col1,col2\n1,2' …`,
`'*@*.com' …`) to `isinstanceof`.

### D4 / D5 — `compare._CONTAINS_EXAMPLES[3]`: two examples merged + stale Symbol repr

`symai/ops/compare.py` (~line 50), a single tuple element:
```
'self-aware' in '([<class 'symai.symbol.Symbol'>(value=("['-', '- AI has become self-aware', '- Trying to figure out what it is']",))],)' =>True'Apple Inc.' in 'Microsoft is a large company that makes software ... ' =>False
```
Two defects in one string:
- **D4 (merge):** the `'self-aware' … =>True` example and the `'Apple Inc.' …
  =>False` example were concatenated (missing comma in the original list), so the
  model sees one exemplar with two `=>` and the RHS `True'Apple Inc.' in …`.
- **D5 (stale repr):** it embeds `<class 'symai.symbol.Symbol'>(value=(…))`, the
  *old* Symbol string form. Under the redesigned `Symbol`, this repr no longer
  matches reality, so it teaches an obsolete internal format.

Confirmed identical in `ContainsValue@a220d6f`. **Fix:** split into two elements
and replace the old-Symbol repr with a plain container example.

### D6 — `text._FORMAT_EXAMPLES[5]`: three examples merged into one

`symai/ops/text.py` (~lines 78–81):
```
… format 'japanese' =>すみません、皆さん。でも、今日は参加できません。text 'Sorry, everyone. …' format 'japanese romanji' =>Sumimasen, minasan. Demo, kyō wa sanka dekimasen.text 'April 1, 2020' format 'EU date' =>01.04.2020
```
Two missing commas in the original `Format` list collapsed three distinct
exemplars (Japanese / Japanese-romanji / EU-date) into one string; note the
run-on boundaries `。text` and `.text`. **Fix:** three separate tuple elements.

### D7 — `logic`: colon-delimited query vs colon-free examples

`symai/ops/reason.py` (~line 148): `f"expr :{left_value!s}: {operator}
:{right_value!s}: =>"`. Every `_LOGIC_EXAMPLES` entry is `expr <A> <op> <B> =>`
with no colons (`expr True and True =>'True'`, `expr 'All humans are mortal' and
'Socrates is a human' =>…`). The colon wrapping is faithful to the old
`LogicExpressionPreProcessor` (`expr :{a}: {operator} :{b}: =>`), which *also*
never matched the `LogicExpression` examples — so this is a pre-existing
cross-layer inconsistency, now preserved. **Fix:** drop the colons in the query.

### D8 — `reason._INTERPRET_EXAMPLES[22]`: dangling operator example

`"x ≠ 5"  not  =>x = 5.` (reason.py ~line 45). The `not` trails the operand with
doubled spaces and no second operand — mirrors the malformed old
`SimpleSymbolicExpressionPreProcessor` template `expr :{val} =: =>`. One odd row
among 39 otherwise-consistent exemplars. **Fix:** rewrite in the prefix form the
same list already uses: `not("x = 5") =>x ≠ 5.`

### D9 / D10 / D11 — cosmetic

- **D9** `_COMBINE_EXAMPLES`: `True + 0 => False` is the only element with `=> `
  spacing (all others `=>`), and its result is dubious; `'One' + 'Two' =>3`
  yields `3` rather than `Three`. Low confidence these are wrong — "logical
  addition" is deliberately fuzzy — but the spacing is a clear inconsistency.
- **D10** `_EQUALS_EXAMPLES`: includes `<=`/`>=` rows (`'eleven' <= 8 =>False`,
  `'eleven' <= 11 =>True`). The `equals` op only ever builds `==` queries, so
  these comparison operators are off-instruction. Harmless, faithful to the
  shared `FuzzyEquals` prompt.
- **D11** `modify`/`replace` queries end `'{arg}'=>` (no space) while their
  examples use ` =>`. Faithful to the old `ModifyPreProcessor`/`ReplacePreProcessor`,
  but internally inconsistent with the exemplars.

---

## What is clean and should be kept

- **Migration fidelity is perfect.** 13/13 example tuples are byte-identical to
  `prompts.py`. The odd multi-line string wrapping (looks like `pprint` output)
  is cosmetic and preserves content exactly — including escaping: raw-string
  LaTeX `r"…\sqrt[2]{\pi}…"` → `"…\\sqrt[2]{\\pi}…"` and raw regex `r"…\/\/…"` →
  `"…\\/\\/…"` both round-trip correctly (verified in `_MODIFY_EXAMPLES` and
  `_EXTRACT_EXAMPLES`).
- **Query formats that were preserved correctly and match their examples:**
  `equals` (`{a} == {b} =>`), `contains` (`{b} in {a} =>`), `interpret`
  (`{val} =>`), `extract` (`from '…' extract '…' =>`), `include`
  (`text '…' include '…' =>`), `combine` (`{a} + {b} =>`), `map`
  (`text '{val}' {instruction} =>`), `convert` (`text {val} format '{fmt}' =>` —
  exactly matches the original `TextFormatPreProcessor`).
- **`embed.py`** carries no few-shot examples (numeric ops only) — nothing to
  mangle, nothing wrong.
- **`filter`, `style`, `summarize`, `translate`, `query`, `template`** pass no
  `examples=`, so they are fidelity-clean; their instruction strings have no
  exemplars to contradict.

## Out-of-scope note (not a fidelity defect)

The old `CompareValues` prompt (semantic `<` / `>` / `>=` comparison, 46
examples) was **not** carried into any op — there is no `compare`/greater-than
op in `symai/ops/`. That is a feature-scope decision, not a migration mangling,
so it is outside this audit; flagging only so it is not mistaken for a lost
example set.
