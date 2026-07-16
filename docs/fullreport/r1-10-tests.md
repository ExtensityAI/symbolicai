# r1-10 — Tests & Verifiability Lens

**Scope:** quality / simplicity / coverage of the `tests/**` tree against
`SYMBOL_REDESIGN.md §11` acceptance criteria. Read-only audit.
**Snapshot:** verified live during audit; the worktree is a **moving target** and I
watched a test file change mid-read (see F1). Anchors are symbol + quoted snippet;
line numbers approximate. 620 tests collect cleanly (`uv run pytest --collect-only`).

---

## Executive summary

1. **The test suite is excellent and, for most §11 criteria, near-exemplary** —
   Symbol (`tests/test_symbol.py`), decoding (`tests/test_decoding.py`), Runtime
   lifecycle (`tests/runtime/test_runtime.py`), and provider adapters all have
   dense, adversarial, well-isolated coverage with realistic bidirectional wire
   fixtures. This is a strength to preserve.
2. **The suite currently encodes the TARGET state and is RED against lagging
   production.** The removal/cutover tests assert `static_context`,
   `current_runtime`, `NoActiveRuntimeError`, and `prompts.py` are gone, but
   production still ships all of them. This is expected mid-migration (tests lead),
   **not** a case of tests pinning drift — I observed `test_components.py` get
   rewritten live to *forbid* the drift it previously pinned.
3. **The one genuine test-tree defect is the typecheck stage** (`tests/typecheck/
   function_decoding.py`): it is wired to **no runner**, there is **no pyright
   config/CI**, and its negative assertion's suppression code is **wrong**
   (`reportGeneralTypeIssues` vs. actual `reportInvalidTypeArguments`), so the file
   does not even pass pyright clean. The §11 "static checking proves `Decoder[T]`
   flows to `decode_output(...) -> T`" criterion has no automated enforcement.
4. **Three §11 criteria have no direct guard:** no-tool-calling-type, "documentation
   contains no examples using removed concepts" (only production `.py` AST is
   scanned, never docs/README/examples), and behavioral truth of `ops.*` prompts.
5. **`ops.*` is broad but shallow, and provider test scaffolding is duplicated.**
   Every remote op has a request-shape case, but assertions are wiring-level and
   partly self-referential (pull production's own `_EXAMPLES`); real semantic proof
   needs bounded live canaries.

---

## Findings table

| ID | Finding | Confidence | Impact | Effort |
|----|---------|-----------|--------|--------|
| F1 | Removal/cutover tests are RED against lagging production (tests lead correctly) | high | high | — (prod work) |
| F2 | Typecheck stage orphaned: no runner, no pyright config, mismatched ignore code | high | high | S |
| F3 | "Docs contain no removed concepts" (§11 Removal) has no test — only prod `.py` AST scanned | high | med | S |
| F4 | "No tool-calling request/output type" (§11 Ops) has no negative guard | high | low | S |
| F5 | `ops.*` request assertions partly self-confirming; depth << provider tests | med | med | M |
| F6 | Provider test scaffolding (`_client`/`_chat_json`/handlers) duplicated, no conftest | high | low | M |
| F7 | `tests/README.md` + `pyproject` source-exclude are stale (`pytest.ini`, `test_imports.py`) | high | low | S |
| G  | Strengths to keep (Symbol/decoding/runtime/provider fixtures) | high | — | — |

---

## Coverage matrix (§11 acceptance criterion → covered? → where)

### Symbol
| Criterion | Covered | Where (test name) |
|---|---|---|
| state cannot be reassigned | ✅ | `test_symbol_wrapper_state_is_immutable_and_cannot_be_extended` |
| Symbol is unhashable | ✅ | `test_symbol_is_unhashable_even_when_its_value_is_hashable` |
| no Runtime/client/provider import below Symbol | ✅ | `test_symbol_module_has_no_execution_or_io_dependencies` (AST import scan) |
| no context/graph/persistence/embedding/semantic state | ✅ | `test_symbol_has_no_forbidden_god_object_surface` |
| native operators never perform I/O | ✅ | `test_native_symbol_operations_do_not_perform_io` (monkeypatches `open`/`socket`/`urlopen`) |
| operator result types match §4.3 table | ✅ (full) | `test_binary_arithmetic_returns_new_symbols`, `test_reflected_arithmetic_*`, `test_bitwise_operations_and_reflections_*`, `test_unary_value_operations_*`, `test_indexing_and_slicing_return_new_symbols`, `test_iteration_yields_symbols_without_copying_elements`, `test_len_truth_and_explicit_casts_return_native_values`, `test_equality_is_symmetric_*`, `test_ordering_returns_native_booleans_*`, `test_containment_uses_the_held_value_as_container`, `test_matrix_multiplication_and_its_reflection_return_symbols`, `test_power_supports_the_native_three_argument_form` |
| original Python exceptions propagate | ✅ | `test_native_exceptions_propagate_unchanged` (asserts `.args` identical to native) |

### Function and decoding
| Criterion | Covered | Where |
|---|---|---|
| Function calls return `LanguageModelResponse` only | ✅ | `test_call_returns_exact_normalized_response_and_forwards_engine`; `test_function_has_one_non_generic_execution_surface` (asserts `__call__` return is `LanguageModelResponse`) |
| request preview is `request()` and performs no I/O | ✅ | `test_request_builds_normalized_request_without_execution` (asserts `engine.requests == []`) |
| response metadata needs no mode flag | ✅ (indirect) | `test_function_has_one_non_generic_execution_surface` (no `return_metadata`/`preview` params); `test_call_...` (`actual.metadata is METADATA` always present) |
| scalar decoding | ✅ | `test_text_and_constructor_decoders_normalize_scalar_text` |
| boolean decoding | ✅ | `test_constructor_decoder_accepts_explicit_boolean_forms`, `test_constructor_decoder_rejects_unknown_boolean_text` |
| Pydantic model decoding | ✅ | `test_pydantic_decoder_validates_a_model` |
| nested container decoding | ✅ | `test_type_adapter_decoder_preserves_nested_parameterized_types` |
| default decoding | ✅ | `test_explicit_default_catches_only_decoder_failure_and_is_limited`, `test_explicit_decode_error_uses_default_and_propagates_without_one`, `test_default_does_not_hide_unexpected_decoder_exception`, `test_default_does_not_catch_output_selection_or_limiting_errors` |
| limit decoding | ✅ | `test_collection_limiting_preserves_sequence_and_mapping_order`, `test_collection_limiting_leaves_unordered_collections_unchanged` (sets pass through) |
| output-index decoding | ✅ | `test_output_selection_uses_normalized_index_not_tuple_position` (normalized index, `IndexError` when absent) |
| **static proof `Decoder[T]` → `decode_output(...) -> T`** | ⚠️ **PARTIAL** | `tests/typecheck/function_decoding.py::prove_decoder_result_inference` (`assert_type`) — proofs hold under pyright, but **no runner/CI**; see F2 |
| sequential multi-execution stable ordering + honest naming | ✅ | `test_execute_many_is_sequential_and_preserves_nested_input_order` |

### Operations
| Criterion | Covered | Where |
|---|---|---|
| each I/O op takes Runtime + Symbol explicitly | ✅ | `test_remote_signatures_have_only_explicit_engine_selection` (param[0]=="runtime"); `test_raw_primary_operands_are_rejected` |
| deterministic local ops take Symbols, no unused Runtime | ✅ | `test_remote_signatures_...` (template/similarity/distance/mmd/kernel have no runtime/engine); `test_template_is_local_fresh_and_non_mutating` |
| each op returns new Symbol, no input mutation | ✅ | `test_language_operation_contract` (`result is not source`, `source.value` unchanged); `test_similarity_metrics` |
| engine names forwarded unchanged by I/O ops | ✅ | `test_language_operation_contract` (`selected.requests == [case.request]`, `other.requests == []`) |
| no provider/model option at op layer | ✅ | `test_remote_signatures_...` (`"provider"`/`"model"` not in params, no `VAR_KEYWORD`) |
| **no tool-calling request or output type** | ❌ **GAP** | no negative guard; satisfied only by absence in `symai/runtime/models.py` (F4) |

### Removal
| Criterion | Covered | Where |
|---|---|---|
| Expr/Result/graph/contexts/persistence/flags/mixins absent from **production** | ⚠️ **RED** | `test_public_cutover.py::test_production_ast_has_no_legacy_graph_references` (asserts `_production_ast_violations(PACKAGE) == []`) — currently fails (F1) |
| …absent from **tests** | ✅ | `test_old_mixin_context_and_symbol_surfaces_are_absent`, `test_symbol_has_no_forbidden_god_object_surface` |
| no deprecated aliases / forwarding shims | ⚠️ **RED** | `test_old_root_names_are_absent_after_canonical_imports`, `test_runtime_module_exposes_no_ambient_registry_or_provider_clients`, `runtime/test_runtime.py::test_runtime_has_no_ambient_registry_slot_or_module_state`, `runtime/test_errors.py` (`not hasattr(errors_module, "NoActiveRuntimeError")`) — assert target state, fail now (F1) |
| **docs contain no examples using removed concepts** | ❌ **GAP** | no test scans `docs/`, `README.md`, or `examples/` — `_production_ast_violations` walks `symai/` `.py` only (F3) |

---

## Detailed findings

### F1 — Suite encodes the target state and is currently RED against lagging production

**What.** The cutover/removal tests are correct — they encode the approved
`SYMBOL_REDESIGN.md` end-state — but production has not caught up, so the suite
does not currently go green. I did **not** run the suite (per rules); this is
derived from reading the assertions against verified production state.

**Where + evidence (verified live, production side still drifted):**
- `symai/function.py` — `Function.__init__` still declares `static_context: str = ""`
  and `dynamic_context: str = ""`, with `_system_prompt()` composing
  `f"<STATIC_CONTEXT/>\n{self.static_context}"`. This violates:
  - `test_components.py::test_function_has_one_non_generic_execution_surface`, which now
    asserts `{"static_context", "dynamic_context", ...}.intersection(init_parameters)`
    is empty.
  - `test_public_cutover.py::test_production_ast_has_no_legacy_graph_references` — I
    replicated its `_production_ast_violations` walk (`FORBIDDEN_IDENTIFIERS` includes
    `static_context`, `current_runtime`, `_CURRENT_RUNTIME`, `NoActiveRuntimeError`,
    `Prompt`, `PromptRegistry`, `clear`, `save`) against the live package: **61 hits**,
    including `function.py:18/35/80/81 static_context`, `runtime/runtime.py current_runtime/_CURRENT_RUNTIME`,
    `errors.py NoActiveRuntimeError`, and all of `prompts.py`.
- `symai/runtime/runtime.py` — still defines `_CURRENT_RUNTIME`, `current_runtime()`,
  sets the ContextVar in `__enter__`/`__exit__`. Violates
  `runtime/test_runtime.py::test_runtime_has_no_ambient_registry_slot_or_module_state`
  and `test_public_cutover.py::test_runtime_module_exposes_no_ambient_registry_or_provider_clients`.
- `symai/prompts.py` — **still exists** (71 KB, `Prompt`/`PromptRegistry`). Violates
  `test_public_cutover.py::test_deleted_modules_have_no_import_spec`
  (`find_spec("symai.prompts") is None`) and `test_deleted_production_tree_and_adapter_inventory`.

**Why it matters.** This is the healthy TDD direction: the tests are the spec and are
red on purpose. The audit-relevant point is (a) the suite is **not green today**, so
"all tests pass" cannot be a release gate claim yet, and (b) the tests themselves are
**correct and should NOT be weakened** to match production — production must move.

**Drift-lens note (directly answering the prompt).** I was asked whether tests *pin*
the known drift (`current_runtime`, `static_context`/`dynamic_context`) and would need
removing when those go. **Answer: no — the opposite.** At my first read,
`test_components.py::test_request_builds_normalized_request_without_execution` still
constructed `Function(static_context="Use arithmetic.", dynamic_context="...")` and
asserted `<STATIC_CONTEXT/>`/`<DYNAMIC_CONTEXT/>` in the system prompt. Minutes later it
had been **rewritten live** to drop those args and expect `"Answer precisely.\n2 + 2 => 4"`,
and `test_function_has_one_non_generic_execution_surface` had `static_context`/
`dynamic_context` added to its forbidden-param set. So the test tree is actively being
de-drifted to *forbid* these; no drift-pinning test remains that needs later removal.

**Proposed change.** None to tests. This is a production-lag observation; production
must delete `static_context`/`dynamic_context`, `current_runtime`/`_CURRENT_RUNTIME`/
`NoActiveRuntimeError`, and `prompts.py` to turn the suite green.

**Feature impact:** keeps-all. **Confidence:** high. **Impact:** high (gate).

---

### F2 — Typecheck stage is orphaned and does not pass pyright clean

**What.** `tests/typecheck/function_decoding.py` is the *only* enforcement of the §11
criterion "static checking proves `Decoder[T]` flows to `decode_output(...) -> T`".
It is a static-assertion file (`assert_type(...)`, doesn't match pytest's `test_*.py`,
so pytest ignores it — confirmed it is collected only as inert). Three problems:

1. **No runner.** There is no `pyrightconfig.json`, no `[tool.pyright]` in
   `pyproject.toml`, `pyright` is not in the `dev` dependency group, and there is no
   `.github/workflows/`. Nothing runs pyright over `tests/typecheck/`, so the
   `assert_type` proofs are enforced only if a human remembers to run it manually.

2. **Mismatched suppression.** Line 35:
   ```python
   Function[int]("Answer.")  # pyright: ignore[reportGeneralTypeIssues]
   ```
   This exists to prove `Function` is non-generic (subscripting must be an error).
   But `uv run pyright tests/typecheck/function_decoding.py` reports the diagnostic as
   **`reportInvalidTypeArguments`**, not `reportGeneralTypeIssues` — so the ignore does
   not match, and pyright exits with **`1 error`**. The file is not pyright-clean. (The
   *positive* `assert_type` proofs on lines 20-30 all pass — the `Decoder[T] -> T`
   inference is genuinely correct; only the negative assertion's suppression is wrong.)

**Why it matters.** A criterion that exists to be *proven by the type checker* is neither
run nor clean. Given the recent commit `test(runtime): make full type check clean`, the
intent is clearly a clean pyright pass, but there is no gate holding it.

**Proposed change.**
- Fix line 35 to `# pyright: ignore[reportInvalidTypeArguments]`.
- Add `[tool.pyright]` (or `pyrightconfig.json`) that `include`s `tests/typecheck`,
  sets `reportUnnecessaryTypeIgnoreComment = "error"` (so a future mismatch fails), and
  add `pyright` to the `dev` group + a CI/nox step asserting `0 errors`.

**Feature impact:** keeps-all. **Confidence:** high. **Impact:** high. **Effort:** S.

---

### F3 — "Documentation contains no examples using removed concepts" is unguarded

**What.** §11 Removal requires that *documentation* contains no removed-concept
examples. The only scanner, `test_public_cutover.py::_production_ast_violations`,
walks `PACKAGE = ROOT / "symai"` `.py` files exclusively. `docs/`, top-level
`README.md`, and `examples/` are never checked for legacy identifiers or import
strings.

**Why it matters.** README/examples are the highest-visibility surface for a breaking
major cutover; a lingering `Symbol().summarize(...)` or `current_runtime()` snippet in
docs would ship silently.

**Proposed change.** Add a test that greps `README.md`, `docs/**/*.md`, and
`examples/**/*.py` for the `FORBIDDEN_IDENTIFIERS` set (and legacy import prefixes),
reusing the existing constant. Markdown fenced Python blocks can be extracted and
`ast.parse`d for stronger checking.

**Feature impact:** keeps-all. **Confidence:** high. **Impact:** med. **Effort:** S.

---

### F4 — No negative guard for "no tool-calling request or output type exists"

**What.** §11 Operations requires that no tool-calling request or output type exists.
`grep` for `tool_call`/`ToolCall`/`tools=` in `symai/runtime/models.py` returns nothing
(good — production is clean), but there is **no test** asserting this absence, unlike
the many `not hasattr`/forbidden-name guards used elsewhere.

**Why it matters.** Removals that are guarded (Expression, mixins, ambient contexts)
cannot silently regress; an accidental re-introduction of a `tools`/`ToolCall` field on
a request/response model would pass unnoticed.

**Proposed change.** Add to `tests/runtime/test_models.py` an assertion that no
public model in `symai.runtime.models` exposes a tool/function-calling field
(scan model `model_fields` for names like `tools`, `tool_calls`, `function_call`).

**Feature impact:** keeps-all. **Confidence:** high. **Impact:** low. **Effort:** S.

---

### F5 — `ops.*` coverage is broad but shallow and partly self-confirming

**What.** All `ops.*` behavior lives in one file, `tests/test_symbol_runtime_cutover.py`.
Its centerpiece, `test_language_operation_contract`, is parametrized over **19
`REMOTE_CASES`** (all of `text.*`, `reason.*`, `compare.*`, `rank.rank`, `embed.embed`)
— genuinely good breadth. But the assertions are wiring-level:

- `assert selected.requests == [case.request]` where each `case.request` is built with
  the **production module's own** private constants, e.g.
  `language_request("Modify the text...", "...", examples=text._MODIFY_EXAMPLES)`. The
  **instruction strings are independently pinned** in the test (a real assertion), but
  the few-shot **example content is self-referential** — if `text._MODIFY_EXAMPLES`
  were wrong, the test still passes.
- Decode assertions only distinguish `TextDecoder` vs `ConstructorDecoder(bool)`; the
  mock returns `"  decoded  "` and the test asserts `result.value == "decoded"`. This
  proves whitespace-strip + wrap, not that any op yields a *useful* value, and no op is
  exercised through `PydanticDecoder`/`TypeAdapterDecoder` at the ops layer.

Compare this to provider tests (`test_responses.py`, `test_chat_completions.py`, ~600
functions total) which exhaustively assert status→error mapping, malformed payloads,
usage validation, reasoning items, and multi-output ordering. The user-facing `ops`
layer — the product's primary surface — gets the thinnest depth.

**Why it matters.** Mocks can prove the op *wires the right request and forwards the
engine*, but cannot prove the prompt/few-shots actually elicit correct behavior from a
provider (e.g. `compare.equals` returning a real boolean, `rank.rank` returning an
ordered list, `text.convert` producing valid JSON). That is the exact class of thing
that silently rots.

**Proposed change.**
- Deterministic growth: add per-op decode-path tests where the op's decoder is
  non-trivial (e.g. does `rank.rank`/`text.convert` decode structure, or return raw
  text?), and assert the *example tuples'* shape independently rather than importing
  the production constant into the expected request.
- Live canaries (the only real proof): a small **bounded, opt-in** live suite (one
  case per op per provider, gated on API keys / a marker) asserting the response
  *decodes to the documented type and is non-empty* — not exact text.

**Feature impact:** keeps-all. **Confidence:** med. **Impact:** med. **Effort:** M.

---

### F6 — Provider test scaffolding is duplicated across providers with no `conftest`

**What.** There is no `conftest.py` anywhere under `tests/`. Each provider engine/client
test redefines its own `_client(handler)` (httpx `MockTransport` wrapper), payload
factory (`_response_json` / `_chat_json` / `_completion_json`), and many local
`def handler(request)` closures. `deepseek/engines/test_chat_completions.py::_chat_json`
and `cerebras/engines/test_chat_completions.py::_chat_json` have **identical signatures**
(`*, choices, usage, model=...`) differing only in the default model string — both are
OpenAI-compatible chat-completions shapes.

**Why it matters.** The two chat-completions providers will drift in test coverage as
one gets an edge-case the other doesn't; the shared mock-transport + payload-factory
boilerplate is copy-paste maintenance.

**Proposed change.** Add `tests/providers/conftest.py` (or a `_client` test helper next
to `providers/_client/`) exposing a `mock_client(handler)` fixture and a
`chat_completions_payload(...)` / `responses_payload(...)` factory. Keep provider-specific
assertions in place; share only the transport + envelope construction.

**Feature impact:** keeps-all. **Confidence:** high. **Impact:** low. **Effort:** M.

---

### F7 — Stale test-tree metadata

**What.** `tests/README.md` documents a workflow that no longer exists: it says the
default run uses `--ignore=tests/test_imports` and to run `tests/test_imports.py`
separately — **no `test_imports.py` exists** in the tree. `pyproject.toml`
`source-exclude` also lists `/pytest.ini` and the README implies flags not in the real
`pytest.ini` (`addopts = -v -rpP --import-mode=importlib`, `filterwarnings = error`).

**Why it matters.** Minor, but it is documentation drift *inside the test tree* — the
same class of issue §11 flags for production docs.

**Proposed change.** Rewrite `tests/README.md` to describe the real invocation
(`uv run pytest`), the `--import-mode=importlib` rationale (same-named provider test
modules), and the `filterwarnings = error` policy.

**Feature impact:** keeps-all. **Confidence:** high. **Impact:** low. **Effort:** S.

---

## G — What is already good and must be kept

- **`tests/test_symbol.py` is exemplary.** It nails the full §4.3 result-type table
  (direct + reflected arithmetic, bitwise, unary, matmul incl. reflected, 3-arg `pow`
  with a 3.14 version branch), asserts native exception `.args` are *byte-identical* to
  Python's (`test_native_exceptions_propagate_unchanged`), proves no I/O two ways (AST
  import scan **and** monkeypatched `open`/`socket`/`urlopen`), and proves immutability,
  unhashability, no `__getattr__` forwarding, and a forbidden god-object surface.
- **`tests/test_decoding.py` independently covers every decode dimension** and, notably,
  the *negative* scoping of `default`: it does **not** hide programming errors
  (`RuntimeError`), transport, or selection/limit errors — `test_default_does_not_*`
  are the kind of tests that catch real bugs. Sets correctly pass through limiting.
- **`tests/runtime/test_runtime.py` (26 tests)** covers lifecycle, thread ownership
  (foreign-thread execute/close rejected *before* engine touch), reverse-order close,
  idempotent close, `BaseExceptionGroup` cleanup grouping, and body-exception-primary
  semantics. `tests/runtime/test_models.py` (23) covers the `JsonObject` AST freezing,
  discriminated unions, frozen/strict/extra-forbid, and bounds.
- **Provider fixtures are realistic, not code-derived.** `test_responses.py::_response_json`
  builds a full OpenAI Responses payload (`output_text`, `reasoning`/`summary_text`,
  `usage.input_tokens_details.cached_tokens`, `output_tokens_details.reasoning_tokens`)
  and the handler **captures the outbound body** to assert request serialization — the
  mock encodes the provider contract in **both directions**, which is the antidote to
  self-confirming fixtures. Error-status→runtime-error mapping is covered for all three
  providers plus `runtime/test_errors.py`.
- **Boundary tests enforce the intended layering.** `test_import_boundaries.py` uses a
  subprocess to prove `loading` modules don't eagerly import heavy provider
  client/engine modules; `test_public_facades.py::test_provider_client_packages_do_not_
  import_symbolic_runtime_layers` proves clients never import `symai.runtime`/`function`/
  `symbol`. `test_public_cutover.py::test_import_symai_is_subprocess_isolated_and_inert`
  proves `import symai` mutates no files, env, or logging config and exposes no public
  names.
- **Hygiene:** `pytest.ini` sets `filterwarnings = error` and `--import-mode=importlib`;
  620 tests collect in ~0.13s with no import errors.

---

## Where deterministic coverage should grow vs. where live canaries are the only proof

**Grow deterministically (cheap, high value):**
- Wire `tests/typecheck/` into pyright CI and fix the mismatched ignore (F2).
- Add the docs/examples removed-concept scanner (F3) and the no-tool-calling guard (F4).
- Add per-op decode-path assertions and de-self-reference the `ops` example checks (F5).

**Live canaries are the only real proof (bounded, opt-in):**
- Whether each `ops.*` instruction + few-shot prompt actually elicits a
  provider-parseable answer of the documented type: `text.summarize`/`translate`/
  `convert` (valid JSON), `compare.equals`/`contains`/`is_instance_of` (real boolean),
  `rank.rank` (ordered list), `reason.query`/`interpret`/`logic`. Mocks structurally
  cannot cover this; one gated case per op per provider is the smallest honest proof.
