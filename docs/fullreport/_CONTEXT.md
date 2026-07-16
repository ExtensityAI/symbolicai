# Shared Context Brief — engine-redesign audit

> **Configuration decision after this snapshot:** final configuration maps names to
> `EngineConfig`, contains no `default_language_model` or `default_embedding`, and resolves an
> omitted engine name only when that capability has exactly one configured engine.

> ## ⚠️ STATE UPDATE (Round 2+ read this first)
> During Round 1 the live sibling agent **completed the legacy cutover** — commit
> `84f703b "refactor: remove legacy runtime and symbol surfaces"` (was `09bab6a` when the
> seeds below were captured). As of the current tree, these Round-1 seeds are **RESOLVED**
> (verify, then treat as done, not open):
> - `symai/prompts.py` and `symai/backend/` are **deleted**; `jinja2`/`python-box`/`tomllib`
>   have 0 code refs; ops now own inline example tuples.
> - Root `symai/__init__.py` is now **0 bytes** (empty root, matching the design). No `__all__`.
> - `current_runtime()`/`_CURRENT_RUNTIME`/`NoActiveRuntimeError` and
>   `Function.static_context`/`dynamic_context` are **removed** (0 prod refs).
> - The two cutover test suites went from 6 failing → **66/66 green**; full suite **620 passed**.
>
> **The SURVIVING opportunities (what Round 2/3 should focus on)** — all re-verify against
> current code: provider-layer duplication (error-mapping ×4, byte-identical
> transport/headers, `_client.py` ~83%, cerebras/deepseek `chat_completions.py` ~55%,
> `settings.py` ×4, ops `_symbol_value`/`_require_text`); `LanguageModelSpec` dead
> fields + orphan `MessageRole`/`ContentType`/`ResponseFormatType` enums (CON-01);
> `JsonObject`/`JsonArray`/`JsonEntry` AST round-trip (CX-01); dead N-output machinery
> (CON-03); logprobs write-only hole (CON-02); decoding (`PydanticDecoder` redundant,
> `ConstructorDecoder` container branch unused, `_normalize_text` single-quote stripping);
> Runtime (over-applied lock, divergent `Runtime` vs `RuntimeConfig` validation,
> per-capability-only name uniqueness); provider adapters (data-driven capability gate vs
> hardcoded checks, over-strict usage-consistency rejects); naming (`operations.py` vs
> `ops/` collision, two `loading.py`, no-op `cast("ImplementationId", ...)`); docs & packaging
> **still fully broken** (README + `docs/source/*` import a dead API, no migration guide,
> `CMDS.md` dead, version still `1.18.0`, `numpy<=2.1.3` cap).
>
> Round-1 report files (`r1-*.md`) are your inputs — read the ones relevant to your goal.

**Every audit agent MUST read this file first.** It is the shared map so you don't
re-derive the architecture. Do not treat the "known signals" below as the answer —
they are seeds; find more, verify these, and go deeper.

---

## What you are auditing

`symbolicai` on branch `refactor/engine-redesign`, worktree at:
`/Users/adrian/Desktop/Projects/symbolicai/.worktrees/engine-redesign`

**AUDIT ONLY THIS WORKTREE. Do not look at or compare against any other worktree
or the main checkout.** All paths below are relative to the worktree root.

This is a **greenfield, pre-release, breaking major rewrite**. It replaces an old
implicit global-engine framework with three explicit layers. Backward compatibility,
minimal diffs, and "smallest change" are NOT goals here — optimize for the best
end-state. Dropping legacy is expected; the question is whether the *current* design
is as simple, elegant, de-duplicated, and cleanly-bounded as it can be **while keeping
the intended features** (only minimal/irrelevant feature loss is acceptable).

Snapshot commit: `09bab6a`. **The code is a MOVING TARGET** — another agent is
actively editing this worktree. Therefore:
- Anchor findings by **symbol name + a short quoted snippet**, not just `file:line`.
  Treat line numbers as approximate.
- If something you cite looks already-changed, say so; don't assume staleness is a bug.

## Layer architecture (intended)

```
Symbol[T]  (value DSL)  ──►  ops.*  ──►  Function  ──►  Runtime  ──►  engines  ──►  clients
                              │                                       (adapters)   (raw HTTP)
                              └──►  decoding (Decoder[T], decode_output)
```

Intended dependency direction: `Symbol ← ops.* → Function → Runtime → engines → clients`,
plus `ops.* → decoding`. Function/Runtime/decoders must NOT import or return Symbol;
`ops.*` is the only layer that wraps decoded values back into Symbols.

## File inventory (symai/, ~7.2k LOC, 62 files)

Core value/exec layer:
- `symai/symbol.py` (184) — `Symbol[T]`, shallow-immutable value DSL, ~40 operator dunders.
- `symai/function.py` (110) — `Function`, builds+executes one language request.
- `symai/decoding.py` (161) — `Decoder` protocol, `decode_output`, Text/Constructor/TypeAdapter/Pydantic decoders.
- `symai/operations.py` (107) — request builders (`language_request`, `image_request`, `embedding_request`, `data_uri`, `parse_embedding_response`).
- `symai/prompts.py` (1042) — legacy `Prompt`/`PromptRegistry` + many few-shot example classes (jinja2, tomllib, python-box).
- `symai/loading.py` (72) — builtin loader registry + public `load_runtime`.
- `symai/__init__.py` (142) — package-root re-exports (`__all__`).
- `symai/backend/__init__.py` (0) — empty vestige.

ops (ergonomic semantic operations):
- `ops/text.py` (308), `ops/embed.py` (276, numpy math), `ops/reason.py` (90),
  `ops/compare.py` (93), `ops/rank.py` (39), `ops/primitives.py` (16, shared `_execute_language`).

runtime (lifecycle + normalized contracts):
- `runtime/runtime.py` (305) — `Runtime` lifecycle, selection, thread-ownership, `current_runtime()`+ContextVar.
- `runtime/models.py` (446) — normalized Pydantic contracts (messages, requests, responses, specs, JsonValue AST).
- `runtime/config.py` (100) — `RuntimeConfig`, `EngineSpec`, `ImplementationId`.
- `runtime/loading.py` (112) — generic `load_runtime` with preflight + failure cleanup.
- `runtime/engines.py` (20) — `LanguageModelEngine`/`EmbeddingEngine` protocols.
- `runtime/errors.py` (113) — runtime error hierarchy + `ErrorMetadata`.

providers (adapters + hand-written clients), per provider `client/` + `engines/` + `loading.py` + `settings.py`:
- `providers/_client/` — shared `models.py`, `errors.py`, `headers.py`.
- `providers/openai/` — `engines/responses.py` (451), `engines/embedding.py` (206), `client/` (responses 460, embeddings, transport, _client 186, headers, errors).
- `providers/cerebras/` — `engines/chat_completions.py` (459), `client/chat.py` (251), `client/_client.py` (119), transport/headers/errors.
- `providers/deepseek/` — `engines/chat_completions.py` (435), `client/chat.py` (153), `client/_client.py` (113), transport/headers/errors.

tests/ (~8.5k LOC): runtime/, providers/**, plus top-level test_symbol, test_decoding,
test_operations, test_import_boundaries, test_public_cutover, test_symbol_runtime_cutover,
test_components. `tests/typecheck/` holds static-typing assertion files.

## Design & prior-audit documents

- `audit/SYMBOL_REDESIGN.md` — **approved** Symbol/Function/decoding/ops design (the spec).
- `audit/FIXPLAN.md` — implementation-order + release-gate plan.
- `audit/FINDINGS.md` — evidence register (BUG-/API-/CLI-/SOC-/CX-/CON-/FP-/EXT- IDs).
- `audit/README.md` — audit summary + ratified direction.

**IMPORTANT:** `audit/FINDINGS.md` and `FIXPLAN.md` were written against an **earlier**
commit (`refactor/cleanup` @ `a220d6f`), BEFORE the engine-redesign commits landed. So
many findings may already be **fixed**. When you cite a prior finding, verify against the
current code and label it: FIXED / PARTIALLY-FIXED / STILL-OPEN / SUPERSEDED. Do not
assume the prior audit reflects current state.

## Known drift/smell signals (seeds — verify and expand, do NOT stop here)

> **Double-staleness warning:** these seeds were captured at a snapshot while the tree
> was being actively edited. Any one of them may **already be fixed or changed** by the
> time you read the code. Re-verify every seed against the LIVE file before reporting it.
> If a seed is already resolved, say so as a positive ("already addressed") rather than
> reporting a stale problem. Never report a signal you did not re-confirm in the code now.

1. `SYMBOL_REDESIGN.md` says `current_runtime()`/ambient ContextVar discovery is **removed**,
   but `runtime/runtime.py` still defines `_CURRENT_RUNTIME`, `current_runtime()`, sets the
   ContextVar in `__enter__`, and exports `current_runtime` + `NoActiveRuntimeError`.
2. Design says Function has **no** static/dynamic context, but `function.py` still has
   `static_context`/`dynamic_context` fields and `_system_prompt()` composes them.
3. `Symbol` is **not** re-exported in `symai/__init__.__all__` (Function, decoders, runtime
   models are). ops namespaces aren't either. Design says canonical imports come from owning
   modules and root "is not a compatibility facade" — yet root re-exports ~90 names.
4. `prompts.py` (1042 LOC) retains `PromptRegistry`, jinja2, tomllib, python-box, and many
   Prompt example classes; ops only uses a handful (Modify, MapExpression, Format, ReplaceText,
   IncludeText, CombineText, ExtractPattern, CompareValues, RankList, ContainsValue, IsInstanceOf,
   etc.). jinja2/box/tomllib appear used ONLY by prompts.py → possible dead deps.
5. Three language engine adapters (`responses.py`, cerebras/deepseek `chat_completions.py`) and
   three client packages look heavily duplicated (errors.py, transport, headers, _client).
6. `runtime/models.py` has a bespoke `JsonObject/JsonArray/JsonEntry` Pydantic "AST" that is
   converted straight back to builtin dict/list; and a `LanguageModelSpec` capability matrix whose
   fields may be populated-but-unread (enforcement is parallel hardcoded checks in each engine).
7. Duplicated helpers across ops modules (`_symbol_value`, `_require_text`, `_string_tuple`).
8. `loading.py` (public) vs `runtime/loading.py` (generic) split; `backend/` empty.

## Report conventions (all agents follow)

- Write exactly ONE markdown file, at the path assigned in your prompt, under
  `docs/fullreport/`. Do NOT edit code, tests, config, or any other file. Read-only otherwise.
- Start with a 5-line **Executive summary** (top findings + overall read).
- Then a findings table, then detailed findings. For each finding include:
  - **What** (the smell/opportunity), **Where** (symbol + snippet), **Why it matters**.
  - **Proposed change** (concrete: before → after sketch is welcome, but do NOT apply it).
  - **Feature impact**: `keeps-all` / `drops-minimal` / `drops-real-feature` (name the feature).
  - **Confidence** (high/med/low) and **Impact** (high/med/low) and **Effort** (S/M/L).
- Also call out what is **already good and should be kept** (not just problems).
- Prefer a few load-bearing, verified findings over a long shallow list. Quote code.
- You may run read-only shell/tools (grep, cat, ruff, pyright, pytest --collect-only) to verify.
  Do not mutate the tree.
