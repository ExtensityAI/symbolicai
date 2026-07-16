# r2-d2 — Boundary & Coupling: the ideal seam map for the provider layer + module placement

**Lens (cross-cutting synthesis):** resolve the central tension — *consolidate the
provider-layer duplication WITHOUT coupling unrelated provider schemas.* Deliver a
concrete target tree, a move table with per-move coupling-risk + hook points, the explicit
line not to cross, and the non-provider seam fixes (`operations.py` vs `ops/`, the two
`loading.py`, client↔engine).

**Scope:** `symai/**` in the `engine-redesign` worktree, current tree (post-cutover:
root `__init__.py` is 0 bytes, `prompts.py`/`backend/` gone — verified). Read-only;
anchored by symbol + snippet, re-verified against live code.

---

## Executive summary

1. **The tension resolves cleanly along one axis the current tree does not yet express:
   a shared *client* tier (`_client/`, knows only pydantic/httpx/stdlib) versus a missing
   shared *engine* tier (`_engine/`, may know `runtime`).** The single most load-bearing
   placement decision in this whole audit: the error→runtime mapper, the `BaseHttpEngine`
   lifecycle, and the capability gate **must live in a new `providers/_engine/`**, *not* in
   `_client/` — because they translate into `symai.runtime.errors`/`models`, and `_client/`
   is the one layer that must never import `runtime`. R1-02's D1 sketch offered
   "`_client/` **or** runtime" as the home; that ambiguity is the trap — `_client/` is wrong.
2. **Three shared tiers, three coupling regimes.** (a) `_client/` base — transport envelope,
   header extraction, `BaseClient` shell, error `__init__`s: **coupling risk none/low**,
   pure infra. (b) `_engine/` base — lifecycle, error-mapper, capability gate: **low**,
   catches shared base classes and reads the matrix generically. (c) a *chat-completions-only*
   adapter shared by **cerebras + deepseek exclusively**: **medium — this is the one place
   to stop.**
3. **The line not to cross is sharp and short:** OpenAI **Responses ≠ chat-completions**
   (status-based finish, output-item walking, "reasoning ⇒ one assistant message" — verified
   in `responses.py`); the per-provider **`MODEL_SPECS` / `_normalized_model_spec` capability
   matrices** and the wire schemas (`chat.py`/`responses.py`/`embeddings.py`) and the
   `ReasoningEffort` enums stay strictly per-provider. Consolidating any of these relocates
   the divergence into a shared file without removing it — false DRY.
4. **The capability matrix is authoritative-in-name-only and re-encoded by hand 47×**
   (`_unsupported(` count: openai 18, cerebras 10, deepseek 19). A data-driven gate that
   *reads* each provider's own matrix — while each provider still *builds* its own matrix and
   keeps an idiosyncratic-rules hook — makes the matrix real without coupling the catalogs.
5. **Non-provider seams:** move `operations.py` (request builders) → **`runtime/requests.py`**
   (co-locate with the `*Request` models it builds; kills the `operations`/`ops` collision);
   rename the generic `runtime/loading.py::load_runtime` → **`compose_runtime`** and the public
   `symai/loading.py` → **`symai/registry.py`** (kills both the file-name and function-name
   collisions). The **client↔engine boundary holds today** (verified: no `client/` imports
   `runtime`; engines are the sole crossing; no cross-provider imports) — the `_engine/` tier
   must be built to preserve it.

**Net removable:** ~**480–620 lines** of provider-layer duplication — ~360–480 at none/low
coupling risk (`_client` base + `_engine` lifecycle/mapper/gate + settings + ops helpers),
plus ~120 at medium risk between the two chat-completions engines only. The layering shape
is right; the two shared tiers are under-populated and one of them (`_engine/`) does not yet
exist.

---

## The central insight: two shared tiers, not one

The current tree has exactly one shared home, `providers/_client/`, and it is deliberately
**runtime-blind** — verified: `grep "from symai.runtime" providers/*/client providers/_client`
returns nothing; `_client/errors.py` defines its own `ClientError → Transport/Response/API →
Auth/RateLimit` hierarchy, distinct from `runtime/errors.py`. That blindness is a *feature*
(the memory's "client = faithful API binding that never knows symai" invariant) and must be
preserved.

But most of the duplicated engine code — the error ladder, `_error_metadata`, the gate —
**translates `_client.errors.*` into `runtime.errors.*`**. It cannot go in `_client/` without
breaking the invariant. So the correct target has **two** shared tiers stacked by what they
are allowed to import:

```
                        may import          duplication it absorbs
 providers/_client/  →  pydantic/httpx/     transport envelope, headers, BaseClient shell,
                        stdlib ONLY         error __init__ bodies, (opt.) ModelSpec shape
        ▲
 providers/_engine/  →  _client/  +         lifecycle/cleanup, error→runtime mapper,
                        runtime.errors,     capability gate, chat-completions parse skeleton
                        runtime.models
        ▲
 providers/<name>/   →  _engine/, _client/, wire schemas, MODEL_SPECS, _normalized_model_spec,
   engines/, client/    runtime, own client idiosyncratic rules  ← the per-provider truth
```

No cycle: `runtime` never imports `providers` (verified), so `_engine/`→`runtime` is strictly
downward. `_client/`→(nothing symai). This is the seam the consolidation must be drawn along.

---

## Target directory tree — `symai/providers/**`

`(exists)` = keep as-is · `(NEW)` = new shared module · `(thin)` = shrinks to provider
identity + hooks · `(extend)` = additive subtype/override.

```
symai/providers/
├─ _client/                         # SHARED CLIENT TIER — pydantic/httpx/stdlib ONLY, never runtime
│  ├─ models.py      (exists)       # StrictModel, TolerantModel, ModelId
│  ├─ errors.py      (extend)       # + APIError/ResponseError/TransportError.__init__ bodies (D4)
│  ├─ headers.py     (extend)       # + REQUEST_ID_HEADER/RETRY_AFTER_HEADER + base extract_response_metadata (D3)
│  ├─ transport.py   (NEW)          # base ResponseMetadata{status_code,request_id,retry_after} + APIResponse[T] (D3)
│  ├─ client.py      (NEW)          # BaseClient: transport construct+cleanup, close, _raise_for_status, _parse_response (D2)
│  └─ settings.py    (NEW)          # HttpProviderSettings base (5-field block) (D8)
│
├─ _engine/          (NEW)          # SHARED ENGINE TIER — may import _client + runtime.errors/models
│  ├─ base.py                       # HttpEngineBase[ModelT,SpecT]: __init__ (MODEL_SPECS lookup+cleanup),
│  │                                #   close, model/model_spec, _retry_after, _unsupported, _error_metadata (D5/PA-3)
│  ├─ mapping.py                    # map_execution(): _client.errors.* → runtime.errors.* (D1/PA-1)
│  ├─ gate.py                       # capability gate reading LanguageModelSpec matrix (PA-2)
│  └─ chat_completions.py           # ChatCompletionsAdapter(HttpLanguageEngine): choices-loop + finish map
│                                   #   + reasoning hook — imported by CEREBRAS + DEEPSEEK ONLY (D6/PA-4)
│
├─ openai/
│  ├─ client/
│  │  ├─ responses.py   (exists)    # Responses wire schema — STAYS per-provider
│  │  ├─ embeddings.py  (exists)    # Embeddings wire schema — STAYS per-provider
│  │  ├─ transport.py   (thin)      # re-export _client base ResponseMetadata/APIResponse
│  │  ├─ headers.py     (thin)      # re-export _client extract_response_metadata
│  │  ├─ errors.py      (thin)      # provider="openai" + subclass wiring only
│  │  └─ _client.py     (thin)      # Client(BaseClient): BASE_URL + errors module + _request/endpoint methods
│  ├─ engines/
│  │  ├─ responses.py   (STANDALONE LanguageAdapter — Responses wire is different; NOT ChatCompletionsAdapter)
│  │  └─ embedding.py   (EmbeddingAdapter: HttpEngineBase + own validate/parse)
│  ├─ settings.py       (thin)      # Responses/EmbeddingSettings = HttpProviderSettings alias
│  └─ loading.py        (exists)    # drop redundant preflight model-check (D8/PA-6)
│
├─ cerebras/
│  ├─ client/
│  │  ├─ chat.py        (exists)    # Chat-Completions wire schema + MODEL_SPECS + ReasoningEffort — per-provider
│  │  ├─ transport.py   (extend)    # subclass base ResponseMetadata + rate_limit: RateLimitState
│  │  ├─ headers.py     (extend)    # + x-ratelimit-* constants, extend extract_response_metadata
│  │  ├─ errors.py      (thin)      # provider="cerebras" only
│  │  └─ _client.py     (thin)      # Client(BaseClient): create_chat_completion (by_alias=True)
│  ├─ engines/chat_completions.py   # ChatCompletionsAdapter + _normalized_model_spec + MODEL_SPECS
│  │                                #   + _validate_provider_specifics hook (stop≤4, top_logprobs⇒logprobs, image-detail)
│  ├─ settings.py       (thin)      # ChatCompletionsSettings = HttpProviderSettings alias
│  └─ loading.py        (exists)    # drop redundant preflight model-check
│
└─ deepseek/                        # same shape as cerebras
   ├─ client/
   │  ├─ chat.py        (exists)    # DeepSeek Chat-Completions wire schema + MODEL_SPECS — per-provider
   │  ├─ transport.py   (thin)      # re-export _client base ResponseMetadata/APIResponse
   │  ├─ headers.py     (thin)      # re-export _client extract_response_metadata
   │  ├─ errors.py      (thin)      # provider="deepseek" only
   │  └─ _client.py     (thin)      # Client(BaseClient): create_chat_completion (no by_alias)
   ├─ engines/chat_completions.py   # ChatCompletionsAdapter + _normalized_model_spec + MODEL_SPECS
   │                                #   + _validate_provider_specifics hook (user-id regex, temp/top_p rule, stop≤16)
   ├─ settings.py       (thin)
   └─ loading.py        (exists)
```

**What stays strictly per-provider (never moves up):** each `client/{chat,responses,embeddings}.py`
wire schema; each engine's `_normalized_model_spec` + `MODEL_SPECS`; each `ReasoningEffort`
enum (per-provider member sets); each `_validate_provider_specifics` hook; OpenAI's entire
Responses parse path. These *are* the provider's truth; sharing them couples unrelated
schemas.

---

## Move table

Coupling risk per the two-tier rule above. LOC-saved is net (removed minus the shared code
added), rough, from live file sizes.

| # | Move | From → To (tier) | Coupling | Hook points that stay per-provider | Feature | ~LOC saved |
|---|------|------------------|----------|-----------------------------------|---------|-----------|
| D3a | Base `ResponseMetadata` + `APIResponse[T]` | `{openai,deepseek,cerebras}/client/transport.py` → **`_client/transport.py`** | **none** | cerebras subclasses to add `rate_limit: RateLimitState` | keeps-all | ~50 |
| D3b | `REQUEST_ID/RETRY_AFTER` consts + base `extract_response_metadata` | 3× `client/headers.py` → **`_client/headers.py`** | low | cerebras extends with 6 `x-ratelimit-*` headers | keeps-all | (in D3a) |
| D4 | `APIError/ResponseError/TransportError.__init__` bodies | 3× `client/errors.py` → **`_client/errors.py`** (param by `provider` ClassVar) | low | each file keeps only `provider="…"` + subclass MRO | keeps-all | ~90 |
| D2 | `BaseClient`: transport construct+cleanup, `close`, `_raise_for_status`, `_parse_response` | 3× `client/_client.py` → **`_client/client.py`** | low | `BASE_URL`, errors module, endpoint methods, `by_alias` flag (arg, not schema) | keeps-all | ~170 |
| D1/PA-1 | `map_execution()` error ladder `_client.errors.* → runtime.errors.*` | 4× engine `execute` except-block → **`_engine/mapping.py`** | **none** (catches shared bases) | provider label + model strings (args) | keeps-all | ~65 |
| D5/PA-3 | `HttpEngineBase`: `__init__`+cleanup, `close`, `model`/`model_spec`, `_retry_after`, `_unsupported`, `_error_metadata` | 4× engines → **`_engine/base.py`** | low | `MODEL_SPECS`, `Model` literal type, provider id | keeps-all | ~135 |
| PA-2 | Data-driven capability **gate** reading `LanguageModelSpec` | hardcoded `_unsupported` walls (47 calls) → **`_engine/gate.py`** + per-engine `_validate_provider_specifics` | low (reads matrix, doesn't own it) | idiosyncratic rules only (regex, cross-field, count limits) | keeps-all | ~60 |
| D6/PA-4 | `ChatCompletionsAdapter`: choices-loop + `_FINISH_REASONS` + reasoning-field hook | cerebras+deepseek `_parse_response`/`_output` → **`_engine/chat_completions.py`** | **MEDIUM — stop here** | `finish_reasons` map, index guard, message optionality, reasoning field name; **do NOT extend to openai** | keeps-all | ~120 |
| D8a | `HttpProviderSettings` 5-field base | 4 `settings.py` blocks → **`_client/settings.py`** | low | provider aliases/subclass | keeps-all | ~25 |
| D8b/PA-6 | Drop loader preflight `model not in MODEL_SPECS` (dup of engine `__init__`) | 3× `loading.py` | none | — (single source: engine ctor) | keeps-all | ~15 |
| D7 | ops `_symbol_value` (3×) + `_require_text` (2×) | `ops/{text,reason,compare}.py` → **`ops/primitives.py`** | none | — | keeps-all | ~18 |
| — | `_normalized_model_spec`, `MODEL_SPECS`, `chat.py`/`responses.py`/`embeddings.py`, `ReasoningEffort`, Responses parse | **STAY per-provider** | **do not touch** | — | — | 0 |

**Dependency ordering (do in this order):** D3 (base `ResponseMetadata`) + D4 (base error
`__init__`, which stores `.metadata`/`.body`) are prerequisites — they make D1's mapper and
D5's `_error_metadata` fully generic, because both read `error.metadata.{status_code,
request_id,retry_after}` off the shared base rather than a per-provider type. Then D2/D5,
then D1/PA-2/PA-3 on top of the base, then D6 last (highest risk, easiest to skip).

---

## The line NOT to cross (explicit)

1. **OpenAI Responses ≠ chat-completions.** Verified in `openai/engines/responses.py`:
   `_finish_reason` is status-based (reads `response.status` + `incomplete_details.reason`,
   lines ~361–378), `_parse_response` walks heterogeneous output items
   (`CompactionOutput`/`OutputMessage`/`ReasoningOutput`, ~305–356), and it enforces
   "reasoning ⇒ exactly one assistant message." The Chat-Completions adapter's contract
   (a `choices[].message` array with a `_FINISH_REASONS` string map) does not fit this. The
   `ChatCompletionsAdapter` is imported by **cerebras + deepseek only**; OpenAI's Responses
   engine stays a standalone `LanguageAdapter` subclass of `HttpEngineBase`. Forcing a shared
   base here would add branches for a shape only one provider has — negative value.

2. **Per-provider capability matrices must not be forced to share.** The three
   `_normalized_model_spec` bodies *look* like one duplicated `LanguageModelSpec(...)` call,
   but each encodes that provider's real capability truth — verified divergent:
   openai gates `content_types` on `spec.vision` and swaps `sampling_fields` between reasoning
   (`MAX_TOKENS, TOP_LOGPROBS`) and non-reasoning (`+TEMPERATURE, TOP_P`) tuples and sets
   `reasoning_summaries`; cerebras hardcodes `content_types=(TEXT, IMAGE)`,
   `sampling_fields=tuple(SamplingField)` (all), sets `reasoning_formats`, `vision=True`;
   deepseek uses `content_types=(TEXT,)`, a fixed 6-field sampling tuple, `vision=False`.
   Extracting a shared builder would pass ~9 provider-specific tuples as arguments — the
   divergence moves to the call site, nothing is removed, and one function must now change
   whenever *any* provider's capability model changes. **The gate (PA-2) is the correct
   consolidation: it *reads* whichever matrix the provider built, it does not centralize the
   matrix.** So the divergence in *what* is supported stays per-provider; only the *mechanism*
   of enforcement is shared.

3. **Wire schemas, `MODEL_SPECS` catalogs, and `ReasoningEffort` enums stay per-provider.**
   `client/{chat,responses,embeddings}.py` are different Pydantic contracts (cerebras
   `choice.message` optional + `image_tokens`/prediction details; deepseek `choice.message`
   required + `prompt_cache_hit/miss` splits; OpenAI Responses items). The `ReasoningEffort`
   values differ per provider. These are catalogs, not containers.

4. **`_client/` must never import `runtime`.** The error-mapper, gate, and `HttpEngineBase`
   go in `_engine/`, never `_client/`. This is the boundary that keeps "the client is a
   faithful API binding that never knows symai" true.

*(Optional / low-value, and my recommendation is to leave it: R1-02 D9 also floats hoisting
the structural client-side `ModelSpec`/`ReasoningSpec` dataclass shape into `_client/models.py`.
Live shapes actually differ — cerebras `ModelSpec` has **no** `vision` field, openai defaults
`vision=True`, deepseek requires it — so a shared base needs `vision` optional or absent, and
the whole win is ~15 lines across three 5-line frozen dataclasses. Coupling risk is low
(shape only) but the payoff is below the noise floor and it introduces a shared type spanning
three provider catalogs. Leave them per-provider; the honest split costs almost nothing.)*

---

## Non-provider seams

### S1 — `operations.py` (request builders) vs `ops/` (Symbol ops) → move to `runtime/requests.py`

**Verified.** `symai/operations.py` imports **only** `runtime.models` and builds
`LanguageModelRequest`/`EmbeddingRequest`/etc.; its consumers are exactly `function.py`
(`from symai.operations import language_request`) and `ops/embed.py`
(`from symai.operations import embedding_request, parse_embedding_response`). It never touches
`Symbol`. The name collides one keystroke away with the `ops/` package (semantic Symbol ops)
at the opposite end of the stack.

**Move:** `symai/operations.py` → **`symai/runtime/requests.py`** (co-locate the request
*builders* with the request *models* they construct). Consumers become
`from symai.runtime.requests import language_request` / `embedding_request, parse_embedding_response`.
No cycle: `runtime/requests.py` imports only `runtime.models`; `function.py` (exec) and
`ops/embed.py` (ops) both sit above `runtime`. `_string_tuple` (used only here) stays local.
Use `pyseam` for the module move. **Coupling: none. Feature: keeps-all. LOC: neutral
(placement + naming).**

### S2 — Two `loading.py`, two `load_runtime` → rename to kill both collisions

**Verified.** `runtime/loading.py::load_runtime(config, *, language_model_loaders,
embedding_loaders)` is the generic mechanism (preflight + allocation-free validation + failure
cleanup, provider-agnostic). `symai/loading.py::load_runtime(config, ...)` is the public
builtin-registry policy: it prepends `BUILTIN_LANGUAGE_MODEL_LOADERS`/`BUILTIN_EMBEDDING_LOADERS`
and delegates to the generic one imported `as _load_runtime` (the alias is the tell). Provider
loaders are imported lazily inside `_load_openai_responses` etc. — keeping `import symai`
inert. **The split is correct and must stay** (runtime knows nothing of providers — verified:
no `runtime/*` imports `providers`).

**Rename (keep the split):** generic `runtime/loading.py::load_runtime` → **`compose_runtime`**
(verb; "compose a runtime from an explicit registry") — removes the function-name collision
and the `as _load_runtime` alias. Public `symai/loading.py` → **`symai/registry.py`**, keeping
the documented public entry `load_runtime` — removes the file-name collision and names the
module for what it is (the builtin provider registry). Ripples to two test imports
(`tests/test_public_cutover.py`, `tests/runtime/test_loading.py`) — note only, do not edit.
**Coupling: none. Feature: keeps-all.**

### S3 — Client↔engine boundary → verified holding; the `_engine/` tier must preserve it

**Verified live** (the memory's key concern — "client = faithful binding that never knows
symai; engine = the only crossing point"):
- No `providers/*/client/*` or `providers/_client/*` imports `symai.runtime` (grep: none).
- Engines (`providers/*/engines/*`) are the **sole** crossing: they import `runtime.errors`,
  `runtime.models`, and their own `client` — never `Symbol`/`ops`/`Function` (grep: none).
- No cross-provider imports (deepseek ↮ cerebras ↮ openai — grep: none).
- `runtime/*` never imports `providers` (grep: none).

**Design rule for the new tier:** `providers/_engine/` sits *at the engine layer* — it may
import `_client` (down) and `runtime.errors`/`runtime.models` (down, since runtime never
imports providers). It must **not** import `Symbol`/`ops`/`Function`. `providers/_client/`
stays runtime-blind. When D1's mapper moves, it goes to `_engine/mapping.py` precisely because
it references `runtime.errors` — putting it in `_client/` would be the one move that breaks the
verified invariant. **This is the boundary the whole consolidation is organized to protect.**

### S4 — minor: `runtime/models.py` doubles as contracts + base-pydantic toolkit (optional)

**Verified.** `runtime/models.py` holds both the wire contracts *and* the base helpers
`FrozenModel` (line 19), `FiniteFloat`/`NonNegativeFiniteFloat`/`PositiveFiniteFloat`
(113–115), `ProviderId` (35). Imported as *foundation* by `runtime/config.py`,
`runtime/errors.py`, and all three `providers/*/settings.py`. It's a DAG sink (imports nothing
internal), so no cycle — purely a cohesion nit: provider settings pull `FrozenModel` /
`PositiveFiniteFloat` from a module named "models" (implying message contracts).
**Optional:** split the base helpers into `runtime/base.py` (or `runtime/pydantic.py`) that
`models.py`, `config.py`, `errors.py`, and provider settings import. Low priority; mentioned
for completeness since it is module placement. If `_client/settings.py` (D8a) is created, the
`HttpProviderSettings` base there would import from `runtime/base.py` rather than reaching into
`runtime/models`. **Coupling: none. Feature: keeps-all.**

---

## What is already good — keep

- **The client↔engine seam is genuinely clean and one-directional** (S3). The client having its
  own `ClientError` hierarchy distinct from `runtime/errors.py`, with engines translating at the
  crossing, is exactly right — and it is what makes D1's shared mapper *safe*: every provider
  error already subclasses the shared `_client/errors.py` bases
  (`class AuthError(APIError, client_errors.AuthError)`), so one mapper catching the base classes
  is zero-coupling.
- **`providers/_client/` is the right idea, just under-populated.** Every provider already
  imports `authorization_header`/`parse_optional_*` from `_client/headers` and subclasses
  `_client/errors.ClientError`. Extending it (D2–D4, D8a) follows the grain.
- **Construction-cleanup discipline is uniform and correct** — every `Client` and engine
  `__init__` wraps partial construction in `try/except BaseException` with `add_note` on cleanup
  failure. That it is *identical* everywhere is precisely why it should be hoisted (D2/D5), not
  rewritten.
- **Per-provider `MODEL_SPECS` normalization at import, behind `MappingProxyType`** is clean; the
  only fix (PA-2) is to *drive enforcement from it* via the gate, not to change it.
- **Lazy provider loading keeps `import symai` inert**, and the mechanism/policy loader split
  (S2) is a real strength — only the names collide.
- **Runtime stays provider-agnostic** (verified) — the inversion (generic loader takes loaders as
  params; builtin registry lives above runtime) is correct and the `_engine/` tier preserves it.
- **Frozen normalized contracts** (`runtime/models.py`: `frozen=True, strict=True,
  extra="forbid"` with real validators) — engines lean on these instead of re-validating.

---

## Verification notes (this round, live tree)

- Root `symai/__init__.py` = 0 bytes; `prompts.py`, `backend/` absent — post-cutover state
  confirmed.
- `_client/` = `models.py` (13) + `errors.py` (26) + `headers.py` (38); **no `transport.py`,
  no `_engine/`** — the two shared tiers this report proposes do not yet exist as such.
- Byte-level duplication confirmed by read: `openai/client/transport.py` ≡
  `deepseek/client/transport.py` (both `ResponseMetadata{status_code,request_id,retry_after}` +
  `APIResponse[T]`); cerebras `transport.py` is a strict superset (`+ rate_limit`). Provider
  `errors.py` `__init__` bodies identical across all three modulo the `provider` label and
  message prefix. `_client.py` `__init__`/`close`/`_raise_for_status`/`_parse_response` identical
  across all three (openai adds a generic `_request` for its richer surface).
- Error ladder identical 4× (verified in all four `execute` methods), differing only in the
  `*_errors` alias and the English provider name.
- Hand-rolled capability checks: `_unsupported(` appears openai 18 / cerebras 10 / deepseek 19;
  `grep` for `message_roles|content_types|response_formats` finds construction + definition only
  — the membership fields are populated-but-unread (PA-2 confirmed).
- ops helpers: `_symbol_value` defined in `ops/{reason,text,compare}.py` (3×), `_require_text`
  in `ops/{reason,text}.py` (2×); `ops/primitives.py` holds only `_execute_language` — the
  shared home exists and is under-used (D7 confirmed).
- Boundary greps all clean: no `client/`→`runtime`, no `engines/`→`Symbol|ops|function`, no
  cross-provider, no `runtime/`→`providers` (S3 confirmed).
</content>
</invoke>
