# r2-d1 — The ideal seam map: provider layer + module placement

**Round:** 2 (cross-cutting synthesis)
**Lens:** where the shared/per-provider line belongs, resolving *consolidate duplication*
vs *don't couple unrelated provider schemas*. Inputs: `r1-02-duplication`,
`r1-07-provider-adapters`, `r1-03a/03b-boundaries`. All findings re-verified against the
**post-cutover** tree (HEAD `84f703b`; root `__init__.py` is 0 bytes, `prompts.py`/`backend/`
gone). Read-only. Anchored by symbol + snippet; line numbers approximate.

---

## Executive summary

1. The provider layer is the redesign's one real duplication hotspot, and the boundaries
   are already the *right shape* — `_client/` (raw HTTP toolkit, zero runtime knowledge) +
   per-provider `client/` + `engines/`. The shared homes are simply **under-populated**: the
   error ladder, transport shell, envelopes, error `__init__`s, and the entire engine
   lifecycle skeleton are copy-pasted 3–4× instead of hoisted.
2. The clean split is **two shared bases, not one**: extend `providers/_client/` (transport
   shell + envelopes + header/error `__init__`s + settings) and add a new
   `providers/_engine/` (adapter lifecycle + error→runtime mapper + capability gate), plus a
   **chat-completions-only** sub-layer (`_engine/chat_completions.py`) shared by cerebras +
   deepseek and **nobody else**.
3. **The line not to cross is sharp and I can name it exactly:** OpenAI Responses is a
   different wire contract from Chat Completions — its `_parse_response` stays standalone. And
   the three `_normalized_model_spec` bodies + `MODEL_SPECS` catalogs + `ReasoningEffort`
   enums + the `chat.py`/`responses.py`/`embeddings.py` wire schemas **stay strictly
   per-provider**. Sharing those would relocate divergence into a call site, not remove it.
4. The client↔engine boundary the project cares about **holds in live code** (verified: no
   `providers/*/client/*` imports `runtime`; engines are the sole crossing). The new
   `_engine/` base preserves it — `_engine` imports `runtime.*` + `_client.*` downward and
   never a concrete provider client.
5. Two non-provider seams remain: rename `symai/operations.py` → `symai/runtime/requests.py`
   (kills the `operations`/`ops` collision) and disambiguate the two `load_runtime`
   (`runtime/loading.py::load_runtime` → `build_runtime`). Both keep-all. Net removable at
   low/no coupling risk: **~350–450 LOC**, plus ~120 at medium risk (the chat adapter).

---

## Target directory tree (`symai/providers/**`)

`[+]` new file · `[~]` gains/loses code · `●` stays strictly per-provider (the coupling wall)

```
symai/providers/
├── _client/                         SHARED CLIENT TOOLKIT — raw HTTP, knows nothing of runtime
│   ├── models.py                    StrictModel, TolerantModel, ModelId              [keep]
│   ├── transport.py            [+]  base ResponseMetadata{status_code,request_id,
│   │                                retry_after} + APIResponse[T]                    (D3)
│   ├── headers.py             [~]  authorization_header, parse_optional_* [keep]
│   │                                + REQUEST_ID/RETRY_AFTER consts
│   │                                + base extract_response_metadata()               (D3)
│   ├── errors.py              [~]  ClientError hierarchy [keep]
│   │                                + shared APIError/ResponseError/TransportError
│   │                                  __init__ bodies (provider: ClassVar[str])      (D4)
│   ├── client.py              [+]  BaseClient: transport ownership + cleanup, close,
│   │                                _raise_for_status(*, errors, display),
│   │                                _parse_response(resp, meta, model),
│   │                                _request(method, path, model, *, body, params,
│   │                                         by_alias)                                (D2)
│   └── settings.py            [+]  HttpProviderSettings (5-field FrozenModel base)   (D8)
│
├── _engine/                    [+]  SHARED ENGINE (ADAPTER) INFRA — provider-neutral;
│   │                                imports runtime.* + _client.*; NEVER a provider client
│   ├── base.py                [+]  BaseHttpEngine[ModelT, SpecT]: __init__ (spec lookup
│   │                                → UnsupportedModelError, adopt client, cleanup),
│   │                                close, model/model_spec props, _retry_after,
│   │                                _unsupported(-> Never), _error_metadata(meta)    (D5/PA-3)
│   ├── mapping.py             [+]  map_execution(call, *, provider, display, model,
│   │                                error_metadata): client_errors.* → runtime.errors.*
│   │                                (the 5-branch ladder, once)                       (D1/PA-1)
│   ├── gate.py                [+]  gate_language_capabilities(request, spec, *,
│   │                                unsupported): data-driven off LanguageModelSpec   (PA-2)
│   ├── chat_completions.py    [+]  ChatCompletionsAdapter(BaseHttpEngine): choices
│   │                                loop (empty/index/dedup/sort), finish-reason
│   │                                lookup, _output template + reasoning hook
│   │                                ── cerebras + deepseek ONLY, never OpenAI         (D6/PA-4)
│   └── loading.py             [+]  build_http_engine(settings, *, settings_model,
│                                    client_factory, engine_factory)                   (D8/PA-6)
│
├── openai/
│   ├── settings.py            [~]  ResponsesSettings/EmbeddingSettings = HttpProviderSettings
│   ├── loading.py             [~]  thin — delegates to _engine.loading.build_http_engine
│   ├── client/
│   │   ├── _client.py         [~]  OpenAIClient(BaseClient) + endpoint methods
│   │   │                            (create_response, retrieve/delete/cancel/list,
│   │   │                             create_embeddings)                               ●-endpoints
│   │   ├── responses.py        ●   Responses wire schema + ModelSpec + ReasoningEffort + MODEL_SPECS
│   │   ├── embeddings.py       ●   embeddings wire schema + ModelSpec + MODEL_SPECS
│   │   ├── transport.py       [~]  re-export base ResponseMetadata/APIResponse (~3 lines)
│   │   ├── headers.py         [~]  re-export base extract_response_metadata (~3 lines)
│   │   └── errors.py          [~]  provider="openai" + class attrs only (bodies inherited)
│   └── engines/
│       ├── responses.py        ●   ResponsesEngine(BaseHttpEngine) — STANDALONE parse path
│       │                            (Responses wire ≠ chat) + _normalized_model_spec
│       └── embedding.py       [~]  EmbeddingEngine(BaseHttpEngine)
│
├── cerebras/
│   ├── settings.py            [~]  ChatCompletionsSettings = HttpProviderSettings
│   ├── loading.py             [~]  thin
│   ├── client/
│   │   ├── _client.py         [~]  CerebrasClient(BaseClient) + create_chat_completion
│   │   ├── chat.py             ●   chat wire schema + ModelSpec + ReasoningEffort + MODEL_SPECS
│   │   ├── transport.py       [~]  ResponseMetadata(base) + RateLimitState (additive subtype)
│   │   ├── headers.py         [~]  extends base extract_* (adds x-ratelimit-* + rate_limit)
│   │   └── errors.py          [~]  provider="cerebras" + class attrs only
│   └── engines/
│       └── chat_completions.py ●   ChatCompletionsEngine(ChatCompletionsAdapter)
│                                    + _normalized_model_spec + _validate_provider_specifics
│                                    + _FINISH_REASONS + _rate_limit
│
└── deepseek/                        (mirror of cerebras, NO rate-limit extension)
    ├── settings.py            [~]  ChatCompletionsSettings = HttpProviderSettings
    ├── loading.py             [~]  thin
    ├── client/
    │   ├── _client.py         [~]  DeepSeekClient(BaseClient) + create_chat_completion (by_alias=False)
    │   ├── chat.py             ●   chat wire schema + ModelSpec + ReasoningEffort + MODEL_SPECS
    │   ├── transport.py       [~]  re-export base ResponseMetadata/APIResponse
    │   ├── headers.py         [~]  re-export base extract_response_metadata
    │   └── errors.py          [~]  provider="deepseek" + class attrs only
    └── engines/
        └── chat_completions.py ●   ChatCompletionsEngine(ChatCompletionsAdapter)
                                     + _normalized_model_spec + _validate_provider_specifics
                                     + _FINISH_REASONS
```

Three tiers, made explicit:

| Tier | Home | Contents | Shared by |
|------|------|----------|-----------|
| **Shared base** | `_client/*`, `_engine/base.py`, `mapping.py`, `gate.py`, `loading.py` | transport shell, envelopes, error `__init__`s, settings, adapter lifecycle, error mapper, capability gate | **all 4 engines / 3 clients** |
| **Chat-completions only** | `_engine/chat_completions.py` | choices-loop parse skeleton, finish-reason lookup, reasoning hook | **cerebras + deepseek only** |
| **Strictly per-provider (●)** | `client/{chat,responses,embeddings}.py`, `engines/*` `_normalized_model_spec` + `_validate_provider_specifics`, `MODEL_SPECS`, `ReasoningEffort` | wire request/response schemas, capability matrices, idiosyncratic rules | **nobody — the wall** |

---

## The line NOT to cross (the coupling wall)

These are separate for a reason; consolidating them is *false DRY* — it moves the divergence
into arguments/hooks without removing it, and creates one function that must change whenever
*any* provider's model catalog or wire shape changes.

1. **OpenAI Responses ≠ Chat Completions.** `openai/engines/responses.py::_parse_response`
   walks a heterogeneous item list (`OutputMessage`/`ReasoningOutput`/`CompactionOutput`),
   derives finish reason from `ResponseStatus` + `incomplete_details.reason`, and enforces
   "reasoning ⇒ exactly one assistant message" (verified, lines ~293–378). The cerebras/deepseek
   path is `choices[].message` + a `_FINISH_REASONS` dict. **`ResponsesEngine` subclasses only
   `BaseHttpEngine`, never `ChatCompletionsAdapter`.** Do not extend the chat adapter to OpenAI.

2. **`_normalized_model_spec` stays per-provider.** The three bodies *look* duplicated (same
   `LanguageModelSpec(...)` call) but each encodes that provider's capability truth:
   cerebras `content_types=(TEXT, IMAGE)`, `sampling_fields=tuple(SamplingField)` (all),
   `reasoning_formats`, `vision=True`; deepseek `content_types=(TEXT,)`, a fixed 6-field
   `sampling_fields`, `vision=False`; openai gates `content_types` on `spec.vision` and swaps
   reasoning/non-reasoning sampling tuples + `reasoning_summaries` (verified across all three
   engine heads). A shared builder would take ~9 provider-specific tuples as arguments — pure
   relocation. Keep it in each engine.

3. **`MODEL_SPECS` catalogs, `ReasoningEffort` enums, and the wire schema modules
   (`chat.py` / `responses.py` / `embeddings.py`) stay per-provider.** These *are* the
   provider's model catalog and API binding.

4. **`ChatCompletionsAdapter` must remain generic over the provider's `Choice`/`ChatCompletion`/
   `Usage` types via hooks — never importing either provider's `chat.py`.** Cerebras
   `choice.index: int | None` + optional `message` + `message.reasoning`; deepseek
   `choice.index: int` (guards `< 0`) + required `message` + `message.reasoning_content`
   (verified). If keeping the base generic starts to cost more than the ~120 lines it saves,
   **leave cerebras/deepseek duplicated** — that duplication is honest. This is the one
   **medium** coupling-risk move; everything above the wall is low/none.

---

## Move table (the core deliverable)

Coupling-risk: **none** = references only shared bases / single layer; **low** = subtype-
extension or a parameter, no schema shared; **med** = generic base over divergent provider
types — stop if it stops paying.

| # | Move | From → To | Hook points the concrete side supplies | Coupling | Feat. impact | ~LOC saved |
|---|------|-----------|----------------------------------------|----------|--------------|-----------|
| M1 | Error→runtime mapper | 4× `execute()` ladders → `_engine/mapping.py::map_execution` | `provider`, `display`, `model`, `error_metadata` callable | **none** | keeps-all | ~55 |
| M2 | BaseClient transport shell | 3× `client/_client.py` → `_client/client.py::BaseClient` | `BASE_URL`, `errors` module, `extract_response_metadata`, `by_alias`, endpoint methods | low | keeps-all | ~120 |
| M3 | Base `ResponseMetadata` + `APIResponse[T]` | 3× `client/transport.py` → `_client/transport.py` | cerebras *subclasses* to add `RateLimitState`; openai/deepseek re-export | low | keeps-all | ~40 |
| M4 | Header consts + base `extract_response_metadata` | 3× `client/headers.py` → `_client/headers.py` | cerebras extends (adds 6 `x-ratelimit-*` + `rate_limit`) | low | keeps-all | ~25 |
| M5 | Error `__init__` bodies | 3× `client/errors.py` → `_client/errors.py` classes | `provider: ClassVar[str]` + default-message prefix | low | keeps-all | ~60 |
| M6 | Engine lifecycle skeleton | 4× engines → `_engine/base.py::BaseHttpEngine` | `MODEL_SPECS`, `provider`, model-kind label; `execute`, `_build_request`, `_parse_response` | low | keeps-all | ~130 |
| M7 | Data-driven capability gate | 4× hand-rolled `_validate_request` checks → `_engine/gate.py` | `_validate_provider_specifics(request)` hook (idiosyncratic rules only) | none* | keeps-all | ~50 |
| M8 | Chat-completions parse skeleton | 2× `engines/chat_completions.py` → `_engine/chat_completions.py::ChatCompletionsAdapter` | `finish_reasons` map, `_extract_reasoning`, index/message-optionality hooks, `_build_request`, `_message`, `_response_format`, `_usage` | **med** | keeps-all | ~120 |
| M9 | `HttpProviderSettings` + loader helper | 4× `settings.py` + 3× `loading.py` → `_client/settings.py` + `_engine/loading.py::build_http_engine` | `settings_model`, `client_factory`, `engine_factory`; drop redundant preflight `model not in MODEL_SPECS` | low | keeps-all | ~65 |
| M10 | Structural `ModelSpec`/`ReasoningSpec` shape | 3× dataclass pairs → `_client/models.py` bases | providers extend with `vision` / their `ReasoningEffort` | low (shape only) | keeps-all | ~20 |
| M11 | ops `_symbol_value` + `_require_text` | 3×+2× in `ops/*.py` → `ops/primitives.py` | replace inline checks in `rank.py`/`compare.py` | none | keeps-all | ~20 |
| M12 | `operations.py` → `runtime/requests.py` | rename/move | consumers: `function.py`, `ops/embed.py` | none | keeps-all | 0 (rename) |
| M13 | `runtime/loading.py::load_runtime` → `build_runtime` | rename | `symai/loading.py` calls `build_runtime`; public `load_runtime` stays | none | keeps-all | 0 (rename) |
| — | **`_normalized_model_spec`, `MODEL_SPECS`, `chat.py`/`responses.py`/`embeddings.py`** | **STAYS per-provider** | — | **wall** | — | — |

`*` M7 is a *behavior change* (removes the parallel hardcoded checks that re-encode the
matrix), not a pure move — zero *schema* coupling (reads matrix data), but coordinate with the
provider-adapter lens. **Net ≈ 350–450 LOC** removable at low/none risk (M1–M7, M9–M11), plus
**~120** at medium risk (M8). Current provider layer is 3472 LOC → target ~2800–2950.

---

## Per-move detail + exact hook points

### M1 — `map_execution` (`_engine/mapping.py`) · coupling none
Every `execute()` wraps the client call in the identical 5-branch ladder
(Auth/RateLimit/Response/Transport/API), differing only in the `*_errors` alias and one
English name. Verified byte-parallel across `openai/engines/responses.py:129–148`,
`openai/engines/embedding.py:95–114`, `cerebras/…:134–153`, `deepseek/…:148–167`.
**Zero-coupling because every provider error subclasses the shared bases** —
`class AuthError(APIError, client_errors.AuthError)`, `class APIError(client_errors.APIError, Error)`
(verified in all three `client/errors.py`). One mapper catches `client_errors.{AuthError,
RateLimitError, ResponseError, TransportError, APIError}`.

```python
def map_execution[T](call, *, provider, display, model, error_metadata) -> T:
    try: return call()
    except client_errors.AuthError as e:      raise AuthenticationError(f"{display} rejected authentication", metadata=error_metadata(e.metadata)) from e
    except client_errors.RateLimitError as e: raise RateLimitError(f"{display} rate-limited the request", metadata=error_metadata(e.metadata)) from e
    except client_errors.ResponseError as e:  raise InvalidResponseError(f"{display} returned an invalid response", metadata=error_metadata(e.metadata)) from e
    except client_errors.TransportError as e: raise TransportError(f"{display} transport failed", metadata=ErrorMetadata(provider=provider, model=model)) from e
    except client_errors.APIError as e:       raise ExecutionError(f"{display} API request failed with status {e.metadata.status_code}", metadata=error_metadata(e.metadata)) from e
```
**Order is load-bearing** (Auth/RateLimit are `APIError` subclasses → caught first;
Response/Transport are ClientError siblings). `execute` collapses to
`return self._parse_response(map_execution(lambda: self._client.create_…, …))`.
**Prerequisite:** M3 (base `ResponseMetadata` with `status_code`) so `error_metadata` reads
metadata uniformly.

### M6 — `BaseHttpEngine` (`_engine/base.py`) · coupling low
`__init__` (spec lookup → `UnsupportedModelError` → adopt client → `try/except BaseException`
cleanup with `add_note`), `close`, `model`/`model_spec` props, `_retry_after`, `_unsupported`,
`_error_metadata` are byte-identical bar the provider word and the `cast(...)` target across all
four engines (verified — the cleanup block `except BaseException as error: try: client.close()
except BaseException as cleanup_error: error.add_note(...)` is verbatim ×4).
Generic `BaseHttpEngine[ModelT, SpecT]`; `__init__(*, client, model, specs, provider,
model_kind)` builds the `f"Unsupported {display} {model_kind} model: {model}"` message.
**Standardize `_unsupported -> Never`** (deepseek already does; the other three return `None`).
**Standardize `_error_metadata(meta: ResponseMetadata)`** non-optional (embedding's `| None` is
the odd one out — PA-6). This base is the natural home for M1's mapper and M7's gate.

### M7 — capability gate (`_engine/gate.py`) · coupling none, behavior change
Grep confirms `message_roles`/`content_types`/`response_formats` are **populated but never read
for enforcement** — each engine re-encodes those facts as hardcoded `isinstance`/`if` rejections
(e.g. deepseek rejects `DeveloperMessage`, `ImageContent` ×2, `JsonSchemaResponseFormat`,
and `seed/frequency_penalty/presence_penalty/logit_bias` by hand at lines 204–279, all of which
its own matrix already declares unsupported). `gate_language_capabilities(request, spec, *,
unsupported)` iterates roles/content-types/response-format/set-sampling-fields/reasoning against
the matrix (single source of truth). Each engine keeps only `_validate_provider_specifics` — the
rules a boolean matrix can't express: deepseek's user-id regex + "temp/top_p ignored unless
thinking disabled" + stop≤16; cerebras's stop≤4 + `top_logprobs requires logprobs`; openai's
`max_tokens ≤ response_tokens` + "no assistant reasoning input". Needs a static
`SamplingField → attr` table to know which fields the request *sets*.

### M8 — `ChatCompletionsAdapter` (`_engine/chat_completions.py`) · coupling MED
The shared choices loop (verified ~55% line overlap, cerebras vs deepseek): empty-choices
guard → per-choice index-validity → `seen_indices` dedup → `_output(choice)` →
`outputs.sort(key=…index)` → wrap in `LanguageModelResponse`. Hooks: `finish_reasons:
Mapping[str, FinishReason]`, `_choice_index`/`_is_valid_index` (cerebras `index is None`,
deepseek `index < 0`), `_extract_reasoning` (`message.reasoning` vs `message.reasoning_content`),
message-optionality. **Generic over `chat_api.Choice`/`ChatCompletion` — must never import
either provider's `chat.py`.** This is *the* boundary to watch: if the hook count balloons past
the ~120 saved lines, keep the two engines separate.

### M2–M5, M9–M10 — client-side hoists · coupling low
- **M2 BaseClient:** `__init__` body verbatim ×3 (auth header, transport ownership + cleanup,
  `_headers`, `_closed`). openai keeps its richer generic `_request(method, path, model, *,
  body, params)` for retrieve/delete/cancel/list; cerebras/deepseek keep a one-line
  `create_chat_completion`. The **only** genuine variance is `by_alias` — deepseek dumps with
  `model_dump(mode="json", exclude_none=True)` (no `by_alias`), openai/cerebras use
  `by_alias=True` (verified). That's a per-call parameter, not schema coupling.
- **M3/M4:** base `ResponseMetadata{status_code, request_id, retry_after}` + `APIResponse[T]`
  are declared **byte-identical** in openai≡deepseek `transport.py`; cerebras is a strict
  additive superset (`RateLimitState` + `rate_limit`). Subtype-extension, not shared mutation.
- **M5:** `APIError/ResponseError/TransportError.__init__` bodies identical ×3 (`self.metadata
  = metadata; self.body = body; super().__init__(message or f"{PROVIDER} API error
  {metadata.status_code}")`). Push bodies to the `_client/errors.py` classes keyed on
  `provider: ClassVar[str]`. This *stabilizes the `.metadata`/`.body` contract M1's mapper
  depends on* — currently defined by copy-paste.
- **M9:** the 5-field settings block is byte-identical ×4; the 3 loaders share the exact shape
  and each **re-runs** the `if parsed.model not in MODEL_SPECS` check the engine `__init__`
  already performs (double-validated invariant, two copies of the message). Drop the preflight;
  the engine ctor is the single source.
- **M10:** the client-side `@dataclass(frozen=True, slots=True) ModelSpec{context_tokens,
  response_tokens, reasoning}` + `ReasoningSpec{efforts}` are structurally identical at
  `openai/client/responses.py:51`, `cerebras/client/chat.py:67`, `deepseek/client/chat.py:20`.
  Hoist the *shape* to `_client/models.py`; the `ReasoningEffort` enum members and the
  `MODEL_SPECS` catalog contents **stay per-provider** (they are the catalog, not a container).

---

## Non-provider seams

### S1 — `operations.py` vs `ops/` collision → `runtime/requests.py`
`symai/operations.py` (request *builders*: `language_request`, `image_request`,
`embedding_request`, `data_uri`, `parse_embedding_response`, `_string_tuple`) imports **only**
`runtime.models` and never touches `Symbol` (verified). `symai/ops/` is the high-level
Symbol-wrapping DSL. Same word, opposite ends of the stack. **Move to `symai/runtime/requests.py`**
— co-locate the builders with the `runtime.models` types they construct. Consumers become
`from symai.runtime.requests import language_request` (`function.py`) and `… import
embedding_request, parse_embedding_response` (`ops/embed.py`). Prefer `pyseam` for the module
rename. Layering stays clean (within-runtime dep + downward consumers). keeps-all.

### S2 — two `loading.py` / two `load_runtime` → keep split, rename generic half
The mechanism/policy split is a genuine strength and **must stay**: `runtime/loading.py`
(generic, provider-agnostic preflight + allocation-free validation + failure cleanup — verified
no `runtime/*` imports `providers`) vs `symai/loading.py` (builtin registry that prepends
`BUILTIN_*_LOADERS` and defers provider imports into function bodies, keeping `import symai`
inert). Only the names collide (forcing `load_runtime as _load_runtime`). **Rename
`runtime/loading.py::load_runtime` → `build_runtime`** (compose-from-explicit-registry); keep the
public `symai/loading.py::load_runtime` as the documented entry. Optionally rename the registry
file `symai/loading.py` → `symai/builtins.py`. keeps-all, low priority. *(Tangential: the four
`cast("ImplementationId", "openai:responses")` in `symai/loading.py` are no-op casts on string
literals — a `TypeAdapter`-validated tuple or plain typing would drop them; not a seam issue.)*

### S3 — client↔engine boundary (memory's key concern) — VERIFIED HOLDS
`grep` over `symai/providers/*/client/*` shows **no import of `runtime` or `symai.ops`/
`symai.function`** — clients import only `pydantic`/`httpx`, their own package, and
`providers._client.*`. Engines (`providers/*/engines/*`) are the **sole** crossing: they import
`runtime.models`/`runtime.errors` downward *and* their own `client`. The client is a faithful
API binding that never learns about symai; the engine is the only crossing point. **The new
`_engine/` base does not weaken this** — it imports `runtime.*` + `_client.*` (both downward)
and must never import a concrete provider's `client`. Enforce with an import-linter contract
(r1-03b B6): `providers.*.client` → only `providers._client`; `runtime.*` ↛ `providers`;
`providers._engine` ↛ `providers.{openai,cerebras,deepseek}`.

---

## What's already good — keep

- **The three-tier shape is correct** (`_client/` toolkit → per-provider `client/` → `engines/`);
  it's ~60% of the way to a clean split and just needs the shared homes filled.
- **Client↔engine layering is honored in live code** (verified above) — the redesign's headline
  boundary claim holds.
- **Construction-cleanup discipline is uniform and correct** (client + engine both `client.close()`
  + `add_note` on failed init without swallowing the original) — *that it's identical everywhere
  is exactly why it should be hoisted (M6), not rewritten.*
- **Typed, layered error taxonomy** (`_client/errors.py` 5-class base; each provider subclasses;
  runtime keeps its own neutral hierarchy) — this is what makes M1's shared mapper trivially safe.
- **Per-provider `MODEL_SPECS` normalization at import behind `MappingProxyType`** — clean; the
  fix (M7) is to make the normalized result *drive* enforcement, not to change the normalization.
- **`ops/primitives.py::_execute_language`** is the model M11 should follow (shared ops primitive,
  imported by all language ops). **Lazy provider loading + inert `import symai`** (empty root,
  `__getattr__` provider facades, deferred loader imports) — preserve; guard with the contract in S3.
- The **cutover is genuinely done**: `function.py` no longer imports `prompts` (examples are
  `Sequence[str] | str | None`), root `__init__.py` is empty, `backend/`/`prompts.py` gone.
  r1-03a/03b's B1/B2/B3 are **RESOLVED**; only the naming seams (S1/S2) and provider dedup remain.
