# r1-02 — Duplication & DRY

**Lens:** Quantify and localize duplication across the three language-model engine
adapters, the three hand-written client packages, and the ops helper layer. Propose
consolidation that does **not** couple unrelated provider schemas together.

**Snapshot:** branch `refactor/cleanup` (worktree engine-redesign). Moving target —
findings anchored by symbol + snippet.

---

## Executive summary

1. **The provider layer is the duplication hotspot, and most of it is safe to
   deduplicate.** Two of the three language engines (`cerebras` / `deepseek`
   `chat_completions.py`) are **~55% identical lines**; all three share a ~25% "HTTP
   engine infrastructure" skeleton (`close`, `model`/`model_spec` properties,
   `__init__` model-lookup + construction-cleanup, `_retry_after`, `_unsupported`,
   and the `execute()` error→runtime mapping) that carries **near-zero schema
   coupling**.
2. **The single highest-value, lowest-risk win** is the `execute()` error→runtime
   mapping, copied **4×** (3 language + 1 embedding engine) as 6 near-identical
   `except` clauses. Because every provider error already subclasses the shared
   `providers/_client/errors.py` base hierarchy, one free function catching the
   *base* classes replaces all four copies with no coupling.
3. **The client packages are the worst offenders.** `_client.py` is **83% identical**
   between cerebras/deepseek; `transport.py` and `headers.py` are **byte-identical**
   (openai ≡ deepseek verbatim, cerebras a strict superset); the three provider
   `errors.py` have identical `__init__` bodies. `providers/_client/` already exists
   as the shared home but only absorbs the leaf helpers — the envelopes, header
   extraction, and request plumbing are re-copied per provider.
4. **ops helpers are duplicated but trivially fixable:** `_symbol_value` appears
   verbatim **3×**, `_require_text` **2× plus 3 inlined variants**. `ops/primitives.py`
   already exists as the shared home (`_execute_language`) — these belong there.
5. **The real coupling tension is narrow and lives in exactly two places:**
   `_normalized_model_spec` (per-provider capability matrix) and the provider request/
   response schema modules (`responses.py`, `chat.py`). Those are correctly separate
   and must stay separate; deduping them *would* force unrelated providers to share a
   type. Everything else on the "duplicated" list is infrastructure, not schema.

**Overall read:** the layering (`_client/` shared base + per-provider `client/` +
`engines/`) is the right shape and is already ~60% of the way to a clean split. The
remaining duplication is because the shared base was under-populated, not because the
boundaries are wrong. Consolidation here is low-risk and high-yield *if* it stops at
the infrastructure line and never reaches into the per-provider request/response
schemas.

---

## Findings table

| ID | What | Where | Coupling risk | Confidence | Impact | Effort |
|----|------|-------|---------------|-----------|--------|--------|
| D1 | `execute()` error→runtime mapping copied 4× (6 except clauses each) | all engines | **none** | high | high | S |
| D2 | Client `_client.py` 83% identical (`_raise_for_status`, `_parse_response`, ctor, `close`, `_request`) | 3× `client/_client.py` | low | high | high | M |
| D3 | `transport.py` + `headers.py` byte-identical (openai≡deepseek) | 3× each | low | high | med | S |
| D4 | Provider `errors.py` `__init__` bodies identical 3× | 3× `client/errors.py` | low | high | med | S |
| D5 | Engine infra skeleton (`close`/props/`__init__` cleanup/`_retry_after`/`_unsupported`) copied 4× | all engines | low | high | med | M |
| D6 | `chat_completions` parse/usage skeleton ~55% shared (cerebras↔deepseek) | 2× engines | med | high | med | M |
| D7 | ops `_symbol_value` (3×) + `_require_text` (2× + 3 inline) duplicated | `ops/*.py` | none | high | med | S |
| D8 | `settings.py` 4 identical class bodies; `loading.py` near-identical + double model-check | providers | low | high | low | S |
| D9 | `_normalized_model_spec` + client `ModelSpec`/`ReasoningSpec` structurally duplicated | all engines/schemas | **real tension** | high | med | M |

---

## What is already good (keep)

- **`providers/_client/` exists and is used consistently for the leaf primitives.**
  Every provider's `client/headers.py` imports `authorization_header`,
  `parse_optional_float`, `parse_optional_int` from `providers/_client/headers`; every
  `transport.py` builds on `providers/_client/models.StrictModel`; every `errors.py`
  subclasses `providers/_client/errors.ClientError`. The shared base is the right idea —
  it is just under-populated (see D2–D4).
- **`ops/primitives.py._execute_language` is a clean shared primitive** imported by all
  five ops modules (`text`, `reason`, `compare`, `rank`, `embed`-adjacent). This is the
  model the ops helpers (D7) should follow.
- **The per-provider request/response schema split is correct and must not be merged.**
  OpenAI's Responses API (`client/responses.py`, `input`/`output` items, reasoning
  items) is genuinely a different wire contract from Chat Completions
  (`choices[].message`). Collapsing these would be false DRY.
- **Construction-cleanup discipline is uniform and correct** — every `Client` and every
  engine `__init__` wraps partial construction in `try/except BaseException` with
  `add_note` on cleanup failure. That this is *identical* everywhere is precisely why it
  should be hoisted (D5), not rewritten.

---

## Detailed findings

### D1 — `execute()` error→runtime mapping copied 4× (zero-coupling dedup)

**What.** Every engine's `execute()` maps provider client exceptions to the
`runtime.errors` hierarchy with the same six-arm `try/except`, differing only in the
provider-name string and the `errors` module alias.

**Where.** `ResponsesEngine.execute` (`openai/engines/responses.py`),
`ChatCompletionsEngine.execute` (cerebras + deepseek `engines/chat_completions.py`),
`EmbeddingEngine.execute` (`openai/engines/embedding.py`). Cerebras copy:

```python
        except cerebras_errors.AuthError as error:
            metadata = self._error_metadata(error.metadata)
            msg = "Cerebras rejected authentication"
            raise AuthenticationError(msg, metadata=metadata) from error
        except cerebras_errors.RateLimitError as error:
            ...
        except cerebras_errors.TransportError as error:
            metadata = ErrorMetadata(provider=self.provider, model=self.model)
            msg = "Cerebras transport failed"
            raise TransportError(msg, metadata=metadata) from error
        except cerebras_errors.APIError as error:
            metadata = self._error_metadata(error.metadata)
            msg = f"Cerebras API request failed with status {error.metadata.status_code}"
            raise ExecutionError(msg, metadata=metadata) from error
```

The deepseek and openai copies are identical modulo the word "Cerebras" and the
`_errors` alias. `rg` confirms 16 provider-message strings of the
`rejected authentication` / `rate-limited` / `transport failed` / `returned an invalid`
family.

**Why it matters.** This is ~24 lines × 4 = ~96 lines of pure copy-paste, and it is the
one place where an *added* runtime-error type (or a changed exception-chaining rule)
must be edited in four files in lockstep. It is also the least justified duplication:
the provider error classes (`cerebras_errors.AuthError`, etc.) **already** subclass the
shared bases in `providers/_client/errors.py` (`client_errors.AuthError`,
`client_errors.RateLimitError`, `client_errors.ResponseError`,
`client_errors.APIError`, `client_errors.TransportError`), and every provider `APIError`
already exposes `.metadata`.

**Proposed change.** One free function in `providers/_client/` (or `runtime`) that
catches the *shared base* classes and is called by every engine:

```python
# providers/_client/mapping.py  (sketch)
def run_mapped[T](
    call: Callable[[], T],
    *,
    provider_label: str,
    error_metadata: Callable[[object | None], ErrorMetadata],
    transport_metadata: ErrorMetadata,
) -> T:
    try:
        return call()
    except client_errors.AuthError as error:
        raise AuthenticationError(f"{provider_label} rejected authentication",
                                  metadata=error_metadata(error.metadata)) from error
    except client_errors.RateLimitError as error:
        ...
    except client_errors.TransportError as error:
        raise TransportError(f"{provider_label} transport failed",
                             metadata=transport_metadata) from error
    except client_errors.APIError as error:
        ...
```

Each engine's `execute()` shrinks to build-request → `run_mapped(...)` →
parse-response. Catching the base classes (not the per-provider leaves) is what makes
this **zero-coupling**: no provider schema is referenced.

**Feature impact:** `keeps-all` (message wording preserved via `provider_label`).
**Confidence:** high. **Impact:** high. **Effort:** S.

---

### D2 — Client `_client.py` is 83% identical across providers

**What.** `providers/{openai,cerebras,deepseek}/client/_client.py` share
`_raise_for_status`, `_parse_response`, the `Client.__init__` (transport ownership +
construction cleanup), `close`, and the request plumbing.

**Where.** Measured: cerebras vs deepseek `_client.py` = **99/119 matching lines
(83%)**. The `__init__` body is verbatim across all three:

```python
        authorization = authorization_header(api_key)
        owned_transport = transport
        if owned_transport is None:
            owned_transport = httpx.HTTPTransport(retries=connect_retries)
        elif connect_retries:
            msg = "connect_retries cannot be combined with an injected transport"
            raise ValueError(msg)
        try:
            http_client = httpx.Client(timeout=timeout, transport=owned_transport)
        except BaseException as error:
            try:
                owned_transport.close()
            except BaseException as cleanup_error:
                error.add_note(f"Client construction cleanup failed: {cleanup_error!r}")
            raise
        self._http_client = http_client
        self._headers = {"authorization": authorization}
        self._closed = False
```

`_raise_for_status` differs only in the provider-name strings; `_parse_response`
differs only in the message text and the target model type. The openai `_client.py`
additionally factors a generic `_request(method, path, model, ...)` (used by its
richer Responses surface: retrieve/delete/cancel/list) — cerebras/deepseek inline a
single `post` because they only expose `create_chat_completion`.

**Why it matters.** ~100 duplicated lines encode the retry/transport-ownership
contract, the JSON-decode-then-validate error taxonomy, and the auth-header wiring.
Any change to timeout defaults, cleanup semantics, or the decode error mapping must be
made three times.

**Proposed change.** Hoist a `BaseClient` (or a `request(...)` free function) into
`providers/_client/` that owns: transport construction + cleanup, `close`,
`_raise_for_status` parameterized by a provider label + the provider's error module,
and `_parse_response(response, metadata, model)`. Each provider `Client` becomes a thin
subclass that supplies `BASE_URL`, its `errors` module, and its endpoint methods
(`create_chat_completion` / `create_response` / `create_embeddings`).

**Coupling risk: low.** The one genuine variance is the `model_dump` call:
deepseek uses `model_dump(mode="json", exclude_none=True)` (no `by_alias`) while
openai/cerebras use `by_alias=True`. That is a per-endpoint parameter, not a schema
coupling — pass it as an argument. The provider request/response *types* stay in each
provider's `client/{chat,responses,embeddings}.py`; only the transport shell is shared.

**Feature impact:** `keeps-all`. **Confidence:** high. **Impact:** high. **Effort:** M.

---

### D3 — `transport.py` and `headers.py` are byte-identical across providers

**What.** The response-envelope models and the header-extraction function are copied
per provider.

**Where.**
- `openai/client/transport.py` ≡ `deepseek/client/transport.py` **verbatim** except the
  module docstring: both define `ResponseMetadata{status_code, request_id, retry_after}`
  and `APIResponse[T](StrictModel, Generic[T])`. Confirmed via `diff` (identical modulo
  provider token/docstring).
- `openai/client/headers.py` ≡ `deepseek/client/headers.py` **verbatim** except the
  import path: same `REQUEST_ID_HEADER` / `RETRY_AFTER_HEADER` constants and same
  `extract_response_metadata`. Confirmed via `diff`.
- Cerebras is a **strict superset**: its `transport.py` adds `RateLimitState` +
  `rate_limit: RateLimitState`, and its `headers.py` adds six `x-ratelimit-*` constants
  and populates `rate_limit`.

**Why it matters.** The base `ResponseMetadata`/`APIResponse` shape is not
provider-specific at all — it is the client-layer's normalized envelope. Re-declaring
it three times means the "every provider response carries status/request-id/retry-after"
invariant is enforced by copy-paste.

**Proposed change.** Move base `ResponseMetadata` + generic `APIResponse[T]` into
`providers/_client/transport.py`, and the `REQUEST_ID_HEADER`/`RETRY_AFTER_HEADER`
constants + a base `extract_response_metadata` into `providers/_client/headers.py`
(the module that already owns `parse_optional_float`/`parse_optional_int`). OpenAI and
DeepSeek then re-export the base directly. Cerebras subclasses `ResponseMetadata` to add
`rate_limit` and extends `extract_response_metadata` — a clean, additive override.

**Coupling risk: low.** Cerebras's rate-limit fields stay in cerebras. This is
subtype-extension, not shared mutation of a common type. **Feature impact:**
`keeps-all`. **Confidence:** high. **Impact:** med. **Effort:** S.

---

### D4 — Provider `errors.py` `__init__` bodies are identical 3×

**What.** Each provider `client/errors.py` re-declares `APIError.__init__`,
`ResponseError.__init__`, and `TransportError.__init__` with identical bodies; only the
`provider` class attribute and the default message prefix differ.

**Where.** All three files:

```python
class APIError(client_errors.APIError, Error):
    def __init__(self, metadata: ResponseMetadata, body: str, message: str | None = None) -> None:
        self.metadata = metadata
        self.body = body
        super().__init__(message or f"{PROVIDER} API error {metadata.status_code}")
```

`diff` (provider tokens stripped) shows the only differences are docstrings and
formatting; the executable lines are identical. `ResponseError.__init__`
(`self.metadata = metadata; self.body = body; super().__init__(message)`) and
`TransportError.__init__` (`self.metadata: None = None; super().__init__(message)`) are
byte-identical.

**Why it matters.** The attribute contract that the *engine* error-mapping (D1) relies
on (`error.metadata`, `error.body`) is defined by copy-paste. If D1's shared mapper is
adopted, it depends on these attributes existing uniformly — better to define them once.

**Proposed change.** Put the `__init__` bodies on the shared `providers/_client/errors`
classes, parameterized by a `provider: ClassVar[str]` and a `_error_prefix` used to
build the default message. Each provider file keeps only the `provider = "openai"` class
attr and (optionally) the docstrings.

**Coupling risk: low.** The one subtlety is the `metadata` *type* differs (cerebras's
`ResponseMetadata` carries `rate_limit`). But the base `__init__` only reads
`metadata.status_code` and stores the reference — a structural `ResponseMetadata` base
(D3) with `status_code` resolves this cleanly. **Feature impact:** `keeps-all`.
**Confidence:** high. **Impact:** med. **Effort:** S.

---

### D5 — Engine infrastructure skeleton copied 4×

**What.** Beyond D1, each engine repeats a fixed infrastructure block: the model-lookup
`__init__` with construction-cleanup, `close()`, the `model`/`model_spec` properties,
`_retry_after`, and `_unsupported`.

**Where.** `rg` confirms `_retry_after` defined 4× and `_unsupported` 3×, all verbatim:

```python
    @staticmethod
    def _retry_after(value: float | None) -> float | None:
        return value if value is not None and value >= 0 and isfinite(value) else None

    @staticmethod
    def _unsupported(message: str) -> None:      # deepseek: -> Never
        raise UnsupportedFeatureError(message)
```

The `__init__` model-lookup + cleanup block is identical across all four engines modulo
the "OpenAI/Cerebras/DeepSeek" word in the `UnsupportedModelError` message and the
`cast` target type:

```python
        try:
            try:
                model_spec = MODEL_SPECS[model]
            except KeyError as error:
                msg = f"Unsupported Cerebras language model: {model}"
                raise UnsupportedModelError(msg) from error
            self._client = client
            self._model = cast(...)
            self._model_spec = model_spec
            self._closed = False
        except BaseException as error:
            try:
                client.close()
            except BaseException as cleanup_error:
                error.add_note(f"Engine construction cleanup failed: {cleanup_error!r}")
            raise
```

**Why it matters.** ~40–50 lines × 4 of lifecycle/cleanup boilerplate. This is the
"HTTP engine" contract (owns a client, closes it once, guards partial construction) and
belongs in one place.

**Proposed change.** A `BaseHttpEngine` (generic over the model-literal type and the
spec type) providing `__init__(client, model, *, specs, provider_label)`, `close`,
`model`/`model_spec` properties, `_retry_after`, `_unsupported`, and `_error_metadata`.
Concrete engines subclass and implement only `execute` (which itself delegates to D1's
mapper) plus the provider-specific build/parse hooks (D6/D9). Note `deepseek._unsupported`
returns `Never` while the others return `None` — standardize on `Never` (it is strictly
more precise and the callers already treat it as no-return).

**Coupling risk: low.** The base touches only `client.close()`, `MODEL_SPECS[model]`,
and message strings — no request/response schema. **Feature impact:** `keeps-all`.
**Confidence:** high. **Impact:** med. **Effort:** M.

---

### D6 — Chat-Completions parse/usage skeleton ~55% shared (cerebras ↔ deepseek only)

**What.** The two Chat-Completions engines share the response-decoding shape: reject
empty `choices`, iterate choices guarding index validity + duplicate indices, sort by
index, wrap in `LanguageModelResponse`, and validate token-usage arithmetic before
building `TokenUsage`.

**Where.** Measured: cerebras vs deepseek `chat_completions.py` = **246/459 matching
lines (~55%)**. The `_parse_response` choice loop is near-identical:

```python
        seen_indices: set[int] = set()
        outputs: list[LanguageModelOutput] = []
        for choice in raw.choices:
            ... # index validity guard (cerebras: `index is None`; deepseek: `index < 0`)
            if choice.index in seen_indices:
                msg = "... response contained duplicate choice indices"
                raise InvalidResponseError(msg, metadata=error_metadata)
            seen_indices.add(choice.index)
            outputs.append(self._output(choice, error_metadata))
        outputs.sort(key=lambda output: output.index)
        return LanguageModelResponse(outputs=tuple(outputs), metadata=metadata)
```

Both also share the `_FINISH_REASONS = MappingProxyType({...})` finish-reason table
pattern (differing only in the provider's raw strings — cerebras maps `"error"`,
deepseek maps `"insufficient_system_resource"`) and the "validate usage arithmetic then
construct `TokenUsage`" pattern. OpenAI's `responses.py` does **not** match this — its
Responses-API output is a heterogeneous item list (`OutputMessage` / `ReasoningOutput` /
`CompactionOutput`), so its `_parse_response` is legitimately different (~45% line
overlap with cerebras, mostly the shared infra of D1/D5).

**Why it matters.** This is the largest *schema-adjacent* duplication. It is
consolidatable — but only between the two Chat-Completions providers, and only if the
base is made generic over the provider's `Choice`/`Message`/`Usage` types.

**Proposed change.** A `chat_completions` mixin/base (living under a shared
`chat_completions` helper, **not** under `_client/models`) that provides the choice-loop
skeleton and a `finish_reason` lookup, parameterized by:
- a `finish_reasons` mapping (provider-supplied),
- an `index_of(choice)` + `is_valid_index(choice)` hook,
- an `_output(choice)` hook,
- a `_usage(...)` hook.

**Coupling risk: MEDIUM — this is the boundary to watch.** The base would be generic
over `chat_api.Choice` / `chat_api.ChatCompletion`, which are *different Pydantic types*
in cerebras vs deepseek (cerebras `choice.message` is optional and has richer usage
details incl. `image_tokens`/prediction tokens; deepseek `choice.message` is required
and carries cache-hit/miss token splits). A shared base is viable *only* if it stays
generic (hooks, not shared fields) and never pulls the two providers' `chat.py` schemas
into a common module. If that genericity starts to cost more than the ~120 saved lines,
**leave these two as-is** — the duplication is honest here. Do **not** extend this base
to OpenAI. **Feature impact:** `keeps-all`. **Confidence:** high. **Impact:** med.
**Effort:** M.

---

### D7 — ops helpers duplicated (`_symbol_value`, `_require_text`) + inconsistent inlining

**What.** The Symbol-unwrap and text-guard helpers are copy-pasted across ops modules,
and two modules inline the same checks instead of calling the helper.

**Where.** `rg` confirms:
- `_symbol_value` defined **verbatim 3×**: `ops/text.py:297`, `ops/reason.py:79`,
  `ops/compare.py:88`:
  ```python
  def _symbol_value[T](symbol: Symbol[T], field: str) -> T:
      if not isinstance(symbol, Symbol):
          msg = f"{field} must be a Symbol"
          raise TypeError(msg)
      return symbol.value
  ```
- `_require_text` defined **2×** (`ops/text.py:305`, `ops/reason.py:87`) and **inlined 3
  more times**: `ops/compare.py:71` (`is_instance_of`: `"type_description must be
  text"`), `ops/rank.py:23-27` (both the Symbol check *and* `"measure must be text"`
  inlined), and `ops/embed.py` has its own bespoke `"source must be a Symbol containing
  non-empty text input(s)"` variant.
- `_string_tuple` lives once in `operations.py:103` and is used only there — that one is
  fine (single home), though it is the same *family* of guard.

**Why it matters.** Five ops modules, three private definitions, plus inlined variants
that have already drifted in wording (`"must be a Symbol"` vs
`"source must be a Symbol containing non-empty text input(s)"`). `ops/primitives.py`
already exists precisely as the shared ops-internal module (it houses
`_execute_language`, imported by all four language ops modules).

**Proposed change.** Move `_symbol_value` and `_require_text` into `ops/primitives.py`
and import them everywhere; replace the inlined checks in `rank.py`/`compare.py` with
the helper. `embed.py`'s `_text_inputs` is a different (richer) contract and can stay,
but its leading `isinstance(source, Symbol)` guard can reuse `_symbol_value`.

**Coupling risk: none** (single-layer, single-package). **Feature impact:** `keeps-all`.
**Confidence:** high. **Impact:** med. **Effort:** S.

---

### D8 — `settings.py` identical + `loading.py` near-identical with a redundant model-check

**What.** Provider settings and loaders are boilerplate clones.

**Where.**
- `settings.py`: **4 identical class bodies** — `openai` has `ResponsesSettings` and
  `EmbeddingSettings` (identical to each other), cerebras and deepseek each have a
  `ChatCompletionsSettings` — all five-field blocks are byte-identical:
  ```python
  class ChatCompletionsSettings(FrozenModel):
      api_key: SecretStr = Field(min_length=1)
      model: str = Field(min_length=1)
      request_timeout: PositiveFiniteFloat = 600.0
      connect_timeout: PositiveFiniteFloat = 10.0
      connect_retries: int = Field(default=0, ge=0)
  ```
- `loading.py`: the three language loaders share the exact shape — validate settings →
  lazy `import httpx` + client/engine → **preflight `if parsed.model not in
  MODEL_SPECS`** → build `Client(api_key, timeout=httpx.Timeout(...), connect_retries)`
  → return engine. The preflight model check **duplicates** the engine `__init__`'s own
  `MODEL_SPECS[model]` `KeyError → UnsupportedModelError` (D5) — the same validation
  runs twice, and both raise the same message.

**Why it matters.** Low individual weight but broad: 4 settings clones + 3 loader clones
+ a double-validated invariant. The double model-check means adding a model requires the
existence check to stay consistent in two places.

**Proposed change.** A single `HttpProviderSettings` base (the five-field block) in
`providers/_client/` or `runtime`; providers alias/subclass it. A shared
`build_http_engine(settings, *, client_factory, engine_factory)` helper collapses the
loader body. Drop the loader's preflight `in MODEL_SPECS` and rely on the engine
constructor's `UnsupportedModelError` (single source of truth) — or vice-versa, but not
both.

**Coupling risk: low** (config shape is genuinely shared, not schema). **Feature
impact:** `keeps-all`. **Confidence:** high. **Impact:** low. **Effort:** S.

---

### D9 — `_normalized_model_spec` + client `ModelSpec`/`ReasoningSpec`: the real coupling tension

**What.** Two related but *different* duplications sit at the schema boundary — one is
safe to touch, one is the genuine tension this lens was asked to flag.

**Where — the safe half (client schemas).** The client-side capability dataclasses are
structurally identical across all three providers:
```python
@dataclass(frozen=True, slots=True)
class ReasoningSpec:
    efforts: tuple[ReasoningEffort, ...]

@dataclass(frozen=True, slots=True)
class ModelSpec:
    context_tokens: int
    response_tokens: int
    reasoning: ReasoningSpec | None
    # + vision: bool  (openai default True, cerebras absent, deepseek explicit)
```
defined 3× (`openai/client/responses.py:51`, `cerebras/client/chat.py:67`,
`deepseek/client/chat.py:20`). The *shape* is shared; the *contents* (`ReasoningEffort`
enum members, the `MODEL_SPECS` catalog) are per-provider.

**Where — the real tension (`_normalized_model_spec`).** Each engine maps its client
`ModelSpec` → the normalized `runtime.models.LanguageModelSpec` capability matrix, and
the three mappings are structurally parallel but **semantically divergent**:
- openai: `content_types` gated on `spec.vision`; `sampling_fields` swaps between a
  reasoning and non-reasoning tuple; sets `reasoning_summaries`.
- cerebras: `content_types` always `(TEXT, IMAGE)`; `sampling_fields = tuple(SamplingField)`
  (all); sets `reasoning_formats`; `vision=True` hardcoded.
- deepseek: `content_types=(TEXT,)`; a fixed six-field `sampling_fields`; `vision=False`.

**Why it matters / why NOT to force-share.** `_normalized_model_spec` *looks*
duplicated (same `LanguageModelSpec(...)` constructor call three times) but each body
encodes that provider's actual capability truth. Extracting a shared builder would
require passing ~9 provider-specific tuples as arguments — you'd relocate the divergence
into the call site without removing it, and you'd create a single function that must
change whenever *any* provider's capability model changes. **This is exactly the
"dedup that couples unrelated provider schemas" the brief warns against.** Keep
`_normalized_model_spec` per-provider.

**Proposed change (only the safe half).** Hoist the *structural* `ReasoningSpec` +
`ModelSpec` dataclasses to `providers/_client/models.py` (which already holds
`StrictModel`/`TolerantModel`/`ModelId`) as generic bases: `ModelSpec` generic over the
reasoning type, or a plain shared base that providers extend with `vision`. The
`ReasoningEffort` enums and the `MODEL_SPECS` dicts stay per-provider — those *are* the
provider's model catalog and must not be shared. Leave `_normalized_model_spec`
untouched.

**Coupling risk: this is the line.** Sharing the `ModelSpec` *dataclass shape* is safe
(it is a container, not a catalog). Sharing `_normalized_model_spec` or `MODEL_SPECS`
would couple provider model catalogs and capability semantics — do not. **Feature
impact:** `keeps-all`. **Confidence:** high. **Impact:** med. **Effort:** M.

---

## Consolidation map (with coupling risk per move)

| Move | Target home | Removes | Coupling risk |
|------|-------------|---------|---------------|
| Error→runtime mapper (D1) | `providers/_client/` free fn | ~96 lines ×4 engines | **none** (catches shared base errors) |
| `BaseClient` transport shell (D2) | `providers/_client/` | ~100 lines ×3 | low (pass `by_alias`/label as args) |
| Base `ResponseMetadata` + `APIResponse` (D3) | `providers/_client/transport.py` | 2 files ×3 (cerebras extends) | low (subtype-extension) |
| Base header constants + `extract_response_metadata` (D3) | `providers/_client/headers.py` | 2 files ×3 (cerebras extends) | low |
| Error `__init__` bodies (D4) | `providers/_client/errors.py` | 3 `__init__`s ×3 | low |
| `BaseHttpEngine` lifecycle (D5) | `providers/_client/` or `runtime` | ~45 lines ×4 | low |
| Chat-Completions parse skeleton (D6) | shared `chat_completions` base | ~120 lines ×2 | **medium — stop here** |
| ops `_symbol_value`/`_require_text` (D7) | `ops/primitives.py` | 3+2 defs + 3 inlines | none |
| `HttpProviderSettings` + loader helper (D8) | `providers/_client/` | 4 settings + 3 loaders | low |
| Structural `ModelSpec`/`ReasoningSpec` shape (D9) | `providers/_client/models.py` | 3 dataclass pairs | medium (shape only) |
| `_normalized_model_spec`, `MODEL_SPECS`, `chat.py`/`responses.py` schemas | **stays per-provider** | — | **do not touch — real coupling** |

**Net:** roughly **350–450 lines** of infrastructure duplication across the provider
layer are removable at low/no coupling risk (D1–D5, D7–D8), plus ~120 more between the
two Chat-Completions engines at medium risk (D6). The hard boundary — capability
semantics and wire schemas (D9's second half) — stays split, and that split is correct.
