# r1-07 — Provider-adapter design quality (`engines/` layer)

**Lens:** the adapters that map a normalized `LanguageModelRequest`/`EmbeddingRequest` ↔ a
provider client. Files audited: `providers/openai/engines/responses.py` (451),
`providers/openai/engines/embedding.py` (206), `providers/cerebras/engines/chat_completions.py` (459),
`providers/deepseek/engines/chat_completions.py` (435), plus each provider's `client/` (errors,
transport, `_client`) and `runtime/models.py`, `runtime/errors.py`, `runtime/engines.py`, `loading.py`.

Snapshot moves under me; findings are anchored by **symbol + snippet**, line numbers approximate.

---

## Executive summary

1. The four engines are **structurally near-identical scaffolding wrapped around three genuinely
   different wire shapes**. `execute`, the 5-branch error ladder, `__init__` + construction-cleanup,
   `close`, `model`/`model_spec` properties, `_retry_after`, `_unsupported` are **verbatim duplicates**
   across engines — a shared adapter base is strongly warranted and loses nothing.
2. The **`LanguageModelSpec` capability matrix is largely populated-but-unread**. `message_roles`,
   `content_types`, `response_formats` are **never consulted for enforcement anywhere in `symai/`**
   (grep-confirmed); `sampling_fields` is only half-consulted. Enforcement is instead a parallel wall
   of hardcoded `if <unsupported>: _unsupported(...)` that *re-encodes what the matrix already says* —
   the exact "spec says X, imperative check says Y" drift the brief flags. A data-driven gate off the
   matrix removes ~60 lines and makes the matrix authoritative.
3. **Cerebras and DeepSeek `_parse_response`/`_output`/`_FINISH_REASONS` are near-duplicate
   chat-completions logic**; they share the OpenAI-chat wire shape and justify a shared
   `ChatCompletionsAdapter`. OpenAI **Responses** is legitimately different (status-based finish, output
   items, reasoning items) and should stay separate.
4. **Token-usage consistency raises `InvalidResponseError`**, which *discards a valid completion* over
   non-load-bearing accounting metadata; the strictness is wildly asymmetric across providers and
   DeepSeek's exact `cache_hit + cache_miss == prompt_tokens` equality is fragile. Recommend degrading
   to `usage=None` rather than failing the whole response.
5. **Already good and worth keeping:** the typed error taxonomy + shared `_client/errors.py` base,
   the construction-cleanup discipline (`client.close()` + `add_note` on failed engine init), index
   de-duplication on multi-output responses, uniform `retry_after` clamping semantics, and the
   per-provider `MODEL_SPECS` normalization at import time.

Overall read: the **contracts and discipline are excellent**; the **factoring is not**. This layer is
~40% boilerplate that four copies keep in sync by hand, plus a capability matrix that describes
capability in data but enforces it in code. Both are cheap to fix pre-release and both reduce a real
drift-bug surface.

---

## Findings table

| ID | Finding | Confidence | Impact | Effort |
|----|---------|-----------|--------|--------|
| PA-1 | Error-mapping except-ladder duplicated verbatim ×4 → shared mapper on `_client.errors.*` | high | high | S–M |
| PA-2 | `LanguageModelSpec` matrix populated-but-unread; enforcement is parallel hardcoded checks that re-encode the matrix → data-driven capability gate | high | high | M |
| PA-3 | `__init__`/cleanup, `close`, `model`/`model_spec`, `_retry_after`, `_unsupported` are verbatim duplicates → shared base | high | med | S |
| PA-4 | Cerebras + DeepSeek `_parse_response`/`_output`/`_FINISH_REASONS` near-duplicate → shared `ChatCompletionsAdapter`; OpenAI Responses stays separate | high | med | M |
| PA-5 | Usage-consistency checks raise `InvalidResponseError` (discard valid completion) over accounting metadata; asymmetric + fragile | med | med | S–M |
| PA-6 | Symmetry nits: embedding engine has no `_build_request`; `_execution_metadata` redundant wrapper; `_retry_after` clamp duplicated ×4 & partly redundant with model validation; `loading.py` re-checks `model not in MODEL_SPECS` | med | low | S |

---

## Symmetry matrix (provider × behavior)

| Behavior | openai/responses | openai/embedding | cerebras/chat | deepseek/chat |
|---|---|---|---|---|
| `execute → _build_request → _validate → _parse` | yes | **no `_build_request`** (validate + inline build in `execute`) | yes | yes |
| `_validate` calls inside `_build_request` | yes | n/a | yes | yes |
| error except-ladder (Auth/RateLimit/Response/Transport/API) | identical | identical | identical | identical |
| `__init__` model-lookup + construction cleanup | identical | identical | identical | identical |
| `close` / `model` / `model_spec` | identical | identical | identical | identical |
| `_retry_after` clamp `>=0 and isfinite` | identical | identical | identical | identical |
| `_unsupported` | `-> None` | (raises inline) | `-> None` | `-> Never` |
| `_error_metadata` signature | `(meta)` non-opt + extra `_execution_metadata` wrapper | `(meta \| None)` | `(meta)` | `(meta)` |
| finish-reason mapping | bespoke status-based `_finish_reason` | n/a | `_FINISH_REASONS` dict | `_FINISH_REASONS` dict |
| consults matrix for message roles | no | n/a | no | no (hardcodes `DeveloperMessage` reject) |
| consults matrix for content types | via `model_spec.vision` | n/a | no | no (hardcodes image reject ×2) |
| consults matrix for response formats | no | n/a | no | no (hardcodes JsonSchema reject) |
| consults matrix `sampling_fields` | temperature/top_p only | n/a | no (matrix = all) | no (hardcodes seed/penalties/logit_bias) |
| consults matrix `reasoning_*` | efforts+summaries+fields | n/a | efforts+formats | efforts |
| usage inconsistency policy | raise `InvalidResponseError` | raise | raise (conditional) | raise (strictest, exact cache sum) |
| rate-limit metadata | none | none | `RateLimitMetadata` | none |

Reading the matrix: the top block (scaffolding) is **all "identical"** → dedup. The middle block
(matrix consultation) is **all "no" or partial** → the matrix is decoration, enforcement is hand-rolled.
The bottom block (wire-specific) legitimately diverges → keep as hooks.

---

## Detailed findings

### PA-1 — The provider error-mapping ladder is duplicated verbatim across all four engines

**What.** Every engine's `execute` wraps the client call in the same five-branch `except`, differing only
in (a) the client-errors module and (b) an English provider name in `msg`.

**Where.** `ResponsesEngine.execute` (`responses.py`), `EmbeddingEngine.execute` (`embedding.py`),
both `ChatCompletionsEngine.execute`. OpenAI form:

```python
except openai_errors.AuthError as error:
    metadata = self._error_metadata(error.metadata)
    msg = "OpenAI rejected authentication"
    raise AuthenticationError(msg, metadata=metadata) from error
except openai_errors.RateLimitError as error:
    ...
except openai_errors.ResponseError as error:
    ...
except openai_errors.TransportError as error:
    metadata = ErrorMetadata(provider=self.provider, model=self.model)
    msg = "OpenAI transport failed"
    raise TransportError(msg, metadata=metadata) from error
except openai_errors.APIError as error:
    metadata = self._error_metadata(error.metadata)
    msg = f"OpenAI API request failed with status {error.metadata.status_code}"
    raise ExecutionError(msg, metadata=metadata) from error
```

The Cerebras and DeepSeek versions are the same statements with `cerebras_errors.`/`deepseek_errors.`
and "Cerebras"/"DeepSeek". Crucially, **every provider's error classes derive from the shared
`providers/_client/errors.py` base** — e.g. `class AuthError(APIError, client_errors.AuthError)`,
`class APIError(client_errors.APIError, Error)`. So one mapper can catch the shared base classes.

**Why it matters.** ~35 lines × 4 = ~140 lines whose only job is to translate a fixed 5-class client
taxonomy into a fixed 5-class runtime taxonomy. Any change to the taxonomy (new error kind, new metadata
field) must be made in four places in lockstep. This is the seed-#5 duplication, at the engine layer.

**Proposed change.** A shared helper parametrized by the two provider-facing strings — `provider`
(the lowercase `ProviderId`) and a display name — mapping `client_errors.*` → `runtime.errors.*`:

```python
# providers/_engine/mapping.py (new, shared)
def map_execution[T](
    call: Callable[[], T], *, provider: ProviderId, display: str, model: str,
) -> T:
    try:
        return call()
    except client_errors.AuthError as e:
        raise AuthenticationError(f"{display} rejected authentication",
                                  metadata=_error_metadata(provider, model, e.metadata)) from e
    except client_errors.RateLimitError as e: ...
    except client_errors.ResponseError as e: ...
    except client_errors.TransportError as e:
        raise TransportError(f"{display} transport failed",
                             metadata=ErrorMetadata(provider=provider, model=model)) from e
    except client_errors.APIError as e: ...
```

Ordering is preserved (Auth/RateLimit are `APIError` subclasses caught first). `execute` becomes
`return self._parse_response(map_execution(lambda: self._client.create_..., ...))`.

**Feature impact:** `keeps-all`. **Confidence:** high. **Impact:** high. **Effort:** S–M.

---

### PA-2 — The `LanguageModelSpec` matrix is populated-but-unread; enforcement re-encodes it by hand

**What.** Each engine builds a rich `LanguageModelSpec` in `_normalized_model_spec`, then `_validate_request`
enforces capability with a long imperative sequence that **mostly ignores the matrix it just built** and
instead hardcodes the same facts a second time. This is the seed-#6 "parallel hardcoded checks" and the
brief's "spec says X but imperative check says Y" risk.

**Where — the fields are provably unread.** Grep across all of `symai/` for the membership fields returns
**only their construction sites and OpenAI's `reasoning_summaries` / Cerebras's `reasoning_formats`
reads** — nothing consults `message_roles`, `content_types`, or `response_formats` for enforcement:

```
$ grep -rn "message_roles\|content_types\|response_formats" symai/
symai/providers/.../*.py: … (construction only)
symai/runtime/models.py:  … (definition only)
```

Concrete re-encodings where the matrix already carries the answer:

- **DeepSeek sampling.** `_DEEPSEEK_SAMPLING_FIELDS` omits `SEED, FREQUENCY_PENALTY, PRESENCE_PENALTY,
  LOGIT_BIAS`. The matrix therefore already declares them unsupported — yet `_validate_request` rejects
  each again by hand:
  ```python
  unsupported_sampling = (("seed", sampling.seed),
                          ("frequency_penalty", sampling.frequency_penalty),
                          ("presence_penalty", sampling.presence_penalty))
  for field, value in unsupported_sampling:
      if value is not None:
          self._unsupported(f"DeepSeek does not support normalized {field}")
  if sampling.logit_bias:
      self._unsupported("DeepSeek does not support normalized logit bias")
  ```
- **OpenAI sampling.** `_OPENAI_NONREASONING_SAMPLING_FIELDS = (MAX_TOKENS, TOP_LOGPROBS, TEMPERATURE,
  TOP_P)` — the matrix omits `stop/seed/frequency/presence/logprobs/logit_bias`, and the code rejects
  exactly those six again by hand (`responses.py` lines ~223–234). OpenAI even mixes the two styles in one
  method: `SamplingField.TEMPERATURE not in self.model_spec.sampling_fields` (matrix-driven) sits three
  lines above hardcoded `if sampling.stop: self._unsupported(...)`.
- **DeepSeek roles/formats/content.** `message_roles` omits `DEVELOPER`, `response_formats` omits
  `JSON_SCHEMA`, `content_types` omits `IMAGE` — and each is *also* rejected by a hardcoded `isinstance`
  check (`DeveloperMessage`, `JsonSchemaResponseFormat`, `ImageContent`). The image reject is encoded
  **three** times: matrix `content_types`, `_validate_request`, and again in `_message`.

Today these agree, so there's no live bug — but nothing enforces the agreement. Editing the matrix (the
"declared capability") silently does nothing; editing a check silently disagrees with the matrix. The
matrix reads as the source of truth and is not.

**Why it matters.** Two representations of one fact is the classic drift trap, and here one of them is
inert data that *looks* authoritative. It also bloats each `_validate_request` to 40–75 lines of
near-mechanical checks.

**Proposed change.** A shared, data-driven **capability gate** in the adapter base that consults the
matrix as the single source of truth, plus a small static `SamplingField → (attr, is-set)` table:

```python
def _gate_capabilities(self, request: LanguageModelRequest) -> None:
    spec = self._model_spec
    for message in request.messages:
        if message.role not in spec.message_roles:
            self._unsupported(f"{self.display} does not support {message.role} messages")
        for part in message.content:
            if part.type not in spec.content_types:
                self._unsupported(f"{self.display} does not support {part.type} content")
    if request.response_format.type not in spec.response_formats:
        self._unsupported(...)
    for field in _SET_SAMPLING_FIELDS(request.sampling):     # fields the request actually sets
        if field not in spec.sampling_fields:
            self._unsupported(f"{self.display} does not support {field}")
    # …same pattern for reasoning_fields / efforts / summaries / formats…
```

Each engine then keeps **only its idiosyncratic constraints** as a `_validate_provider_specifics` hook —
the rules a boolean matrix genuinely can't express: DeepSeek's user-id regex/length, "temperature/top_p
ignored unless thinking disabled", "effort forbidden when thinking disabled", stop-count ≤16;
Cerebras's stop-count ≤4, `top_logprobs requires logprobs`, image-detail unsupported; OpenAI's
`max_tokens ≤ response_tokens`, "assistant reasoning input not accepted". Those are legitimately
divergent and should stay separate; everything above them collapses.

Note this also answers the brief's cross-provider question directly: **the three do not enforce the
same sampling fields, and correctly so** (Cerebras supports all; OpenAI-Responses supports few;
DeepSeek is in between). The smell is not the divergence — it's that the divergence is hand-coded when
it's already fully described by `sampling_fields`.

**Feature impact:** `keeps-all` (identical rejections, single source). **Confidence:** high.
**Impact:** high. **Effort:** M.

---

### PA-3 — Construction/lifecycle boilerplate is verbatim across engines

**What.** `__init__` (model lookup → `UnsupportedModelError`, adopt client, `try/except BaseException`
cleanup with `add_note`), `close`, the `model` and `model_spec` properties, `_retry_after`, and
`_unsupported` are byte-for-byte identical bar the provider string and the `Model` cast type.

**Where.** All four engines. The `__init__` cleanup block is the tell:

```python
except BaseException as error:
    try:
        client.close()
    except BaseException as cleanup_error:
        error.add_note(f"Engine construction cleanup failed: {cleanup_error!r}")
    raise
```

appears identically in `responses.py`, `embedding.py`, and both `chat_completions.py`.

**Why it matters.** This is disciplined, correct code (see "already good") — but four hand-maintained
copies. A future fix (e.g. also closing on a `model_spec` post-check failure) must land in four files.

**Proposed change.** A shared `Adapter` base (or two: `LanguageAdapter`, `EmbeddingAdapter`) holding
`__init__(client, model, *, model_specs, provider, display)`, `close`, `model`, `model_spec`,
`_retry_after`, `_unsupported`, `_error_metadata`. Engines supply `MODEL_SPECS`, the wire builders, and
`_parse_response`. This is the natural home for PA-1's mapper and PA-2's gate too.

**Feature impact:** `keeps-all`. **Confidence:** high. **Impact:** med. **Effort:** S.

---

### PA-4 — Cerebras and DeepSeek response normalization is near-duplicate; OpenAI Responses justifiably differs

**What.** The two chat-completions engines share the OpenAI-chat wire shape and their `_parse_response` /
`_output` / `_FINISH_REASONS` are the same algorithm.

**Where.** Both `_parse_response` implement: reject empty `choices` → per-choice index validation →
`seen_indices` dedup → build outputs → `outputs.sort(key=lambda o: o.index)` → wrap. Both `_output`
resolve a finish reason via a `_FINISH_REASONS` `MappingProxyType` and lift `content`/`reasoning`. The
only real deltas are the hook points:

| hook | cerebras | deepseek |
|---|---|---|
| index type | `int \| None` → `if choice.index is None` | `int` → `if choice.index < 0` |
| message optionality | `message is None` guard | message required |
| reasoning field | `message.reasoning` | `message.reasoning_content` |
| finish map | `{stop,length,content_filter,error}` | `{stop,length,content_filter,insufficient_system_resource→ERROR}` |

By contrast OpenAI Responses genuinely differs — status-based `_finish_reason` (`ResponseStatus` +
`incomplete_details.reason`), `CompactionOutput`/`ReasoningOutput`/`OutputMessage` item walking, and a
"reasoning requires exactly one assistant message" rule. That divergence is real and should be preserved.

**Why it matters.** Two copies of the choices-dedup-sort loop and the finish-reason lookup drift
independently (they already differ subtly in the index guard for no capability reason).

**Proposed change.** A `ChatCompletionsAdapter(LanguageAdapter)` base implementing the choices loop and a
`_normalize_choice` template, parametrized by `finish_reasons: Mapping[str, FinishReason]` and small hooks
(`_extract_reasoning`, index/message optionality). Cerebras and DeepSeek subclass it; OpenAI Responses does
**not**. Keep OpenAI's Responses engine standalone — flag it explicitly as justified divergence.

**Feature impact:** `keeps-all`. **Confidence:** high. **Impact:** med. **Effort:** M.

---

### PA-5 — Usage-consistency failures discard a valid completion; policy is asymmetric and locally fragile

**What.** All engines treat provider token-accounting inconsistency as a hard `InvalidResponseError`,
which propagates out of `execute` and **throws away an otherwise-valid, already-generated completion**.

**Where.** OpenAI `_usage`:

```python
if (usage.total_tokens != usage.input_tokens + usage.output_tokens
        or usage.input_tokens_details.cached_tokens > usage.input_tokens
        or usage.output_tokens_details.reasoning_tokens > usage.output_tokens):
    raise InvalidResponseError("OpenAI token usage was inconsistent", ...)
```

DeepSeek `_usage` is the strictest and most brittle:

```python
if (cache_hit is not None and cache_miss is not None
        and cache_hit + cache_miss != usage.prompt_tokens):
    raise InvalidResponseError("DeepSeek cache token usage was inconsistent", ...)
```

Cerebras only checks totals when all three are present, then adds cached/reasoning/prediction bounds.
So the three providers apply materially different strictness to the same class of metadata.

**Why it matters.** `usage` is billing/telemetry, not the answer. Failing the whole request because the
provider's *arithmetic* is off (or because they added a token category the exact-equality check didn't
anticipate — the DeepSeek `cache_hit + cache_miss == prompt_tokens` case) converts a successful call into
a user-visible error and loses the generated text. That is a poor trade for non-load-bearing data, and
the exact-sum equality is precisely the kind of check that breaks when a provider evolves its accounting.

This is a **deliberate strictness stance** (the whole layer favors fail-loud), so it's a design question,
not an unambiguous bug. But usage is the one place where "drop the bad metadata, keep the good answer"
clearly beats "discard everything."

**Proposed change.** Two options, in order of preference:
1. On inconsistency, **degrade**: return `usage=None` (optionally `logger.warning`), keep the response.
   Reserve `InvalidResponseError` for cases where the *content* can't be normalized.
2. If strictness is intentional, at least **relax exact-equality to bounded checks** (`cache_hit +
   cache_miss <= prompt_tokens`, not `==`) so additive provider changes don't fail valid calls, and
   unify the policy across the three engines so "inconsistent" means the same thing everywhere.

**Feature impact:** `drops-minimal` (loses the guarantee that returned usage is arithmetically perfect;
gains not discarding valid completions). **Confidence:** med (depends on intended strictness).
**Impact:** med. **Effort:** S–M.

---

### PA-6 — Symmetry nits (low impact, cheap)

**What / Where.**
- **Embedding engine breaks the `_build_request` shape.** `EmbeddingEngine.execute` calls
  `self._validate_request(request)` then inlines `embeddings_api.CreateEmbeddingRequest(...)` — there is
  **no `_build_request`** method, unlike all three language engines. Minor, but it's the odd one out in
  the "execute → build → validate → parse" pattern the brief asks about.
- **`_execution_metadata` is a redundant indirection.** In `responses.py` it is exactly
  `return self._error_metadata(response.metadata)`; Cerebras/DeepSeek inline that call. Drop the wrapper.
- **`_error_metadata` signatures diverge** for no strong reason: `responses.py` takes non-optional
  metadata (+ the wrapper above), `embedding.py` takes `metadata | None`. A shared base (PA-3) would pick
  one shape.
- **`_retry_after` clamp duplicated ×4 and partly redundant with model validation.** The clamp
  (`value >= 0 and isfinite(value)`) is identical in all four engines, and both `ResponseMetadata.retry_after`
  and `ErrorMetadata.retry_after` are already `NonNegativeFiniteFloat | None` (ge=0, allow_inf_nan=False).
  The clamp's *only* remaining job is to silently coerce a bad transport value to `None` instead of raising
  at model construction — a real behavior, but it belongs in one shared place, not four.
- **`loading.py` re-checks `parsed.model not in MODEL_SPECS`** and raises `UnsupportedModelError` with the
  same message the engine `__init__` already raises (and the engine also cleans up the client on that
  path). It's a fail-fast-before-client-construction optimization, but it duplicates the model-support
  contract in two layers with two copies of the message string.

**Why it matters.** Individually trivial; together they're the residue that a shared base (PA-3) erases
for free.

**Proposed change.** Fold all of these into the shared adapter base; give the embedding adapter a
`_build_request` for shape parity or explicitly document embeddings as the intentionally-simpler path.

**Feature impact:** `keeps-all`. **Confidence:** med. **Impact:** low. **Effort:** S.

---

## What is already good (keep it)

- **Typed, layered error taxonomy.** `providers/_client/errors.py` gives every client a shared 5-class
  base (`ClientError → Transport/Response/API → Auth/RateLimit`), and each provider subclasses it while
  the runtime keeps its own neutral hierarchy (`ExecutionError` + `ErrorMetadata`). The engine is the
  single crossing point. This is exactly the client/engine layering the project wants — and it's what
  makes PA-1's shared mapper trivially safe.
- **Construction-cleanup discipline.** Both the clients and the engines close the resource they adopted on
  a failed init and `add_note` if cleanup itself fails, without swallowing the original error. This is
  correct and rare; keep it (just centralize it — PA-3).
- **Multi-output index integrity.** Both chat engines reject duplicate/invalid choice indices and sort by
  index before returning; OpenAI enforces "reasoning ⇒ exactly one assistant message". Good defensive
  normalization of untrusted provider payloads.
- **Frozen normalized contracts.** `runtime/models.py` models are `frozen=True, strict=True,
  extra="forbid"` with real validators (unique metadata keys, unique logit-bias tokens, content-or-reasoning
  invariants). The engines lean on these instead of re-validating — good separation.
- **Per-provider `MODEL_SPECS` normalization at import.** Translating each provider's native spec into the
  normalized `LanguageModelSpec`/`EmbeddingModelSpec` once, at module load, behind a `MappingProxyType`, is
  clean. The problem (PA-2) is not the normalization — it's that the normalized result isn't then used to
  drive enforcement.

---

## Recommended shared skeleton (net shape)

```
providers/_engine/
  base.py        # Adapter[SpecT]: __init__/cleanup, close, model/model_spec,
                 #   _retry_after, _unsupported, _error_metadata
  mapping.py     # map_execution(): client_errors.* -> runtime.errors.*  (PA-1)
  gate.py        # capability gate driven by LanguageModelSpec              (PA-2)
  chat.py        # ChatCompletionsAdapter(LanguageAdapter): choices loop,
                 #   _FINISH_REASONS-driven finish mapping, reasoning hook   (PA-4)

providers/openai/engines/responses.py   # standalone LanguageAdapter (Responses wire is different)
providers/openai/engines/embedding.py   # EmbeddingAdapter
providers/cerebras/engines/...          # ChatCompletionsAdapter + _validate_provider_specifics hook
providers/deepseek/engines/...          # ChatCompletionsAdapter + _validate_provider_specifics hook
```

Provider-specific hook points that **must** remain per-engine (divergence is real): the wire request
builder (`_build_request`/`_message`/`_response_format`/`_reasoning_config`), the idiosyncratic
constraint validator, the finish-reason map, the reasoning field name, and — for OpenAI — the entire
Responses-shaped parse path.

Net effect: the four engines shrink to roughly {`MODEL_SPECS`, request builder, provider-specific
constraints, response hooks}; the ~40% that is scaffolding + re-encoded capability moves to one place and
the `LanguageModelSpec` matrix becomes the authoritative capability source it was designed to be.
