# r1-06 — Contract / type model coherence (`runtime/models.py`, `runtime/errors.py`)

**Lens:** Is the normalized contract minimal-yet-complete for the retained features? For
every model/field: is it PRODUCED and CONSUMED somewhere? Recommend cut/merge/relax while
keeping provider fidelity + type safety.

All findings verified against live code at read time (models.py ~447 LOC). Line numbers are
approximate; anchors are symbol names + quoted snippets. Prior-audit IDs (CON-01/02/03) are
re-verified against current code and labeled.

---

## Executive summary

1. **Write-only logprobs (CON-02, STILL-OPEN).** `SamplingConfig.logprobs`/`top_logprobs`/
   `logit_bias` are forwarded to all providers, provider *clients* even parse the returned
   logprobs, but `LanguageModelOutput` has **no** field to hold them and no engine maps them
   back. The contract can request logprobs and can never express the answer.
2. **Half the capability matrix is dead (CON-01, PARTIALLY-OPEN).** `LanguageModelSpec.`
   `context_tokens`/`message_roles`/`content_types`/`response_formats` and
   `EmbeddingModelSpec.context_tokens` have **zero reads** anywhere; the `MessageRole`/
   `ContentType`/`ResponseFormatType` enums exist only to fill them. `Runtime` never exposes
   `model_spec` at all — the "contract" is engine-private validation data, and half of it is unread.
3. **N-output machinery is permanently dead (CON-03, STILL-OPEN).** No `n`/`candidates` field
   exists on any request; providers always return one choice; yet `outputs` is a tuple with
   per-output `index`, dedup+sort logic, and a `decode_output(output_index=0)` scan.
4. **The `JsonObject`/`JsonArray`/`JsonEntry` AST is never produced by the library** and is
   round-tripped straight back to a builtin `dict` that engines immediately cast to
   `pydantic.JsonValue` — the wire type the code already trusts. ~70 LOC of AST + `parse`/
   `to_builtin` for a property (deep-freeze) nothing consumes.
5. **What's right:** the strict/tolerant split is correct (inbound provider JSON is parsed by
   `TolerantModel`, not `FrozenModel`); `TokenUsage`'s 9 fields each have ≥1 real producer;
   the `AssistantMessage`/`AssistantOutputMessage` split is a genuine input/output lifecycle
   distinction; the `LanguageModelOutput` content-or-reasoning-or-refusal-or-filter validator
   is coherent.

Overall read: the *message/request/response* core is sound and provider-faithful. The rot is
concentrated in three write-only or unread sub-surfaces (logprobs, the spec matrix, N-output)
plus one over-engineered value type (the JSON AST). Cutting them removes ~150 LOC and several
public types without touching any feature a request can actually exercise.

---

## Findings table

| ID | Finding | Feature impact | Conf | Impact | Effort |
|----|---------|----------------|------|--------|--------|
| C1 | logprobs request↔response coherence hole (write-only) | drops-minimal / decide | high | med | S–M |
| C2 | Spec matrix: 4 fields + 3 enums populated-but-unread; not exposed | drops-minimal | high | med | M |
| C3 | N-output tuple/index/dedup/sort with no `n` request field | drops-minimal | high | low | S |
| C4 | `JsonObject`/`JsonArray`/`JsonEntry` AST never produced; gratuitous | drops-minimal | med | med | M |
| C5 | `AssistantMessage` vs `AssistantOutputMessage` — keep, minor DRY | keeps-all | high | low | S |
| C6 | Metadata surface never internally consumed; `RateLimitMetadata` single-provider | keeps-all | high | low | — |
| C7 | POSITIVE: strict/tolerant boundary is placed correctly | keeps-all | high | — | — |

---

## C1 — logprobs are sent and parsed but the normalized response cannot hold them (CON-02, STILL-OPEN)

**What.** `SamplingConfig` carries three live request fields:

```python
logprobs: bool | None = None
top_logprobs: int | None = Field(default=None, ge=0, le=20)
logit_bias: tuple[LogitBias, ...] = ()
```

They are forwarded to providers: Cerebras (`logprobs=sampling.logprobs`, `top_logprobs=...`,
`logit_bias={bias.token: bias.value ...}`), DeepSeek (`logprobs`, `top_logprobs`), OpenAI
(`top_logprobs`). The provider **clients even parse the returned logprobs** —
`openai/client/responses.py` has `TokenLogprob`/`TopLogprob` on `OutputText.logprobs`,
`cerebras/client/chat.py` has `logprobs`/`reasoning_logprobs` on the response message,
`deepseek/client/chat.py` has a `Logprobs` model on `Choice`.

**But `LanguageModelOutput` has no logprobs field:**

```python
class LanguageModelOutput(FrozenModel):
    index: int = Field(ge=0)
    message: AssistantOutputMessage
    refusal: str | None = Field(default=None, min_length=1)
    finish_reason: FinishReason
```

Every `_output`/`_language_output` builder ignores `choice.logprobs`. Grep confirms no engine
reads `choice.logprobs` into a normalized field. The parsed client-side logprobs are dropped.

**Why it matters.** The public type system advertises a capability (request per-token
logprobs) that the response half cannot express — a request/response coherence hole. Providers
may also *bill* differently when `logprobs=true`, so this is not free to send. `logit_bias`
additionally carries the `LogitBias` model + `validate_unique_logit_bias_tokens` validator that
exist only to serialize an input no output ever reflects.

**Proposed change.** Pick one:
- **Close the loop** (if logprobs is an intended feature): add e.g.
  `logprobs: tuple[TokenLogprob, ...] = ()` to `LanguageModelOutput` and map it in the three
  `_output` builders. Then the request fields earn their keep.
- **Cut the request side** (if not intended): drop `logprobs`/`top_logprobs`/`logit_bias` from
  `SamplingConfig`, delete `LogitBias` + `validate_unique_logit_bias_tokens` +
  `SamplingField.LOGPROBS/TOP_LOGPROBS/LOGIT_BIAS`, and the client-side logprobs DTOs.

Do not keep the current half-state. Given "keep intended features, minimal loss acceptable" and
that nothing in ops/decoding consumes logprobs, **cutting is the lower-risk default** unless the
product wants logprobs as a first-class result.

**Feature impact:** drops-minimal (no consumer today) — but this is the one finding where the
intended-feature question genuinely gates the direction; flag for the owner. **Conf:** high.
**Impact:** med. **Effort:** S (cut) / M (close loop).

---

## C2 — `LanguageModelSpec`/`EmbeddingModelSpec`: four fields + three enums are populated but never read; spec isn't exposed (CON-01, PARTIALLY-OPEN)

**What.** Read-count across all engines + runtime (excluding the models.py definition and the
`_normalized_model_spec` builder that reads the *client* spec):

| `LanguageModelSpec` field | `model_spec.<field>` reads | verdict |
|---|---|---|
| `response_tokens` | 6 (max_tokens validation) | **alive** |
| `reasoning_efforts` | 3 | alive |
| `reasoning_fields` | 2 | alive |
| `sampling_fields` | 2 | alive |
| `vision` | 3 | alive |
| `reasoning_summaries` | 1 | alive |
| `reasoning_formats` | 1 | alive |
| `context_tokens` | **0** | **dead** |
| `message_roles` | **0** | **dead** |
| `content_types` | **0** | **dead** |
| `response_formats` | **0** | **dead** |

`EmbeddingModelSpec.context_tokens` — populated (`context_tokens=spec.context_tokens`) but only
`.dimensions` is ever read → **dead**.

The three enums whose sole job is to populate the dead fields are themselves dead surface:
`MessageRole` → `message_roles`, `ContentType` → `content_types`, `ResponseFormatType` →
`response_formats`. Grep shows these enums appear *only* inside `_normalized_model_spec`
building the unread tuples (e.g. `_ALL_MESSAGE_ROLES = tuple(MessageRole)`,
`content_types=(ContentType.TEXT, ContentType.IMAGE)`), nowhere else.

Critically, **`Runtime` never exposes `model_spec`** (0 hits for `model_spec`/`LanguageModelSpec`
in `runtime/runtime.py` and `runtime/engines.py`). So the spec is not a consumed cross-provider
contract at all — it is an engine-private validation table that each engine builds and then reads
back *some* of its own fields from inside `_validate_request`.

**Why it matters.** A normalized Pydantic model in the shared `runtime/models.py` reads as a
public capability contract. In reality half its fields have no reader and the object never
crosses the engine boundary. Modeling engine-private state as a shared contract is misleading and
drags three otherwise-unused enums into the public surface.

**Proposed change.** Decide what the spec *is*:
- **Make it a real introspection contract:** expose `model_spec` through `Runtime`
  (e.g. a `capabilities(engine)` accessor) so callers can pre-flight. Then `message_roles`/
  `content_types`/`response_formats`/`context_tokens` gain a consumer and the enums are justified.
- **Or demote to engine-private data + drop the dead fields:** remove `context_tokens`,
  `message_roles`, `content_types`, `response_formats` from `LanguageModelSpec` and
  `context_tokens` from `EmbeddingModelSpec`; delete `MessageRole`/`ContentType`/
  `ResponseFormatType`. Keep only the fields `_validate_request` actually reads.

Either is coherent; the status quo (shared-contract shape, engine-private use, half unread) is
not. This overlaps the capability-matrix lens — deferring the "expose vs demote" call there, but
the **data-model verdict is unambiguous: four fields + three enums have zero readers today.**

**Feature impact:** drops-minimal (no behavior depends on the dead fields). **Conf:** high.
**Impact:** med. **Effort:** M.

---

## C3 — N-output modeling with no request that can ask for N>1 (CON-03, STILL-OPEN)

**What.** `LanguageModelResponse.outputs: tuple[LanguageModelOutput, ...] = Field(min_length=1)`
plus `LanguageModelOutput.index: int`. The chat engines carefully dedup and sort choices:

```python
seen_indices: set[int] = set()
...
if choice.index in seen_indices: raise InvalidResponseError("... duplicate choice indices")
seen_indices.add(choice.index)
outputs.append(self._output(choice, error_metadata))
outputs.sort(key=lambda output: output.index)
```

And `decode_output(..., output_index: int = 0)` scans `_output_text` for a matching index. But
**there is no `n`/`candidates`/`best_of` field on `LanguageModelRequest`, `SamplingConfig`, or
any provider client request** (grep across `providers/*/client/*.py` and models.py finds none).
`ops/primitives.py` always calls `decode_output(response, decoder)` → `output_index=0`. So every
response has exactly one output at index 0; the tuple, the `index` field, the dedup, the sort,
and the `output_index` parameter are all permanently exercising the single-element path.

**Why it matters.** Dead defensive machinery across four files (models, two engines, decoding)
that the type system presents as a real many-outputs capability.

**Proposed change.** Either add `n: int = 1` to `SamplingConfig`/request and thread it through
(making the machinery real), or collapse to a single output: `LanguageModelResponse.output:
LanguageModelOutput`, drop `index`/dedup/sort/`output_index`. Given nothing wants N>1 and OpenAI's
own path already joins parts into one message, **collapse is simpler**; re-introduce `n` only if
multi-candidate becomes a product goal.

**Feature impact:** drops-minimal (no request can produce N>1 today). **Conf:** high.
**Impact:** low. **Effort:** S.

---

## C4 — `JsonObject`/`JsonArray`/`JsonEntry` AST is never produced by the library and round-trips to builtins the engines already accept

**What.** The bespoke JSON "AST" (`JsonObject`, `JsonArray`, `JsonEntry`, `JsonScalar`,
`JsonValue`) plus `JsonObject.parse`, `to_builtin`, `_parse_json_value`, `_json_value_to_builtin`,
`validate_unique_keys`, and the three `model_rebuild(...)` calls exists solely to type the
`json_schema` field of `JsonSchemaResponseFormat`. Two problems:

1. **No producer inside the library.** `operations.py` (`language_request`, `image_request`)
   never sets `response_format` — it always defaults to `TextResponseFormat()`. No op emits a
   `JsonSchemaResponseFormat`. Grep finds `JsonSchemaResponseFormat`/`JsonObject.parse`/`JsonObject(`
   constructed nowhere in `symai/` (only `isinstance` checks + `to_builtin` consumers in engines,
   and constructions in tests). Structured output is a user-hand-assembled request path only.

2. **The AST is immediately discarded at the boundary.** Both consuming engines do
   `cast("JsonValue", response_format.json_schema.to_builtin())` where `JsonValue` is
   **`pydantic.JsonValue`** (imported at the top of both engine files). So the code builds a
   frozen tuple-of-entries AST, then flattens it back to a builtin `dict`/`list`, then hands it
   to a client field typed as the pydantic JSON type the framework already trusts everywhere else.

The only property the AST adds over a validated `pydantic.JsonValue` is **deep immutability**
(tuples all the way down) — and nothing consumes that: the request is built once and executed
once; nothing hashes or caches it. The `validate_unique_keys` guard is near-vacuous from the
`parse(Mapping)` path (a `Mapping` cannot have duplicate keys); it only guards hand-built
`entries` tuples.

**Why it matters.** ~70 LOC and three public models (`JsonObject`/`JsonArray`/`JsonEntry`) plus
the fragile forward-ref `model_rebuild` dance, all to carry a schema that is converted to a plain
dict before it leaves the process, for a request the library never issues itself.

**Proposed change.** Type the field as pydantic's own JSON value:

```python
class JsonSchemaResponseFormat(FrozenModel):
    type: Literal["json_schema"] = "json_schema"
    name: str = Field(min_length=1)
    json_schema: JsonValue            # pydantic.JsonValue (validated, JSON-serializable)
    description: str | None = None
    strict: bool
```

Delete `JsonObject`/`JsonArray`/`JsonEntry`/`JsonScalar`/`_parse_json_value`/
`_json_value_to_builtin`/`to_builtin`/`parse`/the `model_rebuild` calls; engines pass
`response_format.json_schema` straight through (no `to_builtin`). **Tradeoff to weigh honestly:**
a `dict`/`list` inside `json_schema` is mutable, so the request is only shallow-frozen there. If
a hard deep-freeze/hashable-request invariant is required, keep an immutable representation — but
prefer a single `frozendict`-style wrapper or a canonical JSON *string* over a three-model AST.
Since no consumer relies on deep-freeze today, the validated-`JsonValue` form is the right size.

**Feature impact:** drops-minimal (keeps json_schema requests; loses only unused deep-immutability
of the schema subtree). **Conf:** med (gated on whether deep-freeze is a stated invariant).
**Impact:** med. **Effort:** M.

---

## C5 — `AssistantMessage` (validated) vs `AssistantOutputMessage` (unvalidated): keep the split, DRY the fields

**What.** Identical fields (`role`, `content: tuple[TextContent, ...] = ()`, `reasoning:
TextContent | None = None`). The only difference:

```python
class AssistantMessage(FrozenModel):        # INPUT (in the Message union)
    ...
    @model_validator(mode="after")
    def validate_content_or_reasoning(self): ...  # must be non-empty

class AssistantOutputMessage(FrozenModel):  # OUTPUT (in LanguageModelOutput)
    ...                                            # may be empty
```

**Why the split is justified (keep it).** An *input* assistant turn (conversation history the
caller supplies) must carry something — an empty assistant turn is meaningless. An *output*
message legitimately can be empty: on a content-filter block or refusal-only response the model
returns no content and no reasoning, and that emptiness is validated one level up by
`LanguageModelOutput.validate_content_reasoning_or_refusal` (which permits empty when
`finish_reason is CONTENT_FILTER` or a `refusal` is present). If the output reused the validated
`AssistantMessage`, construction would fail *before* reaching the output-level validator on
exactly those content-filter/refusal cases. So this is a real input/output capability distinction,
matching the semantic-types guideline ("encodes a lifecycle transition or capability").

Note the runtime `AssistantMessage` is never built by the library itself (`operations.py` emits
only `SystemMessage`/`UserMessage`); it is a public multi-turn-history input type, consumed via
`isinstance` in the OpenAI/Cerebras engines. Alive.

**Proposed change (minor).** Remove the field duplication by making the unvalidated form the base
and the validated form a subclass that only adds the validator — or share a mixin. Purely a DRY
tidy; the two-type surface should remain.

**Feature impact:** keeps-all. **Conf:** high. **Impact:** low. **Effort:** S.

---

## C6 — Metadata surface is never internally consumed; `RateLimitMetadata` is single-provider; `TokenUsage` fields are all producer-backed

**What.** Nothing in `ops/`, `decoding.py`, `function.py`, or `runtime/runtime.py` reads
`ResponseMetadata`, `TokenUsage`, or `RateLimitMetadata` (grep: 0 non-provider, non-`__init__`
consumers). Decoding reads only `output.text`. These models are pure public return data.

Producer matrix I verified:

**`TokenUsage` (9 fields)** — each has ≥1 real producer, so none is strictly dead:
| field | OpenAI | Cerebras | DeepSeek |
|---|---|---|---|
| prompt/completion/total_tokens | ✓ | ✓ | ✓ |
| cached_prompt_tokens | ✓ | ✓ | ✓ |
| reasoning_tokens | ✓ | ✓ | ✓ |
| cache_miss_prompt_tokens | — | — | ✓ only |
| image_tokens | — | ✓ only | — |
| accepted_prediction_tokens | — | ✓ only | — |
| rejected_prediction_tokens | — | ✓ only | — |

**`RateLimitMetadata` (6 fields)** — populated **only by Cerebras** (`_rate_limit`); OpenAI and
DeepSeek never set `rate_limit`, so it stays `None` on two of three providers.

**Why it matters / verdict.** This is honest provider-fidelity data, so **keep it** — the
single-provider fields reflect genuine provider differences, not modeling waste, and library users
(not the framework) are the consumers. Two caveats worth recording: (a) the framework derives
nothing from usage/rate-limit, so if the goal were "minimal contract for what the code uses," this
whole surface is out of scope; keep it only because it is public API. (b) `RateLimitMetadata`
being 1-of-3 providers means callers must treat it as best-effort. No change recommended.

**Feature impact:** keeps-all. **Conf:** high.

---

## C7 — POSITIVE: the strict / tolerant boundary is placed correctly (answers the `FrozenModel` seed)

**What.** The seed asks whether `FrozenModel` (`strict=True, extra="forbid"`) is the right base
for *inbound provider data* given providers add fields. **It is — because `FrozenModel` never
parses raw provider JSON.** Inbound provider bodies are parsed by the client DTO layer, which
already makes the right per-direction choice:

- **Request DTOs we control** → `StrictModel` (`extra="forbid"`): e.g.
  `CreateResponseRequest`, `CreateChatCompletionRequest`, `CreateEmbeddingRequest`.
- **Response DTOs from the provider** → `TolerantModel` (`extra="allow"`): e.g. OpenAI
  `Response`/`Usage`/`OutputMessage`, Cerebras `ChatCompletion`/`Choice`/`Usage`, DeepSeek
  `ChatCompletion`/`Choice`/`Usage`, embedding `EmbeddingList`/`EmbeddingData`.

By the time an engine constructs a `runtime/models.py` `FrozenModel`, the data is already curated
field-by-field by the engine from `TolerantModel`-parsed objects — so `strict=True` (no coercion)
+ `extra="forbid"` is exactly right for internally-assembled normalized models, and would be
*wrong* only if `FrozenModel` were fed raw JSON, which it never is. **Keep this split as-is;** it
is the model layer's strongest design decision. (One consistency nit: `runtime/models.py`
redefines its own `FrozenModel` identical to `providers/_client/models.py::StrictModel` — the two
could share one base, but they live in different layers so the duplication is defensible.)

---

## Prior-finding status recap

- **CON-01** (spec fields populated-but-unread): **PARTIALLY-OPEN** — verified still dead:
  `context_tokens`, `message_roles`, `content_types`, `response_formats` (Language) +
  `context_tokens` (Embedding). The reasoning/sampling/vision/response_tokens fields the prior
  audit lumped in *are* read now. See C2.
- **CON-02** (logprobs write-only): **STILL-OPEN**, unchanged. See C1.
- **CON-03** (N-output dead machinery): **STILL-OPEN**, unchanged. See C3.

---

## Net recommendation

Cut, in priority order: **C1** (resolve the logprobs half-state — cut or close), **C2** (drop 4
dead spec fields + 3 enums, or expose the spec), **C3** (collapse N-output), **C4** (replace the
JSON AST with validated `pydantic.JsonValue`). Keep, unchanged: the message/request/response core,
the strict/tolerant boundary (C7), `TokenUsage`/`RateLimitMetadata` as public provider data (C6),
and the `AssistantMessage`/`AssistantOutputMessage` split (C5, modulo a DRY tidy). Estimated
removal: ~150 LOC and ~6 public types, zero features a request can currently exercise.
