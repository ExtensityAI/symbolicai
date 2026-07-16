# r3-m3a — Adversarial verification of load-bearing claims

> **Historical snapshot terminology.** Executed probes below intentionally use the API present at
> commit `84f703b`. `EngineSpec` and configured `default_*` fields are not the target API; they are
> superseded by `EngineConfig` and sole-engine-only unnamed selection.

**Round 3 / META.** Goal: try HARD to *refute* the highest-impact claims prior reports make,
against live HEAD `84f703b`. Read-only. Verdict per claim uses live grep counts, byte diffs,
and executed Python probes (`uv run python`), not the reports' own numbers. I did **not** read
any `r3-*` file; every result below is independently re-derived.

Default stance was skepticism (assume each claim wrong until code proves it). The honest
outcome: the load-bearing claims are unusually well-supported. Of 11 targets, **10 survive
fully**; **1 survives only in corrected form** (an over-broad "all three providers" phrasing).

---

## Confidence-weighted scoreboard

| # | Claim (abbreviated) | Verdict | Conf |
|---|---------------------|---------|------|
| 1 | `LanguageModelSpec` `message_roles`/`content_types`/`response_formats`/`context_tokens` have 0 enforcement reads; 3 enums orphaned | **CONFIRMED** | high |
| 2 | Provider `client/transport.py`+`headers.py` identical across providers except provider name | **PARTIAL** (openai≡deepseek yes; cerebras is a strict superset) | high |
| 3 | cerebras vs deepseek `_client.py` ~83%, `chat_completions.py` ~55% identical | **CONFIRMED** (83% exact; 53–56%) | high |
| 4 | Multimodal image path wired end-to-end + importable; deleting drops a real feature | **CONFIRMED** | high |
| 5 | `JsonObject`/`JsonArray`/`JsonEntry` never constructed in production; round-trips to `pydantic.JsonValue` | **CONFIRMED** | high |
| 6 | DeepSeek serializes WITHOUT `by_alias=True` while others use it; harmless (0 aliased fields) | **CONFIRMED** | high |
| 7 | `_normalize_text` strips wrapping single-quotes for EVERY decoder incl. `TextDecoder` | **CONFIRMED** | high |
| 8 | `Runtime.__init__` accepts `" chat "` alias that `RuntimeConfig` rejects | **CONFIRMED** | high |
| 9 | `PydanticDecoder` functionally identical to `TypeAdapterDecoder` for a `BaseModel` | **CONFIRMED** | high |
| 10 | NO `providers/*/client/` module imports `symai.runtime` | **CONFIRMED** | high |
| 11 | N-output: no request field can make a provider return >1 output | **CONFIRMED** | high |

**Survivors: 10 fully CONFIRMED + 1 PARTIAL (corrected). 0 refuted.** Every consequential
finding the prior rounds built on holds against live code.

---

## Claim 1 — dead capability-matrix fields + orphan enums — CONFIRMED

**Restated.** `LanguageModelSpec.message_roles`, `.content_types`, `.response_formats`, and the
normalized `.context_tokens` (+ `EmbeddingModelSpec.context_tokens`) have zero enforcement reads;
`MessageRole`/`ContentType`/`ResponseFormatType` exist only to populate them.

**Method.** `rg` for each field name and each enum across all of `symai/`, plus an adversarial
sweep for dynamic access (`getattr(...spec`, `model_spec.<field>`).

**Evidence.** Every hit is a *definition* (`models.py:429–433`) or a *construction* site in an
engine's `_normalized_model_spec` — never a read:
```
message_roles   → models.py:431 (def) + 3 engine ctor sites (message_roles=…)
content_types   → models.py:432 (def) + 3 engine ctor sites
response_formats→ models.py:433 (def) + 3 engine ctor sites
context_tokens  → models.py:429,443 (def); every read is `context_tokens=spec.context_tokens`
                  i.e. reads the CLIENT spec, writes the normalized one — normalized is never read
```
The dynamic-access sweep returned `NONE (no reads)`. The three enums appear only as
`_ALL_MESSAGE_ROLES = tuple(MessageRole)`, `content_types=(ContentType.TEXT, …)`,
`_DEEPSEEK_MESSAGE_ROLES = (MessageRole.SYSTEM, …)`, `_DEEPSEEK_RESPONSE_FORMATS =
(ResponseFormatType.TEXT, ResponseFormatType.JSON_OBJECT)` — construction only. Image support is
enforced through the *separate* `model_spec.vision` bool (`responses.py:173
if has_image and not self.model_spec.vision`), not through `content_types` — the double-encoding
the reports flag.

**Refutation attempt.** Looked for a serialization/introspection consumer (`model_dump` of the
spec, a `Runtime.capabilities()` accessor, a test that enforces via these fields). None exists;
`Runtime` never exposes `model_spec`. Claim stands.

---

## Claim 2 — transport.py/headers.py identical except provider name — PARTIAL

**Restated.** The per-provider `client/transport.py` and `client/headers.py` are identical across
all providers except the provider name.

**Method.** Byte `diff` of openai↔deepseek and openai↔cerebras for both files.

**Evidence.**
- `transport.py` openai↔deepseek: **only the docstring differs** (`"…OpenAI client."` vs
  `"…DeepSeek client."`). Byte-identical otherwise.
- `headers.py` openai↔deepseek: **only the import path differs**
  (`openai.client.transport` vs `deepseek.client.transport`).
- **cerebras is NOT identical** — it is a *strict additive superset*: `transport.py` adds a
  `RateLimitState` model + `rate_limit: RateLimitState` field; `headers.py` adds six
  `x-ratelimit-*` constants and populates `rate_limit` in `extract_response_metadata`.

**Verdict.** The literal "identical across *providers*" is **false for cerebras**. The claim is
true only for the openai↔deepseek pair. The *underlying* duplication finding (r1-02 D3) is
correctly stated there ("openai≡deepseek verbatim; cerebras a strict superset") and survives —
the consolidation opportunity is real (openai+deepseek re-export a base; cerebras subclasses).

**Corrected claim.** "`transport.py`/`headers.py` are byte-identical between openai and deepseek
(mod docstring/import path); cerebras is a strict additive superset adding rate-limit state."

---

## Claim 3 — cerebras vs deepseek `_client.py` ~83%, `chat_completions.py` ~55% — CONFIRMED

**Method.** `difflib.SequenceMatcher` longest-common-block line count over the two file pairs.

**Evidence.**
```
_client.py            : 119 vs 113 lines; LCS matched = 99 → 83% of cerebras (exact match to claim)
chat_completions.py   : 459 vs 435 lines; LCS matched = 242 → 53% of cerebras / 56% of deepseek
                        difflib ratio 54%
```
83% is exact; "~55%" is accurate (53–56% depending on denominator). Claim stands.

---

## Claim 4 — multimodal image path wired end-to-end + importable — CONFIRMED

**Restated.** `ImageContent`/`image_request`/`data_uri`/`vision` form a real, importable,
end-to-end vision feature; deleting them drops a real capability.

**Method.** Trace `image_request` → `ImageContent` → wire serialization in each engine; confirm
importability and gating.

**Evidence.**
- `operations.py:45 image_request(...)` builds a `UserMessage` containing
  `ImageContent(url=image_url, detail=detail)`; `operations.py:71 data_uri(...)` base64-encodes to
  a `data:` URI. Both are module-level, importable (`from symai.operations import image_request`);
  `operations.py` already imports cleanly (`function.py` imports `language_request` from it).
- **OpenAI wire path (live, not dead):** `responses.py:236 _input_message` iterates message content
  and emits `responses_api.InputImage(type="input_image", detail=…, image_url=part.url)` for each
  `ImageContent` — this is the content-build path used for every request. Vision is gated at
  `responses.py:173 if has_image and not self.model_spec.vision: _unsupported(...)`.
- **Cerebras wire path:** `chat_completions.py:262–266 _message` emits `chat_api.ImageContentPart(...)`
  from `ImageContent`; `vision=True` in its spec.
- **DeepSeek** rejects images (`vision=False`, `isinstance(part, ImageContent)` → unsupported).

**Refutation attempt.** Checked whether the path is a dead stub: it is not — two of three providers
serialize images to distinct wire types. Nuance (not a refutation): `image_request` has **no
in-library op caller** (`rg image_request symai/ops symai/runtime` → none); it is a user-assembled
builder, exactly as the claim's "importable" wording implies. Deleting the four symbols removes
real OpenAI+Cerebras vision support. Claim stands.

---

## Claim 5 — JSON AST never produced in-library; round-trips to `pydantic.JsonValue` — CONFIRMED

**Method.** `rg` for `JsonObject(`/`JsonArray(`/`JsonEntry(`/`JsonObject.parse`/`.to_builtin(` and
for any op/operations/function that sets `response_format`.

**Evidence.**
- Constructions of the AST live **only** in `models.py` itself (inside `parse`/`_parse_json_value`
  recursion) and in `tests/`. No `symai/` op, `operations.py`, or engine constructs them.
- The runtime `JsonSchemaResponseFormat` is constructed **only in tests**; `operations.py`/`ops/*`/
  `function.py` never set `response_format` (`rg 'response_format\s*=' … → NONE`), so structured
  output is a user-assembled request path — no in-library producer.
- Round-trip confirmed at the engine boundary: both consumers do
  `cast("JsonValue", response_format.json_schema.to_builtin())` (`responses.py:261`,
  `cerebras/chat_completions.py:279`) where `JsonValue` is **`pydantic.JsonValue`**
  (`from pydantic import JsonValue` at the top of both files). The bespoke AST is flattened to a
  builtin dict/list and handed to a client field typed as the pydantic JSON type. Claim stands.

---

## Claim 6 — DeepSeek omits `by_alias=True`; harmless (0 aliased fields) — CONFIRMED

**Method.** `rg model_dump|by_alias` in each `_client.py`; `rg alias` across each provider's client.

**Evidence.**
```
cerebras/_client.py:110  model_dump(mode="json", by_alias=True,  exclude_none=True)
openai/_client.py:94/99  model_dump(mode="json", by_alias=True,  exclude_none=True)
deepseek/_client.py:104  model_dump(mode="json",                 exclude_none=True)   ← no by_alias
```
Aliased fields exist only in openai (`responses.py:195 schema_: … Field(alias="schema")`) and
cerebras (`chat.py:148 Field(alias="schema")`), both for the reserved word `schema`, and both
belt-and-suspender it with `serialize_by_alias=True` in `model_config`. `rg alias
symai/providers/deepseek/` → **NONE**. So `by_alias` is a genuine no-op for DeepSeek today →
divergence is harmless-but-fragile-by-omission, exactly as claimed. Claim stands.

---

## Claim 7 — `_normalize_text` strips wrapping single-quotes for every decoder — CONFIRMED

**Method.** Read `decoding.py`; confirm all four decoders call `_normalize_text`; execute probes.

**Evidence.** `_normalize_text` (`decoding.py:134`) strips one layer of wrapping `'…'`. It is called
by `TextDecoder.decode` (l.43), `ConstructorDecoder.decode` (l.51), `TypeAdapterDecoder.decode`
(l.81), `PydanticDecoder.decode` (l.92) — every decoder, TextDecoder included. Executed:
```
TextDecoder().decode("'Twas the night'")  →  "Twas the night"      # leading apostrophe lost
_normalize_text('"value"')                →  '"value"'             # double quotes preserved (asymmetric)
_normalize_text("'value'")                →  'value'
```
The content-altering, asymmetric footgun applies to faithful text output via `TextDecoder`. Claim
stands.

---

## Claim 8 — `Runtime` accepts `" chat "`; `RuntimeConfig` rejects it — CONFIRMED

**Method.** Read both validators; execute both construction paths.

**Evidence.** `Runtime.__init__` snapshots keys verbatim (`dict(language_models)`), then
`_validate_aliases` (`runtime.py:87`) checks only `isinstance str` + `not alias` — **no whitespace
check**. `RuntimeConfig._validate_aliases` (`config.py:81`) adds `if alias != alias.strip(): raise
… "must not contain outer whitespace"`. Executed:
```
Runtime(language_models={" chat ": DummyEngine()})   → ACCEPTED, keys == [' chat ']
RuntimeConfig(language_models={" chat ": EngineSpec(…)}) → REJECTED (pydantic ValidationError)
```
Two supported construction paths enforce different alias syntax. Claim stands.

---

## Claim 9 — `PydanticDecoder` ≡ `TypeAdapterDecoder` for a `BaseModel` — CONFIRMED

**Method.** Read both; execute on a `BaseModel`.

**Evidence.** Executed on `class Answer(BaseModel): value: int` with `'{"value": 7}'`:
```
PydanticDecoder(Answer).decode(s)                  → value=7
TypeAdapterDecoder(TypeAdapter(Answer)).decode(s)  → value=7
equal value: True   same type (both Answer): True
```
Both apply `_normalize_text`, both wrap `ValidationError` via `_decode_error`. For any `BaseModel`
subclass the success result and type are identical; the only differences are cosmetic (the decoder
class name embedded in the `DecodeError` string) and static typing (`PydanticDecoder` is
`ModelT`-bound). Claim stands.

---

## Claim 10 — no `providers/*/client/` imports `symai.runtime` — CONFIRMED

**Method.** `rg` for any `runtime`/`symai.ops`/`symai.function`/`symai.symbol`/`symai.decoding`/
`symai.operations` import under every provider `client/` dir, then enumerate all imports there.

**Evidence.** The targeted grep returned `NO MATCHES`. The full import enumeration shows client
modules import only: stdlib (`json`, `typing`, `urllib`, `dataclasses`, `enum`), `httpx`,
`pydantic`, `symai.providers._client.*`, and their own provider package. The engine layer is the
sole crossing point. The redesign's headline client↔engine boundary holds in live code. Claim stands.

---

## Claim 11 — no request field can make a provider return >1 output — CONFIRMED

**Method.** `rg` for `n`/`candidates`/`best_of`/`num_*` fields in `runtime/models.py` and every
provider wire request DTO; read the request DTO field lists.

**Evidence.** No such field anywhere: `SamplingConfig`/`LanguageModelRequest` have none;
`deepseek CreateChatCompletionRequest` fields = {messages, model, thinking, reasoning_effort,
max_tokens, response_format, stop, temperature, top_p, logprobs, top_logprobs, user_id} — no `n`;
`openai CreateResponseRequest` has no `n` (Responses returns one response). Request DTOs are
`StrictModel` (`extra="forbid"`), so an `n` cannot even be smuggled through. Absent `n`, chat
providers default to one choice. Meanwhile the response side keeps
`outputs: tuple[…] = Field(min_length=1)` + `index` + `seen_indices` dedup + `outputs.sort(key=index)`
(cerebras/deepseek) + `decode_output(output_index=0)` scan — all permanently exercising the
single-element path. Claim stands (about the request side; a provider spontaneously returning
multiple outputs is out of scope and does not occur — OpenAI aggregates output items into one
message).

---

## Net

The prior rounds' load-bearing claims are robust. The only correction is a phrasing one (Claim 2:
cerebras is a superset, not "identical"), and it does not weaken the underlying dedup finding. No
claim was refuted. The dead-surface findings (unread spec fields + orphan enums, JSON AST,
N-output machinery, write-only logprobs' siblings), the provider duplication measurements, the
`_normalize_text`/decoder observations, the Runtime/RuntimeConfig seam, and the client↔engine
boundary all reproduce exactly against live HEAD `84f703b`.
