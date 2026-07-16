# r3-m3b — Adversarial verification of the highest-impact prior claims

**Mandate.** Try HARD to *refute* the ~11 most consequential claims from r1/r2 against live
code at HEAD `84f703b`. Read-only; every verdict below is re-derived from the code now, not
trusted from any sibling report. Method per claim: grep / read / `diff` / live `uv run python`.

## Scoreboard (claim survival)

| # | Claim (compressed) | Verdict |
|---|--------------------|---------|
| 1 | `LanguageModelSpec.message_roles/content_types/response_formats/context_tokens` have ZERO enforcement reads | **CONFIRMED** |
| 2 | Provider `client/transport.py`+`headers.py` identical across providers except provider name | **PARTIAL** (openai≡deepseek yes; cerebras is a real superset) |
| 3 | cerebras vs deepseek `_client.py` ~83%, `chat_completions.py` ~55% identical | **CONFIRMED** (99/119=83%, 251/459=55%) |
| 4 | Multimodal image path wired end-to-end + importable; deleting drops a real feature | **CONFIRMED** |
| 5 | `JsonObject/JsonArray/JsonEntry` never constructed in production; round-trips to `pydantic.JsonValue` | **CONFIRMED** |
| 6 | DeepSeek serializes WITHOUT `by_alias=True` while others use it; currently harmless | **CONFIRMED** |
| 7 | `_normalize_text` strips wrapping single-quotes for EVERY decoder incl. `TextDecoder` | **CONFIRMED** |
| 8 | `Runtime.__init__` accepts a `" chat "` alias that `RuntimeConfig` rejects | **CONFIRMED** |
| 9 | `PydanticDecoder` ≡ `TypeAdapterDecoder` for a `BaseModel` | **CONFIRMED** |
| 10 | NO `providers/*/client/` module imports `symai.runtime` | **CONFIRMED** |
| 11 | N-output: no request field can make a provider return >1 output | **CONFIRMED** |

**Survival: 10 CONFIRMED, 1 PARTIAL, 0 REFUTED.** The prior corpus holds up under attack.
Only claim 2's *compressed blanket phrasing* is imprecise — and r1-02 D3 itself stated the
precise version correctly (see below). No load-bearing prior claim was refuted.

State note: the cutover is done at HEAD `84f703b` — `symai/prompts.py` and `symai/backend/`
are deleted, root `symai/__init__.py` is 0 bytes. Any r1 finding about those (B1/B2/B3, R5-1,
r1-09 A–H) is now RESOLVED and out of scope here; I targeted only the *surviving* claims.

---

## Claim 1 — spec matrix fields have zero enforcement reads — CONFIRMED

**Method.** `rg '\.message_roles' / '\.content_types' / '\.response_formats' / '\.context_tokens'`
across `symai/`; then confirm the field-population direction in `_normalized_model_spec`.

**Evidence.**
- `.message_roles`, `.content_types`, `.response_formats` → **0 reads anywhere** in `symai/`.
- `.context_tokens` → 4 hits, *all* of the form `context_tokens=spec.context_tokens` inside
  `_normalized_model_spec(spec: chat_api.ModelSpec)` (deepseek line 81, cerebras 64, openai
  responses 66, openai embedding 39). `spec` is the **client** `ModelSpec`; these lines *write*
  the runtime `LanguageModelSpec.context_tokens` from the client value. The runtime field is
  never read back. `EmbeddingModelSpec.context_tokens` is likewise written (embedding.py:39)
  and never read, whereas `.dimensions` *is* read (embedding.py:119,129).
- No dynamic access either: `rg 'getattr\(.*spec|model_spec\['` → none.
- Enforcement is instead hand-coded and re-encodes the matrix, e.g. deepseek rejects images via
  `if any(isinstance(content, ImageContent) ...): self._unsupported("DeepSeek does not support
  image content")` (chat_completions.py:220) and developer messages at :218 — mirroring the
  dead `content_types=(ContentType.TEXT,)` / `message_roles=_DEEPSEEK_MESSAGE_ROLES`.

**Verdict.** CONFIRMED, verbatim. Four `LanguageModelSpec` fields + `EmbeddingModelSpec.context_tokens`
are populated-but-unread; the three enums (`MessageRole`/`ContentType`/`ResponseFormatType`) exist
only to fill them. Matches r1-06 C2 and r1-07 PA-2 exactly.

## Claim 2 — transport.py/headers.py identical except provider name — PARTIAL

**Method.** `diff` openai↔deepseek and openai↔cerebras for both files.

**Evidence.**
- `openai/client/transport.py` vs `deepseek/client/transport.py`: **single differing line** — the
  module docstring (`OpenAI` vs `DeepSeek`). `ResponseMetadata` + `APIResponse[T]` byte-identical.
- `openai/client/headers.py` vs `deepseek/client/headers.py`: **single differing line** — the import
  path (`openai.client.transport` vs `deepseek.client.transport`). Constants + `extract_response_metadata`
  byte-identical.
- `cerebras` is **NOT** identical: its `transport.py` adds a whole `RateLimitState` model (6 fields)
  and a `rate_limit: RateLimitState` field on `ResponseMetadata`; its `headers.py` adds 6
  `x-ratelimit-*` header constants and populates `rate_limit` in `extract_response_metadata`.

**Verdict.** PARTIAL. The blanket "identical across providers except provider name" is **refuted for
cerebras** — it is a functional superset (rate-limit parsing), not a name change. It is exactly true
for openai≡deepseek. Note r1-02 D3 already phrased this precisely ("openai ≡ deepseek verbatim;
cerebras a strict superset"), so the *report* is correct; only the compressed target claim overreaches.

**Corrected claim.** "openai and deepseek `transport.py`/`headers.py` are byte-identical modulo the
provider token in one docstring / one import path; cerebras extends both additively with rate-limit
state." Consolidation target stands (shared base + cerebras subtype-extension).

## Claim 3 — cerebras vs deepseek `_client.py` ~83%, `chat_completions.py` ~55% — CONFIRMED

**Method.** `diff` line counts (common = larger_file − larger-only lines).

**Evidence.**
- `_client.py`: cerebras 119 L, deepseek 113 L; cerebras-only `<`=20 → common **99/119 = 83.2%**
  (deepseek-relative 99/113 = 87.6%). Reproduces r1-02 D2's "99/119 (83%)" to the line.
- `chat_completions.py`: cerebras 459 L, deepseek 435 L; cerebras-only `<`=208 → common
  **251/459 = 54.7%** (deepseek-relative 251/435 = 57.7%). r1-02 D6 said 246/459 ≈ 55% — within
  ~1% of my count.

**Verdict.** CONFIRMED with independent measurement.

## Claim 4 — multimodal image path wired end-to-end + importable — CONFIRMED

**Method.** `rg 'ImageContent'`; read the openai/cerebras wire builders and `operations.image_request`.

**Evidence.**
- `ImageContent` is a public model in `runtime/models.py:200` (importable).
- Builder: `operations.image_request` constructs `UserMessage(content=(TextContent(...),
  ImageContent(url=image_url, detail=detail)))` (operations.py:61); `data_uri` builds base64 data URIs.
- **Wire emission, openai:** `_input_message` builds `responses_api.InputImage(type="input_image",
  image_url=part.url)` (responses.py:243–246).
- **Wire emission, cerebras:** `_content` builds `chat_api.ImageContentPart(type="image_url",
  image_url=chat_api.ImageURL(url=content.url))` (chat_completions.py:266).
- deepseek deliberately rejects images (`content_types=(TEXT,)`, `vision=False`, hardcoded reject).

**Verdict.** CONFIRMED. The path builder → `ImageContent` → engine → provider image part is complete
for 2 of 3 providers. Deleting `ImageContent`/`image_request`/`data_uri` drops a genuinely-plumbed
feature (r1-09 finding I). Not dead code.

## Claim 5 — JSON AST never constructed in production; round-trips to `pydantic.JsonValue` — CONFIRMED

**Method.** `rg` for `JsonObject(`/`JsonArray(`/`JsonEntry(`/`JsonObject.parse`/`JsonSchemaResponseFormat(`
in `symai/` vs `tests/`; read the two engine consumers.

**Evidence.**
- In `symai/`, `JsonObject`/`JsonArray`/`JsonEntry` appear only as *definitions* + internal recursion
  in `_parse_json_value`/`JsonObject.parse` (models.py). `JsonObject.parse` is invoked **only from
  tests** (`tests/runtime/test_models.py`, provider engine tests) — never from any production path.
  `operations.py` never sets `response_format` (defaults to `TextResponseFormat`). The one
  `chat_api.JsonSchemaResponseFormat(` in cerebras engine (chat_completions.py:274) is the *client's*
  own StrictModel type, unrelated to the runtime AST.
- Round-trip: both consuming engines do `cast("JsonValue", response_format.json_schema.to_builtin())`
  (openai responses.py:261, cerebras chat_completions.py:279), and `JsonValue` there is
  `from pydantic import JsonValue` (responses.py:5, chat_completions.py:5). So the deep-frozen
  tuple-of-entries AST is flattened to a builtin dict/list and typed as pydantic's own JSON type.

**Verdict.** CONFIRMED. The library never emits the AST itself; it can only exist if a caller
hand-assembles a `JsonSchemaResponseFormat` (or a test does), and at the boundary it is discarded
back to `pydantic.JsonValue`. Matches r1-06 C4.

## Claim 6 — DeepSeek serializes without `by_alias=True`; harmless — CONFIRMED

**Method.** `rg 'model_dump'` in each `_client.py`; check for any `alias=`/alias_generator on the
DeepSeek request models and the shared `StrictModel` base.

**Evidence.**
- deepseek `_client.py:104`: `request.model_dump(mode="json", exclude_none=True)` — **no** `by_alias`.
- cerebras `_client.py:110` and openai `_client.py:94,99`: `by_alias=True`.
- Harmless because DeepSeek's request models declare **no aliases**: `rg 'alias=' deepseek/client/chat.py`
  → none, and `providers/_client/models.py::StrictModel` config is `frozen/strict/extra=forbid` with
  no `alias_generator`/`populate_by_name`. With zero aliases, `by_alias=True` and its omission produce
  identical JSON.

**Verdict.** CONFIRMED — currently a no-op divergence, latent only if an aliased field is ever added
to a DeepSeek request model. Matches r1-02 D2's coupling note.

## Claim 7 — `_normalize_text` strips wrapping single-quotes for every decoder incl. `TextDecoder` — CONFIRMED

**Method.** Read `decoding.py`.

**Evidence.** `_normalize_text` (line 134): `strip()`, then `if len>=2 and startswith("'") and
endswith("'"): normalized[1:-1].strip()`. It is called by **all four** decoders:
`TextDecoder.decode` (43), `ConstructorDecoder.decode` (51), `TypeAdapterDecoder.decode` (81),
`PydanticDecoder.decode` (92). Double-quotes are *not* stripped (asymmetric), so a faithful
`TextDecoder` result like `'Twas ... night'` loses its outer apostrophes.

**Verdict.** CONFIRMED, unchanged from r1-04 finding 3.

## Claim 8 — `Runtime.__init__` accepts `" chat "` that `RuntimeConfig` rejects — CONFIRMED

**Method.** Read both `_validate_aliases`.

**Evidence.**
- `Runtime._validate_aliases` (runtime.py:87–97): checks `isinstance(alias, str)` and `not alias`
  only — **no** outer-whitespace check. `" chat "` passes.
- `RuntimeConfig._validate_aliases` (config.py): checks `not alias` **and** `alias != alias.strip()`
  → raises "must not contain outer whitespace". `" chat "` is rejected.

**Verdict.** CONFIRMED. Divergence is bidirectional (Runtime alone type-checks keys since it accepts an
arbitrary `Mapping`; RuntimeConfig alone enforces whitespace). Matches r1-05 R5-3.

## Claim 9 — `PydanticDecoder` ≡ `TypeAdapterDecoder` for a `BaseModel` — CONFIRMED

**Method.** Live `uv run python`.

**Evidence.**
```
PydanticDecoder(Answer).decode('{"value": 7}')                    -> value=7  (Answer)
TypeAdapterDecoder(TypeAdapter(Answer)).decode('{"value": 7}')    -> value=7  (Answer)
equal: True   same type: True
both raise DecodeError on '{"value": "x"}'
```
Source parity: both apply `_normalize_text` then `validate_json`/`model_validate_json` and wrap
`ValidationError` via `_decode_error`.

**Verdict.** CONFIRMED — functionally identical over the BaseModel domain. (`TypeAdapterDecoder` is
strictly broader: it also handles non-model `TypeAdapter[T]`; so `PydanticDecoder` is the redundant
subset, as r1-04 finding 1 argued.)

## Claim 10 — no `providers/*/client/` module imports `symai.runtime` — CONFIRMED

**Method.** `rg -rn 'symai\.runtime' symai/providers/*/client/ symai/providers/_client/`.

**Evidence.** Zero hits. The only `import symai...` lines in client packages are intra-client
(`import symai.providers.<p>.client.chat/errors/...`). Client layer knows nothing of `runtime`;
engines are the sole crossing point. Confirms the r1-03a / r1-07 layering "keep".

**Verdict.** CONFIRMED.

## Claim 11 — no request field can make a provider return >1 output — CONFIRMED

**Method.** `rg` for `n:`/`num`/`candidates`/`best_of`/`choices:int` across all client request models
and `runtime/models.py`; enumerate the request model field lists.

**Evidence.**
- Normalized side: `LanguageModelRequest`/`SamplingConfig` (models.py) have no `n`/candidates field.
- Client side: `cerebras`/`deepseek` `CreateChatCompletionRequest` and openai `CreateResponseRequest`
  field lists contain no output-count field (they carry `logprobs`/`top_logprobs`/etc., never `n`).
  Cross-file grep for any such field → **none**.
- Consequence: providers default to one choice; `LanguageModelResponse.outputs` (min_length=1) always
  has exactly one element at `index=0`; the per-choice dedup/sort in the chat engines and
  `decode_output(output_index=0)` permanently exercise the single-element path.

**Verdict.** CONFIRMED. The N-output tuple/index/dedup/sort machinery is defensive against a state no
request surface can produce. Matches r1-06 C3.

---

## Net read

Under deliberate refutation pressure, the prior corpus is accurate. Ten of the eleven highest-impact
claims are confirmed exactly (three of them reproduced numerically to within a line/percent: the 83%
`_client.py`, the ~55% `chat_completions.py`, and the JSON-AST round-trip). The lone PARTIAL (claim 2)
is a compression artifact of the target phrasing, not a defect in r1-02, which stated the precise
"openai≡deepseek verbatim, cerebras superset" version. No fabricated or stale claim survived into the
targeted set. The auditors' verified-and-quoted discipline held.
