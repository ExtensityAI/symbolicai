# R3 · Adversarial verification of load-bearing claims

Authored in the main loop after the Round-3 subagents died on the shared session API
limit. All checks are local greps/diffs against live HEAD `84f703b`; each verdict quotes
its evidence. Default stance was skeptical (assume the claim is wrong until proven).

## Scoreboard

| # | Claim (from R1/R2) | Verdict | Note |
|---|---|---|---|
| 1 | `LanguageModelSpec.message_roles/content_types/response_formats` have **zero reads** | **CONFIRMED** | 0 read sites outside `models.py`. |
| 1b | `...context_tokens` is dead-for-enforcement | **CONFIRMED** | Only *written* (`spec.context_tokens=spec.context_tokens` ×4); never read to gate. |
| 2 | provider `transport.py`/`headers.py` are **byte-identical** | **REFUTED (literal) / CONFIRMED (intent)** | They differ **only** in one docstring word + one import path. Dedup case is *stronger*, not weaker. |
| 3 | cerebras vs deepseek `_client.py` ~83%, `chat_completions.py` ~55% identical | **PLAUSIBLE** | Not byte-diffed here; consistent with r1-02's measured figures. |
| 4 | multimodal path (`ImageContent`/`image_request`/`vision`) is wired end-to-end + importable → real feature | **CONFIRMED** | `vision` gated in openai `_validate_request`; image content parts built in engines; deleting it drops a real capability. |
| 5 | `JsonObject/Array/Entry` never constructed in production, round-trips to `pydantic.JsonValue` | **CONFIRMED** | Only construction sites are inside `models.py` itself; `to_builtin()` consumed only by engines that `cast("JsonValue", ...)`. No `operations.py`/`function.py` path builds it. |
| 6 | DeepSeek serializes **without** `by_alias=True` while openai/cerebras use it; currently harmless | **CONFIRMED (both halves)** | `deepseek/client/_client.py`: `model_dump(mode="json", exclude_none=True)` — no `by_alias`; deepseek `chat.py` defines **no** aliases → latent, not yet live. |
| 7 | `_normalize_text` strips wrapping single-quotes for **every** decoder incl. `TextDecoder` | **CONFIRMED** | Called at `decoding.py:43` (Text), `:51` (Constructor), `:81` (TypeAdapter), `:92` (Pydantic). Content-altering for text. |
| 8 | `Runtime.__init__` accepts a `" chat "` alias that `RuntimeConfig` rejects | **CONFIRMED** (by inspection) | `Runtime._validate_aliases` checks str-type+non-empty only; `RuntimeConfig._validate_aliases` also rejects outer whitespace. |
| 9 | `PydanticDecoder` ≡ `TypeAdapterDecoder` for a `BaseModel` | **CONFIRMED** | Both call `model_validate_json`/`validate_json` on normalized text; same result+type. |
| 10 | **No** `providers/*/client/` module imports `symai.runtime` (client↔engine boundary) | **CONFIRMED** | grep across all four `client/` packages + `_client/`: zero runtime references. |
| 11 | No request field can make a provider return >1 output (N-output dead) | **CONFIRMED** | No `n`/`candidates`/`num_outputs` on `LanguageModelRequest` or `SamplingConfig`. |

**Survival rate:** of 11 load-bearing claims, **9 CONFIRMED, 1 PLAUSIBLE, 1 corrected** (claim 2:
"byte-identical" is technically false but the dedup conclusion is reinforced). No load-bearing
finding collapsed under scrutiny — the audit's spine is sound.

## The two corrections that matter

1. **"byte-identical transport/headers" → "identical except the provider name."** `diff` shows the
   *only* differences between `openai/client/transport.py` and `deepseek/client/transport.py` are the
   docstring provider word and (for headers) the intra-package import path. Use precise wording in the
   plan: these files are **structurally identical**, which makes the shared-base extraction *safer*
   than "byte-identical" implied (the residual difference is a string, not logic).

2. **DeepSeek `by_alias` is confirmed a latent bug, not a live one.** Both halves hold: DeepSeek omits
   `by_alias=True` **and** its request model currently declares no aliased fields. Fix is one keyword;
   value is preventing a silent wire regression the day someone adds an alias.
