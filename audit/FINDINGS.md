# SymbolicAI `refactor/cleanup` — Complete Findings Register

Every finding from every lens and every audit round, named and anchored to `file:line`.
Baseline: `main` (merge-base `da28e25`). Branch head: `a220d6f`. Audited 2026-07-15.

**Severity is the *calibrated* severity** after adversarial reproduction (Round 4) where a
finding was reproduced; the originally-reported severity is noted when it changed.
**Status legend:** `repro-script` = reproduced with a runnable script · `repro-read` = confirmed by reading the exact code path (provider-behaviour claims that need a live key) · `docs` = confirmed against official provider documentation · `measured` = confirmed by measurement · `self` = independently reproduced by the lead auditor outside the workflow.

**On the IDs and counts:** finding IDs are stable and non-sequential. Cross-listed IDs are counted once under their primary lens. The scoreboard counts active findings only; informational notes, explicit product non-goals, withdrawn claims, and Appendix observations are excluded. Gaps are therefore intentional. A follow-up pass opened 259 anchors against the audited head and corrected the anchor/wording nits it found.

| Rounds | What ran |
|---|---|
| R1 | 7 parallel readers mapped each layer + old architecture → architecture synthesis + 20-item hunting list |
| R2 | 16 independent audit lenses → 75 raw findings |
| R3 | 3 provider clients verified against **live** OpenAI / Cerebras / DeepSeek API docs (web) |
| R4 | 20 correctness findings each independently reproduced-or-refuted with throwaway scripts + severity calibration |

---

## Active-findings scoreboard

| Lens | Findings | Critical | High | Medium | Low |
|---|---:|---:|---:|---:|---:|
| Correctness / runtime | 5 | 0 | 2 | 1 | 2 |
| Engine-adapter correctness | 4 | 1 | 2 | 1 | 0 |
| Provider API fidelity | 9 | 0 | 2 | 3 | 4 |
| OpenAI client fidelity | 5 | 0 | 1 | 4 | 0 |
| Cerebras / DeepSeek client fidelity | 2 | 0 | 0 | 0 | 2 |
| Symbol-layer correctness | 6 | 0 | 3 | 2 | 1 |
| Separation of concerns / duplication | 7 | 0 | 3 | 3 | 1 |
| Over-engineering / accidental complexity | 4 | 0 | 2 | 1 | 1 |
| Contract coherence | 3 | 0 | 2 | 1 | 0 |
| Feature parity vs `main` | 14 | 0 | 3 | 8 | 3 |
| Extensibility / instance model | 4 | 0 | 2 | 0 | 2 |
| Test quality & coverage | 5 | 0 | 1 | 2 | 2 |
| Docs truthfulness | 3 | 0 | 1 | 2 | 0 |
| Public API & packaging | 5 | 0 | 1 | 3 | 1 |
| Error-handling UX | 3 | 0 | 1 | 1 | 1 |
| Performance | 1 | 0 | 0 | 1 | 0 |
| Persistence / file safety | 2 | 0 | 2 | 0 | 0 |

> The decisive observed ship-blocker is the OpenAI model-echo check: documented resolved model identifiers do not equal the configured aliases. DeepSeek has the same fragile equality code, but its exact live echo behavior was not observed and must not be claimed with OpenAI-level certainty.

---

## A. Correctness & runtime lifecycle

### BUG-01 — Interrupted `close()` wedges the runtime permanently and leaks the httpx client
**Severity: High** · `symai/runtime/runtime.py:145-172` · status: `repro-script`, `self`
`close()` sets `_state = CLOSING` (`:154`) then blocks in `wait_for(_in_flight == 0)` (`:155`) with no timeout; `_state = CLOSED` is only assigned in the `finally` at `:167`, reached *after* the wait returns. If a `BaseException` (Ctrl-C `KeyboardInterrupt`) interrupts the wait while another thread is mid-`execute()`, it unwinds out of `close()` with state stuck at `CLOSING`, handles never detached, and `httpx.Client.close()` never run (connection leak). Every subsequent `close()` then takes the `CLOSING` branch (`:150-151`) and blocks on `wait_for(_state is CLOSED)` forever — permanent deadlock. Reproduced end-to-end: first interrupted `close()` leaves `state=closing, cleanup=[]`; second `close()` never returns. Reachability: concurrent `execute` + interrupt during a multi-second in-flight call (the exact scenario the `_in_flight`/`Condition` machinery exists for), then the common catch-`KeyboardInterrupt`-and-shut-down pattern.

### BUG-02 — `current_runtime()` is invisible to worker threads / `ThreadPoolExecutor`
**Severity: Low** (reported Medium) · `symai/runtime/runtime.py:73,191-196` · status: `repro-script`
`__enter__` stores the runtime in a `ContextVar`; ContextVars do not propagate into freshly-spawned OS threads, so any `Symbol` op dispatched to a `ThreadPoolExecutor` runs in an empty context and raises `NoActiveRuntimeError`, while the same call on the main thread succeeds. Every user op routes through `current_runtime()` (`symbol.py:747`, `ops/primitives.py:95,157,2127`, `components.py:99`). Downgraded because it fails **loudly** with a typed error (no silent corruption), the library ships no threading of its own, `asyncio.to_thread` *does* propagate correctly (that part of the original claim was refuted), and workarounds exist (`copy_context().run(...)`). Still a real gap: there is no supported way to fan Symbol ops across threads.

### BUG-03 — `__exit__` resets the ContextVar before `close()`; cross-context exit leaks the client and masks the body exception
**Severity: Low** · `symai/runtime/runtime.py:79-101` (`reset` at `:90` before `close()` at `:94`) · status: `repro-script`
If a runtime is entered in one context and exited in another (enter in one asyncio task, exit from another; or `copy_context().run(rt.__exit__)`), `_CURRENT_RUNTIME.reset(token)` raises `ValueError: Token was created in a different Context`, aborting `__exit__` before `close()` runs. Handles never closed (leak); a `with`-body exception is replaced by the `ValueError`. Reproduced. Exotic/unsupported usage path — normal `with runtime:` is unaffected — hence Low.

### CX-04 — Runtime mixes ambient discovery with a public concurrently callable object
**Severity: Medium** · `symai/runtime/runtime.py:60,63,109-172` · status: `repro-read`
`current_runtime()` is context-local, but `Runtime.execute()` is public and any holder can call the same Runtime directly from another thread; `asyncio.to_thread()` also propagates ContextVars. Concurrency is therefore reachable even though the user-facing operation path suggests single-context ownership. The `Condition`/`_in_flight`/`CLOSING` machinery attempts to support that concurrency and enables BUG-01. Deleting it is safe only after the synchronous contract enforces owner-thread affinity and exclusive handle ownership; otherwise synchronization must be retained and repaired.

### SEC-01 — Malformed `api_key` leaks in plaintext into tracebacks/logs, defeating `SecretStr`
**Severity: High** · `symai/runtime/models.py:428` + every client's `f"Bearer {api_key}"` · status: `repro-script`, `self`
`api_key: SecretStr = Field(min_length=1)` is the only validation. A key with a trailing newline/space/control char (the extremely common `open('key.txt').read()` / `subprocess.check_output(...).decode()`) is accepted, then interpolated raw into the `Authorization` header. `httpcore` rejects the malformed header with `httpx.LocalProtocolError: Illegal header value b'Bearer sk-…\n'` — the plaintext key is in the exception **message**, which is chained up into `TransportError`. httpx redacts the header only in the request *repr*, not in this exception string, so `logger.exception(...)` / Sentry / any traceback capture writes the raw key to logs. Reproduced independently: the formatted traceback contained `REALSECRETKEY`.

---

## B. Engine-adapter correctness

### BUG-05 / API-01 — OpenAI model-echo rejects documented resolved identifiers *(SHIP-BLOCKER)*
**Severity: Critical** · `symai/backend/engines/language_model/openai.py:283`; analogous equality at `deepseek.py:294` · status: `repro-script`, `docs`
The OpenAI engine does `if raw.model != self.model: raise InvalidResponseError`. `self.model` is the configured alias (`gpt-4.1`, `gpt-5.5`), while OpenAI documentation shows responses identifying resolved dated snapshots (for example `o4-mini-2025-04-16`). A successful response can therefore be discarded solely because the provider reports the concrete model it served. DeepSeek uses the same exact-equality assumption, but its documentation is ambiguous and no live response was observed; for DeepSeek this is a confirmed fragility, not a demonstrated total outage. Arbitrary prefix matching is not a safe correction because sibling model names can share prefixes.

### BUG-06 — OpenAI embedding engine rejects `text-embedding-ada-002` for the same reason
**Severity: High** · `symai/backend/engines/embedding/openai.py:130` · status: `repro-script`
Same exact-string model check. OpenAI's embeddings endpoint echoes the resolved id, and `text-embedding-ada-002` classically resolves to `text-embedding-ada-002-v2`, so every ada-002 embedding response is rejected. `text-embedding-3-small/large` echo verbatim and are unaffected — so this hits one of three catalog models, hence High rather than Critical.

### BUG-07 — Cerebras & DeepSeek engines crash on legitimate null-content responses (content filter / refusal)
**Severity: High** · `symai/backend/engines/language_model/cerebras.py:322-329`; `deepseek.py:332-345` · status: `repro-script`, `self`
`_output` builds `content=(TextContent(text=message.content),) if message.content is not None else ()` and never extracts a refusal. When the provider returns `content: null` (the normal shape for a content-filtered/refused/empty completion), the output has empty content, no reasoning, no refusal, so `LanguageModelOutput.validate_content_reasoning_or_refusal` (`models.py:363-367`) raises → re-raised as `InvalidResponseError`. Confirmed: `ResponseMessage.content` is typed `str | None` and the model carries **no `refusal` field**, so refusal text and `finish_reason=content_filter` are silently lost and indistinguishable from a transport failure.

### BUG-08 — OpenAI engine rejects reasoning responses truncated during thinking, or with multiple phase messages
**Severity: Medium** · `symai/backend/engines/language_model/openai.py:297-301` · status: `repro-script`
`_parse_response` hard-couples reasoning to exactly one assistant message (`if reasoning is not None and len(messages) != 1: raise`) and requires ≥1 `OutputMessage` (`LanguageModelResponse.outputs` `min_length=1`). A reasoning model that exhausts `max_output_tokens` during thinking returns `status=incomplete` with a reasoning-only output and **zero** assistant messages → engine raises "requires exactly one assistant message" and the truncation signal (`finish_reason=LENGTH`) + usage are lost. Multi-phase (commentary + final) responses with 2+ messages are likewise rejected.

---

## C. Provider API fidelity — verified against live docs (Round 3)

### API-01 — OpenAI model-echo (Critical) — see **BUG-05** above. Source: OpenAI cookbook (dated-snapshot echo).

### API-02 — OpenAI reasoning-effort table advertises efforts the live models reject
**Severity: High** · `symai/clients/openai/responses.py:63` (`_REASONING = ReasoningSpec(tuple(ReasoningEffort))`) · status: `docs`
The client grants **all seven** efforts to every reasoning model. Live: `gpt-5.5`/`gpt-5.4` accept `{none,low,medium,high,xhigh}` only (no `minimal`/`max`); `o3`/`o3-pro` accept `{low,medium,high}` only. A caller who sets `minimal`/`max` (any model) or `none`/`xhigh` (o3) passes local validation and gets an API **400**. Default path (`medium`/`high`) is safe. Source: OpenAI model pages.

### API-03 — OpenAI response required-with-no-default fields are latent parse-failures
**Severity: Low** · `symai/clients/openai/responses.py:419-460` (`background`, `store`, `truncation`, `metadata`) · status: `docs`
These are echoed request-config fields declared with no default. If the API ever omits one (or a future revision drops it) the whole `Response` fails to parse and a good completion becomes `InvalidResponseError`. `usage`/`error`/`incomplete_details` are correctly modelled required-but-nullable. Latent, not a confirmed break.

### API-04 — Model catalogs use real, current IDs *(process note — NOT a fabrication)*
**Severity: informational** · status: `docs`
All ten OpenAI IDs (`gpt-5.5`, `gpt-5.5-pro`, `gpt-5.4`, `gpt-5.4-pro`, `gpt-5.4-mini`, `gpt-5.4-nano`, `o3-pro`, `o3`, `gpt-4.1`, `gpt-4.1-mini`), all three Cerebras IDs (`gpt-oss-120b`, `gemma-4-31b`, `zai-glm-4.7`), and both DeepSeek IDs (`deepseek-v4-flash`, `deepseek-v4-pro`) were verified to exist and be served today. The hypothesis that the catalogs were invented is **false** — they were sourced from the providers. (Legacy `deepseek-chat`/`deepseek-reasoner` are documented deprecated 2026-07-24.) This is the mechanism working as intended; recorded so the model-echo bug is not mistaken for a catalog error.

### API-05 — Cerebras advertises & forwards vision/image for text-only models
**Severity: High** · `symai/backend/engines/language_model/cerebras.py:67,75,174-220` · status: `docs`
The engine hardcodes `vision=True` + IMAGE content for all three Cerebras models, but `gpt-oss-120b` and `zai-glm-4.7` are text-only. An image request passes local validation and 400s at the API. The Cerebras client `ModelSpec` has no vision field, so vision is an engine constant rather than a catalog fact (see EXT-04).

### API-06 — Cerebras `clear_thinking` sent to models other than `zai-glm-4.7`
**Severity: Medium** · `symai/backend/engines/language_model/cerebras.py` · status: `docs`
`clear_thinking` is advertised/sent for models that do not support it; only `zai-glm-4.7` accepts it. API rejects for the others.

### API-07 — Cerebras `raw`/`hidden` `reasoning_format` advertised for `gemma-4-31b`, which rejects them
**Severity: Medium** · `symai/backend/engines/language_model/cerebras.py:57` · status: `docs`

### API-08 — Cerebras `finish_reason='tool_calls'` is outside the supported product contract
**Classification: explicit non-goal; excluded from scoreboard** · `symai/backend/engines/language_model/cerebras.py:83-95`
The provider documents `tool_calls`, but SymbolicAI intentionally does not support tool/function calling and cannot send a tool request through the normalized contract. Do not add a payload-less normalized finish reason. If the provider returns it unexpectedly, raise a precise unsupported-response error. The old phantom `'error'` mapping remains a separate fidelity correction.

### API-09 — Cerebras `gemma-4-31b` `reasoning_effort='none'` is valid but the client rejects it
**Severity: Low** · `symai/backend/engines/language_model/cerebras.py` · status: `docs`

### API-10 — Cerebras context/output token limits use paid-tier values; free-tier ceilings are lower
**Severity: Low** · `symai/clients/cerebras/chat.py` (`MODEL_SPECS`) · status: `docs`
`max_tokens` validation against these ceilings can pass locally and 400 on a free-tier key.

### API-11 — DeepSeek engine hard-errors on `temperature`/`top_p` in thinking mode; the live API accepts and silently ignores them
**Severity: Medium** · `symai/backend/engines/language_model/deepseek.py:247-252` · status: `docs`
The engine raises `UnsupportedFeatureError`; DeepSeek instead accepts the params and ignores them for the reasoner. Over-strict — rejects legal requests. (Also note DeepSeek's *spec* advertises TEMPERATURE/TOP_P via `_DEEPSEEK_SAMPLING_FIELDS` at `deepseek.py:64-65` bound at `:88`, contradicting this rejection — see CON-01 / CX-02.)

### API-12 — DeepSeek `reasoning_effort` restricted to `high`/`max`; API also accepts `low`/`medium`/`xhigh`
**Severity: Low** · `symai/backend/engines/language_model/deepseek.py` · status: `docs`

### API-13 — DeepSeek tool-call completion is outside the supported product contract
**Classification: explicit non-goal; excluded from scoreboard** · `symai/backend/engines/language_model/deepseek.py:97`
DeepSeek documents `finish_reason='tool_calls'`, but no tool request/output contract exists by design. This is not roadmap work and should not be represented without the corresponding payload.

---

## D. OpenAI-client fidelity (strict-model parsing)

### CLI-01 — Strict citation submodels defeat the tolerant `Response` on the success path
**Severity: Medium** (reported High) · `symai/clients/openai/responses.py:254-287` · status: `repro-script`
`Response` is a `TolerantModel` (`extra="allow"`) but its citation submodels (`FileCitation`/`URLCitation`/`ContainerFileCitation`/`FilePath`) are `StrictModel` (`extra="forbid"`). One new subfield on any citation object OpenAI adds later turns a valid 200 into a whole-response `ValidationError` → `InvalidResponseError`. Downgraded because it only fires on responses that carry citation annotations (web/file-search tool runs) *and* only after OpenAI extends those objects — but the top-level `TolerantModel` gives false forward-compat confidence the strict leaves silently negate.

### CLI-02 — `list_input_items` reuses the request-only `InputMessage` strict model and cannot decode real items
**Severity: High** (when reached) · `symai/clients/openai/responses.py:416,455-460` · status: `repro-script`
`InputItemList.data` is `InputItem = InputMessage | OutputMessage | …`; `InputMessage` is a `StrictModel` for *building* requests (`extra="forbid"`, `strict=True`, no `id` field). Real returned items carry `id` + structured `content` arrays, so decoding fails for essentially every item — the method is non-functional for the common case. No internal caller today (only affects direct users of this binding).

### CLI-03 — `Response.text.format` submodels are strict; additive fields break structured-output responses
**Severity: Medium** · `symai/clients/openai/responses.py:181-196,210` · status: `repro-script`
`ResponseTextConfig` is tolerant but its `format` references `OutputFormat` whose members (`TextFormat`/`JsonObjectFormat`/`JsonSchemaFormat`) are `StrictModel`. A successful `json_schema` request whose echoed `text.format` object gains any new field fails to parse — a good completion becomes `InvalidResponseError`.

### CLI-04 — Error classification is status-code-only; the OpenAI error envelope is discarded
**Severity: Medium** · `symai/clients/openai/client.py:29-35` (and all clients) · status: `repro-read`
`_raise_for_status` maps 401→Auth, 429→RateLimit, else→APIError purely on status; the JSON error body (`{error:{message,type,param,code}}` — the *why*) is stored as opaque `body` and never surfaced. See also SEC-02 / UX-03.

### CLI-05 — `ResponseErrorCode` is a closed StrEnum; unlisted codes mask the real failure
**Severity: Medium** · `symai/clients/openai/responses.py:374-398` · status: `repro-read`
On the inbound path a new/unknown error code raises a parse error instead of surfacing the provider's actual code. Part of the broader "closed inbound enums turn a legit 200 into a `ResponseError`" pattern (also `ResponseStatus` `:31-37`, `OutputItem` `:352-355`).

---

## E. Cerebras / DeepSeek client fidelity & cross-provider consistency

### CLI-06 — Cerebras request model is the only one that silently forwards unknown/misspelled fields
**Severity: Low** (reported High) · `symai/clients/cerebras/chat.py:172-176` · status: `repro-script`, `self`
`CerebrasCreateChatCompletionRequest` overrides `StrictModel` with `ConfigDict(extra="allow")`, so a typo (`temperatur=0.5`) or wrong-provider field (`max_tokens` vs Cerebras' `max_completion_tokens`) is accepted and serialized onto the wire; the same mistake against DeepSeek/OpenAI raises at construction. Reproduced: both keys serialize. Downgraded because the shipped engine hardcodes field names — only a developer constructing the client model directly is exposed — but it is a real cross-provider inconsistency in the single most fundamental validation policy.

### CLI-07 — DeepSeek client omits `by_alias=True` while Cerebras/OpenAI use it
**Severity: Low** · `symai/clients/deepseek/client.py:82` vs `cerebras/client.py:107`, `openai/client.py:72` · status: `repro-script`
Zero impact today (no DeepSeek field is aliased), but the moment an aliased field is added, DeepSeek emits the Python name instead of the API key. A maintainer copying one of the two client patterns has a 50/50 chance of picking the alias-dropping one — the drift that 3× duplication (SOC-03) invites.

---

## F. Symbol-layer correctness

### BUG-09 — `Symbol.map()` and `Symbol.rank()` silently truncate their collection output to one element by default
**Severity: High** · `symai/ops/primitives.py:1800-1805,1914-1919`; `operations.py:571-572` (`limit_value` list branch) · status: `repro-script`, `self`
Both route through `_execute_symbol(..., literal=True)` which defaults `limit=1` (`primitives.py:119`); the model returns a list literal, so `parse_literal_or_text_output` → `limit_value(list, 1)` slices to `[:1]` (`operations.py:571-572`). `Symbol([...]).map('...')`/`.rank()` return a **1-element** list unless the caller explicitly passes `limit=None`, contradicting the operations' own intent ("Transform each element…" `map_request` prompt `operations.py:283`; "Order the list of objects…" `rank_request` prompt `operations.py:347`). Silent regression vs `main` (where the full list was returned). No test exercises the default-limit path. Reproduced: `limit_value(['a','b','c','d'], 1) == ['a']`.

### BUG-10 — `set` return type unconditionally raises `TypeError`, even with a default supplied
**Severity: High** · `symai/operations.py:471,577-579`; entry points default `limit=1` (`components.py:26,51`, `primitives.py:119`) · status: `repro-script`, `self`
`parse_typed_value` explicitly supports `set`, but `limit_value` raises `TypeError('Cannot deterministically limit an unordered set')` for any set, and it runs **outside** the try/except that honours `default`. So `Function('extract tags', return_type=set)('text')` or `Symbol('x').query('give a set', return_type=set)` always raises — even for a valid set literal, even with `default=set()`. Only non-obvious workaround: `limit=None`. Reproduced: `limit_value({1,2,3}, 1)` raises.

### BUG-11 — Recursive literal coercion re-parses string elements of `map`/`rank`/`setitem` output
**Severity: Low** (reported Medium) · `symai/operations.py:519-533,590-604` · status: `repro-script`, `self`
`parse_literal_or_text_output` runs everything through `_recursive_literal`, which re-`ast.literal_eval`s **every** string element at every depth. Literal-valued strings change type: `['1','2'] → [1,2]`, `['True','(1,2)'] → [True,(1,2)]`. Downgraded because the behaviour is intentional and pinned by a passing test, and the original claim was overstated — `'007'` and `'(a,b)'` are *not* coerced (invalid literals). Still surprising: string tokens that happen to be valid literals silently change type for downstream code.

### BUG-12 — `Symbol._dynamic_context` is a process-wide class global that leaks one instance's `adapt()` into every other Symbol's prompt
**Severity: Medium** · `symai/symbol.py:113,151-155,402-408,507-531` · status: `repro-script`, `self`
`_dynamic_context` is a class-level dict keyed only by `str(type(self))`; every base `Symbol` built without an explicit dynamic context shares one entry, which `adapt()`/`clear()` mutate. It is a plain dict (not a `ContextVar`), so it also crosses threads, and it is folded into the system prompt of every semantic op via `contextualize_language_request`. Reproduced: `a = Symbol('alpha'); b = Symbol('beta'); a.adapt('INJECTED')` makes `b.dynamic_context == '\nINJECTED'` — one request's context contaminates unrelated ones until `clear()`.

### BUG-13 — Equal Symbols can have different hashes, and mutation changes a Symbol's hash
**Severity: High** · `symai/symbol.py:356-363`; equality at `symai/ops/primitives.py:269-296` · status: `repro-script`, `self`
`Symbol.__eq__` compares held values, while `__hash__` hashes `str(self.value)`. Equal dictionaries with different insertion order compare equal but stringify differently, so their hashes differ. Item mutation also changes the string and therefore the hash after insertion into a set/dict. Both violate Python's hash contract. Reproduced with two equal dictionaries yielding unequal hashes and with `Symbol([1])` changing hash after `symbol[0] = 2`. The approved redesign makes the shallow-immutable wrapper unhashable.

### BUG-14 — Native item operations erase Python's actionable exception types
**Severity: Medium** · `symai/ops/primitives.py:1371-1376,1405-1411,1437-1443` · status: `repro-script`, `self`
`__getitem__`, `__setitem__`, and `__delitem__` catch every `Exception` and raise generic `Exception`. Out-of-range list access loses `IndexError`; indexing a non-container loses `TypeError`; dictionary misses lose `KeyError`. This contradicts the methods' own `Raises:` documentation and prevents callers from reacting correctly. Reproduced for list and integer indexing. Native paths should propagate the original exception; the immutable redesign removes item mutation entirely.

---

## G. Separation of concerns & duplication

### SOC-01 — Composition root (factory) lives in the runtime core, inverting the stated layering into a `runtime ⇄ backend` package cycle
**Severity: High** · `symai/runtime/factory.py:8-10` · status: `repro-read`
Stated layering is clients → backend/engines → runtime, but `factory.py` (in the runtime core) imports `symai.backend.engine_handle` and every `symai.backend.engines.*` adapter, while those adapters import back up into `symai.runtime.*`. The composition root belongs in its own top-level package, not inside the layer it wires.

### SOC-02 — `factory.py` eagerly imports every provider's client + engine, so `import symai` compiles all provider schemas
**Severity: High** · `symai/runtime/factory.py:10`; `symai/__init__.py` · status: `measured`
`__init__` → `factory` → unconditional import of all three LM engines, the embedding engine, and all three client packages, each dragging its full pydantic schema tree. Cost is O(providers) even when the app configures one. Measured `import symai` ≈ 82 ms; it grows with every provider added — the opposite of the lazy-load goal.

### SOC-03 — Three language-model engine adapters are ~66% duplicated
**Severity: High** · `symai/backend/engines/language_model/{openai,cerebras,deepseek}.py` · status: `repro-read`, `self`
Each repeats the same scaffolding: `__init__` model lookup, `model`/`model_spec` properties, the 6-arm `execute()` error ladder mapping client → runtime errors, `_validate_request`, `_input_message`, `_response_format`, `_parse_response`, `_usage`, `_error_metadata`. `_retry_after` is **byte-identical** across all four adapters (`openai.py:433`, `cerebras.py:441`, `deepseek.py:417`, `embedding/openai.py:192`); `_unsupported` is near-identical but not byte-identical (present in 3 of the 4 adapters — `openai.py:437`/`cerebras.py:445` return `None`, `deepseek.py:421` returns `Never`; the embedding engine has none). difflib on provider-normalized lines: cerebras↔deepseek 271 identical lines; openai↔cerebras 221.

### SOC-04 — Client HTTP plumbing is 62–78% duplicated across the three client packages
**Severity: Medium** · `symai/clients/{openai,cerebras,deepseek}/client.py` · status: `repro-read`, `self`
`Client.__init__` (empty-key check + bearer header), `_raise_for_status` (401/429/else), `_parse_response` (`json()` + `model_validate`), transport-error wrapping, and the `transport.py` envelopes (`ResponseMetadata` + `APIResponse[T]`) are reimplemented per provider. Cerebras and DeepSeek independently re-model the *same* OpenAI-compatible chat schema.

### SOC-05 — The three client `errors.py` are ~91% verbatim copies
**Severity: Medium** · `symai/clients/{openai,cerebras,deepseek}/errors.py` · status: `repro-read`, `self`
Identical `ClientError/APIError/AuthError/RateLimitError/ResponseError/TransportError` hierarchy and `__init__` bodies. The *code* is identical modulo the provider token, but the files also diverge in added class docstrings and `__init__` formatting — 37 (openai) / 58 (cerebras) / 54 (deepseek) lines, a >20-line spread — which is itself evidence of uncontrolled copy-paste drift.

### SOC-06 — Rate-limit header extraction diverges silently between providers
**Severity: Medium** · `symai/clients/cerebras/client.py:18-40` vs `symai/clients/deepseek/client.py:16-21` · status: `self`
Cerebras extracts six `x-ratelimit-*` headers into a `RateLimitState`; DeepSeek's near-identical client extracts **none**. Copy-paste drift means "which pattern do I copy?" is a coin-flip for the next provider (see EXT-01).

### SOC-07 — `symai/backend` is a vestigial wrapper
**Severity: Low** · `symai/backend/engine_handle.py:8`; empty `__init__.py`s · status: `repro-read`
Beyond `engines/`, `backend/` holds only `engine_handle.py` — a runtime/composition type used exclusively by `symai/runtime`. The package boundary no longer carries meaning.

---

## H. Over-engineering / accidental complexity

### CX-01 — Bespoke `JsonObject`/`JsonArray`/`JsonEntry` pydantic AST re-implements JSON that is immediately converted back to a dict
**Severity: High** · `symai/runtime/models.py:102-175` · status: `repro-read`
A recursive frozen pydantic JSON AST (+ forward-ref alias + three `model_rebuild` calls + `_parse_json_value`/`_json_value_to_builtin`) exists for exactly one use: carrying a JSON-Schema into `JsonSchemaResponseFormat` (`models.py:244`), which the adapters immediately turn back into a builtin dict via `.to_builtin()` (`openai.py:246`, and identically `cerebras.py:264`). It also forces callers to convert their schema into this AST instead of passing a dict, and depends on pydantic's **private** `model_rebuild(_types_namespace=…)` (`:141-143`). A plain `dict[str, JsonValue]` (pydantic already ships a `JsonValue`) gives the same guarantee.

### CX-02 — `LanguageModelSpec` capability matrix is mostly populated-but-never-read; enforcement is parallel hardcoded checks
**Severity: High** · `symai/runtime/models.py:400-411` · status: `repro-read`
Enforcement reads 7 of the 11 spec fields (`response_tokens`, `reasoning_fields` `openai.py:162,257`, `reasoning_efforts`, `reasoning_summaries` `openai.py:182`, `reasoning_formats`, `sampling_fields`, `vision`); the other **4 are dead** — exactly the set in CON-01. The problem is not that most of the matrix is unread but that the gating that *does* happen is imperative `if`-ladders in each `_validate_request` rather than being *driven* by the spec, so the declared matrix and the enforced rules are a hand-synchronized invariant the type system can't enforce — and they already disagree (DeepSeek spec advertises TEMPERATURE/TOP_P at `deepseek.py:64-65,88`; `_validate_request` rejects them — CON-01 / API-11).

### CX-03 — Five StrEnums exist mainly to populate spec tuples whose members are never membership-tested
**Severity: Medium** · `symai/runtime/models.py:27,34,39,68,76` (`MessageRole`, `ContentType`, `ResponseFormatType`, `ReasoningField`, `SamplingField`) · status: `repro-read`
They type the capability-matrix tuples, but almost no code tests an individual member. They are ceremony around CX-02's dead matrix.

*(CX-04 — the dormant `Runtime` concurrency machinery — is defined under section A, grouped with the runtime-lifecycle findings it enables.)*

### CX-05 — `EngineHandle` wraps an optional callable in a `Lock` guarding a `close()` never called concurrently, plus a dead `owns_resources` property
**Severity: Low** · `symai/backend/engine_handle.py:20,26-36` · status: `repro-read`
`close()` is only ever invoked serially (`Runtime.close()` at `runtime.py:162` detaches then closes). `owns_resources` has no *production* reader (only `tests/engines/test_engine_handle.py` asserts on it). The lock and property are speculative generality.

---

## I. Contract coherence (dead type-system surface)

### CON-01 — `LanguageModelSpec`/`EmbeddingModelSpec` capability fields are populated by every engine but read by nothing
**Severity: High** · `symai/runtime/models.py:400-417` · status: `repro-read`
`message_roles` (`:403`), `content_types` (`:404`), `response_formats` (`:405`), `context_tokens` (`:401`), and `EmbeddingModelSpec.context_tokens` are pure dead weight — every engine computes and stores them; nothing reads them. `context_tokens` is never checked against prompt size (no token counting exists — FP-04). A normalized contract whose fields are enforced nowhere is a lie in the type system.

### CON-02 — `logprobs`/`top_logprobs` are sent to providers but the normalized response has nowhere to hold the answer
**Severity: High** · `symai/runtime/models.py:281-282,357-361` · status: `repro-script`
`SamplingConfig.logprobs`/`top_logprobs` are live request fields (Cerebras forwards both `cerebras.py:152,169`; DeepSeek both `deepseek.py:184-185`; OpenAI `top_logprobs` `openai.py:147`). Providers compute and return per-token logprobs (and may bill differently), but `LanguageModelOutput` has **no logprobs field** and the `_output` builders never read `choice.logprobs`. The contract can *request* logprobs but cannot *express* the result — a request/response coherence hole that the public type system advertises as supported.

### CON-03 — Response contract models N outputs and engines dedup/sort them, but no request can ever ask for more than one
**Severity: Medium** · `symai/runtime/models.py:376-377`; `cerebras.py:287-298`, `deepseek.py:303-314` · status: `repro-read`
`LanguageModelResponse.outputs` is a tuple and the chat engines carefully dedup + sort multiple `choice` indices, but there is no `n`/`num_completions` request field, so the multi-output machinery is permanently dead. (OpenAI's own `text` property joins parts.)

---

## J. Feature parity vs `main` (intentional capability reductions)

> Provider/model count reduction and tool/function calling are explicit product decisions, not defects. The remaining entries record other removed capabilities so compatibility and migration cost stay visible; they are not automatically roadmap commitments.

### FP-01 — Tool / function calling intentionally unsupported
**Classification: explicit non-goal; excluded from scoreboard** · `symai/runtime/models.py:297-314`
The normalized request/output contract has no tools or tool-call payload by design. Do not add tool fields, tool finish reasons, or tool-calling roadmap work unless the product decision is explicitly reversed.

### FP-02 — Vector-index / RAG stack removed; capability set closed to `{language_model, embedding}`
**Severity: High** · `symai/runtime/runtime.py:45` · status: `repro-read`
`main` shipped Pinecone/Qdrant/vectordb engines + `extended/vectordb.py` + RAG servers. Gone, and the two-slot `Runtime` (EXT-02) has no place to host an index capability.

### FP-03 — Search / scrape / OCR / speech-to-text / TTS / file-reader engines removed with no capability slot
**Severity: High** · `symai/runtime/factory.py:71` · status: `repro-read`
`main` had search (OpenAI/Gemini/Perplexity/SerpAPI/Parallel/Firecrawl), scrape, OCR (Mistral), Whisper STT, OpenAI TTS, and a file/PDF reader — each behind an `Interface`. None can be expressed by the current two capabilities.

### FP-04 — Client-side token counting & automatic context-window truncation removed
**Severity: High** · `symai/runtime/models.py:400` (`context_tokens` carried but unused) · status: `repro-read`
`main` carried a tokenizer + `compute_required/remaining/truncation`. Gone; `context_tokens` is declared (CON-01) but never checked, so an over-long prompt now fails at the provider instead of being truncated or rejected locally.

### FP-05 — Streaming responses removed; client and runtime are single-shot only
**Severity: Medium** · `symai/clients/openai/client.py:65` · status: `repro-read`
No `stream` field on any request model; Cerebras' `extra="allow"` would let `stream=True` onto the wire and then fail parsing the SSE body (CLI-06).

### FP-06 — No async/await support; the entire stack is synchronous
**Severity: Medium** · `symai/runtime/runtime.py:109` · status: `repro-read`
`main` had async wrappers + async engine paths. `Runtime.execute` is sync-only; `factory` builds a sync `httpx.Client`. Combined with BUG-02, there is no supported concurrency story at all.

### FP-07 — Local/in-process model engines (llama.cpp, HF, vLLM) excluded by the config mechanism, not just unimplemented
**Severity: Medium** · `symai/runtime/models.py:428` · status: `repro-read`
`_create_handle` mandatorily requires an `api_key` (SecretStr, `min_length=1`) and an HTTP transport, so a local in-process engine cannot be expressed even by a user — a mechanism barrier, not "fewer providers".

### FP-08 — Structured-output validation with self-healing retry is a required capability that must be rebuilt
**Severity: High (required capability, not an acceptable reduction)** · `symai/operations.py:466` (cleanup baseline) · status: `repro-read`
Basic JSON-schema passthrough survives (`JsonSchemaResponseFormat` → structured-output request; `parse_typed_value` does `model_validate_json`), but `main`'s `strategy.py`/`LLMDataModel` validate-then-remedy loop is gone; a malformed model output now raises rather than self-correcting.

Design-by-Contract — `@contract` / `Contract[In, Out]`: typed `LLMDataModel` I/O, `pre`/`act`/`post`, LLM **semantic validation**, and the **self-healing remediation loop** — is the most-used SymbolicAI capability and an explicit product requirement. The engine-redesign is not a SymbolicAI successor until it is rebuilt on the explicit-runtime architecture: the engine is passed as a bound handle, remediation cost is captured via the observer seam, and structured-output requests are kept. Port design and plan: [`../docs/fullreport/r7-contracts.md`](../docs/fullreport/r7-contracts.md), [`../docs/fullreport/r8-contracts-plan.md`](../docs/fullreport/r8-contracts-plan.md); canonical impl on `dev`: `symai/strategy.py`, `docs/source/FEATURES/contracts.md`, `tests/contract/`.

### FP-09 — Conversation / chat / memory composition removed; only stateless `Function`/`Symbol` remain
**Severity: Medium** · `symai/components.py:15` · status: `repro-read`
`Conversation` (sliding-window memory, save/load state), `chat.py`, `memory.py` all deleted. No multi-turn state.

### FP-10 — Plugin / imports system removed with no path back
**Severity: Medium** · `symai/__init__.py` · status: `repro-read`
`symai/imports.py` (`Import` to load community packages) + `sympkg`/`symdev`/`symrun` gone; no dynamic-package ecosystem.

### FP-11 — CLI / interactive shell (`symsh`) and menu tooling removed
**Severity: Medium** · `symai/__init__.py` · status: `repro-read`
`shell.py`, `shellsv.py`, `menu/`, `misc/{console,loader}.py`, `endpoints/api.py` deleted; no CLI surface (see PKG-05).

### FP-12 — Logging / tracing / callbacks / metrics removed from the execution path
**Severity: Medium** · `symai/runtime/runtime.py:109-143` · status: `repro-read`, `self`
`main` had a logger in `backend/base.py`, Trace/Log/Output components, and `extended/metrics`. `Runtime.execute` emits **no logs** and exposes no hook. The only `logging` import in the whole library is `symai/prompts.py`. Combined with CLI-04/SEC-02, a failed request is opaque (UX-03).

### FP-13 — Config-file + env-var bootstrap removed *(userland-replaceable)*
**Severity: Low** · `symai/runtime/models.py:425` · status: `repro-read`, `self`
`main` auto-loaded `symai.config.json` + `NEUROSYMBOLIC_ENGINE_API_KEY` env. Now credentials must be passed in code. `grep environ|getenv|dotenv symai/` returns nothing. Deliberate (app owns config) but a real migration cost, and it makes CMDS.md's config instructions dead (DOC-03).

### FP-14 — Automatic retry / exponential backoff removed *(userland-replaceable)*
**Severity: Low** · `symai/backend/engines/language_model/openai.py:118` · status: `repro-read`
`main` had `strategy.py` retry params + `_retry_func`. Now `RateLimitError` + `retry_after` are surfaced but **nothing retries** — `httpx.HTTPTransport(retries=n)` retries connection establishment only, not 429/5xx (see UX-01).

### FP-15 — Response caching removed; provider prompt-caching unreachable *(userland-replaceable)*
**Severity: Low** · `symai/backend/engines/language_model/openai.py:137` · status: `repro-read`
`main`'s `cache()` decorator gone. Separately, OpenAI `prompt_cache_key`/`prompt_cache_retention` exist on the client model but are not exposed through the normalized request.

---

## K. Extensibility

### EXT-01 — Adding a provider forces a core fork across ≥4 sites; no registry/plugin seam
**Severity: High** · `symai/runtime/factory.py:71`; `symai/runtime/models.py` (`Provider` enum) · status: `repro-read`
`_FACTORIES` is a module-private frozen `MappingProxyType` with a bespoke `_create_<provider>_<capability>` closure per entry, and `Provider` is a closed StrEnum in the contract core. A new provider means editing the enum, adding a closure, adding a map entry, writing a client package + engine module, and updating a test that pins the exact matrix — no single registration point.

### EXT-02 — Future capability additions could be misrouted by a binary `else`
**Severity: Low** · `symai/runtime/factory.py:104-129`; `symai/runtime/runtime.py:45-59` · status: `repro-read`
Capabilities are intentionally limited to language models and embeddings. Within that closed set the resolver emits only valid values, so no current request is misrouted. The `else` assignment would silently treat a future third capability as embedding if someone widened `_Capability` without redesigning the slots. This is a future edit hazard, not evidence that a generic N-capability Runtime is currently required.

### EXT-03 — Bring-your-own-engine ownership is not a supported public contract
**Classification: unresolved product scope; excluded from scoreboard** · `symai/runtime/models.py:345`; `symai/backend/engine_handle.py`
`Runtime.__init__` can technically receive an `EngineHandle`, but exporting that close-owning handle would make sharing and double ownership representable. If custom engines become supported, the API needs an explicit factory and ownership-transfer contract plus a provider-identity policy. The current accidental injection seam is not a public extension guarantee.

### EXT-04 — Per-model behaviour is split between the client catalog and hardcoded engine frozensets
**Severity: Low** · `symai/backend/engines/language_model/openai.py:48`; `embedding/openai.py:32` · status: `repro-read`
Adding a model to a client `MODEL_SPECS` auto-flows into the engine, but capability facts also live in ad-hoc engine frozensets (`_HIGH_REASONING_EFFORT_MODELS`, `_DIMENSIONALITY_MODELS`) and constants (Cerebras `vision`), so adding a model needs a non-obvious second edit whose omission silently degrades behaviour.

### EXT-05 — Runtime cannot hold two independently keyed instances of the same model
**Severity: High** · `symai/runtime/models.py:432-442`; `symai/runtime/factory.py:93-109`; `symai/runtime/runtime.py:45-59` · status: `repro-read`
`RuntimeConfig`, `create_runtime`, and `Runtime` each have one scalar language-model slot and one scalar embedding slot. Two OpenAI `gpt-5.4` engines with different API keys or transport settings cannot coexist in one Runtime. Provider/model cannot serve as instance identity because those values intentionally collide, and credentials must never participate in identity. The required seam is an immutable named configured-instance collection; each name resolves to a separately owned handle/client.

---

## L. Test quality & coverage

### TST-01 — Embedding math and persistence have zero coverage
**Severity: High** · `symai/ops/primitives.py:2052,2276-2335,2337,2381,2430,2459+` · status: `measured`
`similarity()`, `distance()`, all 12 kernel handlers (`_kernel_gaussian` `:2276` … `_kernel_mmd` `:2334`), `calculate_mmd()`, `zip()`, and `save`/`load` persistence are entirely untested — a large, math-heavy, easy-to-get-wrong surface of the user-facing type.

### TST-02 — Semantic LLM-fallback of ~45 operator dunders is untested; only `+` is exercised
**Severity: Medium** · `symai/ops/primitives.py:179-1259` · status: `repro-read`
Each comparison/arithmetic dunder hardcodes a distinct operator token + request builder in its fallback path (e.g. `__gt__` → `compare_request(self, ">", other)`); a copy-paste error in any of ~45 would be invisible.

### TST-03 — OpenAI engine `_validate_request`: ~12 of ~14 unsupported-feature guards untested
**Severity: Medium** · `symai/backend/engines/language_model/openai.py:152-219`; `tests/.../test_openai.py:282` · status: `measured`
The parametrized test covers 5 cases (DeepSeek's covers 20). Coverage confirms most OpenAI guards are never hit.

### TST-04 — Operation-builder tests mirror the implementation (change-detector, not oracle)
**Severity: Low** · `tests/test_operations.py:485` · status: `repro-read`, `self`
Each builder assertion recomputes `language_request(...)` from the *same* prompt constants and example source used by production, so a wrong prompt in `prompts.py` passes on both sides. Leaves `symai/prompts.py` (58% covered) effectively unpinned.

### TST-05 — 79% aggregate coverage masks a 33% user-facing layer
**Severity: Low** · `symai/ops/primitives.py` · status: `measured`
Aggregate is weighted by near-100% runtime/clients; `ops/primitives.py` (2,511 lines — the whole Symbol operation surface) is **33%**, `prompts.py` 58%, `symbol.py` 76%. All 448 tests run in 0.91 s entirely in-memory (`httpx.MockTransport`) — **zero** integration/live tests, and fixtures hand-author `model: "gpt-5.4"` to echo exactly, so the suite bakes in the same assumption BUG-05 gets wrong.

---

## M. Documentation truthfulness

### DOC-01 — `CMDS.md` documents a testing workflow whose first command errors
**Severity: High** · `CMDS.md:46` · status: `repro-script`, `self`
Every test command uses `--engine-api=mock|live`, which is not a registered pytest option (no `conftest.py`/`pytest_addoption` anywhere). Reproduced: `uv run pytest tests/engines --engine-api=mock` → `error: unrecognized arguments: --engine-api=mock`. `CMDS.md` is **new on this branch**.

### DOC-02 — `CMDS.md` "Engine request API" points at deleted files and a nonexistent method flow
**Severity: Medium** · `CMDS.md:87` · status: `self`
References `backend/mixin/deepseek.py`, `engine_deepseekX_reasoning.py`, `tests/data/*`, and a `prepare→build_request→call_request` flow — all removed in the cutover.

### DOC-03 — `CMDS.md` instructs editing a `symai.config.json` the library no longer reads
**Severity: Medium** · `CMDS.md:9,49` · status: `self`
Tells the user to configure `.venv/.symai/symai.config.json`; config-file loading was deleted (FP-13) and it directly contradicts `docs/source/INSTALLATION.md:51`. (Note: the five `docs/source/*` runtime pages + README were verified accurate — every snippet executes — so the doc rot is isolated to `CMDS.md`.)

---

## N. Public API & packaging

### PKG-01 — Entire public API replaced but version frozen at `1.18.0`
**Severity: High** · `pyproject.toml:11` · status: `self`
HEAD is `1.18.0`, identical to `main`, yet exposes a mutually incompatible surface (`main`'s `__all__` had `Symbol`/`Expression`/`Function`/`Conversation`; HEAD exports only runtime types). A `pip install -U` is a silent, un-versioned breaking change; the wheel is effectively unpublishable at this version.

### PKG-02 — Legacy imports fail without migration guidance
**Severity: Medium** · `symai/__init__.py` · status: `repro-script`, `self`
`from symai import Symbol` now raises `ImportError: cannot import name 'Symbol'` (verified) while the package still reports the old version and ships no migration note. A clean major-version cutover does not require a runtime forwarding shim, but it does require an explicit public import contract and migration guide so removed names and replacements are discoverable.

### PKG-03 — `symai.__version__` / `symai.SYMAI_VERSION` removed → `AttributeError`
**Severity: Medium** · `symai/__init__.py` · status: `self`
Both were in `main`'s `__all__`; neither exists now. Tooling that introspects the version breaks.

### PKG-04 — `numpy` pinned `<=2.1.3` after torch removal blocks install on newer Pythons
**Severity: Medium** · `pyproject.toml:27` · status: `repro-read`
The cap was inherited from `main` where `torch<2.10.0` forced an old numpy. Torch is gone (0 hits), so the cap is now an unmotivated ceiling that blocks Python 3.14 despite `requires-python` allowing it. (Also: numpy is a heavy hard dependency used only by the embedding-math primitives — reconsider making it optional.)

### PKG-05 — All seven console scripts removed with no replacement CLI
**Severity: Low** · `pyproject.toml` (no `[project.scripts]`) · status: `self`
`symchat`/`symsh`/`sympkg`/`symdev`/`symrun`/`symconfig`/`symserver` gone; on upgrade they vanish from users' PATH silently (see FP-11).

---

## O. Error-handling UX

### UX-01 — No retry/backoff for 429 or 5xx anywhere
**Severity: High** · `symai/runtime/factory.py:159` (`httpx.HTTPTransport(retries=connect_retries)`) · status: `repro-read`
Connection-establishment retries only — never response retries. `RateLimitError.metadata.retry_after` is parsed and surfaced but nothing consumes it. A regression vs `main` (FP-14) and a burden pushed entirely to callers with no helper.

### UX-02 — `Retry-After` parser only accepts delta-seconds; the RFC-allowed HTTP-date form is dropped to `None`
**Severity: Low** (reported Medium) · `symai/clients/_headers.py:5` · status: `repro-script`, `self`
`parse_optional_float('Wed, 15 Jul 2026 12:00:30 GMT')` → `None` (RFC 9110 permits both forms). Reproduced. Downgraded because the runtime performs no internal retry (graceful degradation to a valid `Optional`), but it defeats the only backoff signal a user's own retry loop is given.

### UX-03 — Provider error body + request_id captured but dropped from the message; nothing is logged
**Severity: Medium** · `symai/backend/engines/language_model/openai.py:132` · status: `repro-read`
Non-401/429 errors raise `ExecutionError(f"...failed with status {code}")` — status code only. The provider's JSON body (the actual *why*) and request ID are captured in `ErrorMetadata`/`body` but never surfaced as bounded structured details, and the library logs nothing (FP-12), so a failed request is opaque to debug.

---

## P. Concurrency & performance

### PERF-01 — Embedding response normalization re-validates and re-copies the entire float payload 3+ times
**Severity: Medium** (reported High) · `symai/backend/engines/embedding/openai.py:159-163`; `symai/runtime/models.py:385-392` · status: `repro-script`, `measured`
The same float array is fully materialized and walked ≥3 times after the raw JSON parse: (1) the client parses `EmbeddingData.embedding: tuple[float, ...]` (pydantic validates every float); (2) the engine rebuilds each vector as `EmbeddingVector(values=...)`, a frozen `strict=True` model whose `values` runs `allow_inf_nan=False` per-element **and** a before-validator doing a second O(dims) `type(v) is not float` scan per vector (`models.py:385-392`); (3) `operations.parse_embedding_response` walks every value again with `float(value)` and re-sorts. Measured ≈231–260 ms of pure GIL-bound CPU normalizing one max-batch response (2048 inputs × 3072 dims ≈ 6.29M floats) — work that dwarfs a fast embeddings round-trip and caps throughput. Downgraded from High to Medium (real cost, but only at large batch sizes on the common `Symbol.embed()` path).

**Resolution (FIXPLAN §11): fixed; remeasured 167.5 ms → 32.7 ms (5.1×) on the documented max-batch case.** Two of the three passes were removed. Pass (2)'s before-validator was deleted: measured in isolation it cost 75 ms, 2.4× more than pydantic-core's entire validation of the same 6.29M floats (31 ms), and it can never fire on the engine path because the client's own pydantic parse already yields `float`. Its only power beyond the `tuple[FiniteFloat, ...]` annotation was rejecting `int`/`Decimal` — values pydantic widens losslessly to the identical float, and which a third-party engine handing over raw `json.loads` output (`[0, 1, 0.5]` is valid JSON for a float vector) would have been falsely rejected for. Pass (3)'s `float(value)` walk was deleted: `EmbeddingVector.values` is already an immutable tuple of finite floats, so `parse_embedding_response` now returns those exact objects (`tuple[tuple[float, ...], ...]`, 0.3 ms) instead of copying them (59 ms). Pass (1) is the load-bearing parse and remains. No weakening: `strict=True` still rejects `str` and `bool`, `allow_inf_nan=False` still rejects `inf`/`nan`, and dimension, count, and index validation are untouched — pinned by `tests/test_performance.py`, whose benchmark is calibrated against a reference pass measured in-run (not a fixed millisecond budget) and was verified to fail if either removed pass is reintroduced.

### PERF-02 — Separate HTTP clients are intentional per-instance isolation
**Classification: withdrawn; excluded from scoreboard** · `symai/runtime/factory.py:139-161`
The original audit treated one client per capability as a wasted-pool issue. The approved instance model requires independent credentials, transport settings, ownership, and teardown, including multiple engines for the same provider/model. A fresh `httpx.Client` per configured engine is therefore deliberate. Registration metadata may be shared; live clients and handles are not deduplicated.

---

## Q. Persistence / file safety

### PERSIST-01 — `replace=False` can overwrite an existing pickle
**Severity: High** · `symai/ops/primitives.py:2477-2492` · status: `repro-script`, `self`
`save()` checks collisions against the path before adding `.pkl`. If `item.pkl` exists but `item` does not, `Symbol(...).save(\"item\", replace=False, serialize=True)` opens `item.pkl` with `\"wb\"` and overwrites it. Reproduced in a temporary directory: pre-existing bytes changed despite `replace=False`. The write is also non-atomic, so interruption can leave a corrupt artifact. The approved redesign removes built-in Symbol persistence rather than preserving this contract.

### SEC-03 — `Symbol.load()` executes arbitrary pickle payloads
**Severity: High** · `symai/ops/primitives.py:2497-2508` · status: `repro-read`
`load(path)` passes caller-selected bytes directly to `pickle.load()` without a trusted-input-only name or warning. Loading an attacker-controlled pickle executes arbitrary Python code. Moving the method to another namespace would not make it safe. Built-in persistence is removed; any future durable format requires a separate versioned, validated, non-executable codec design.

---

## Appendix — Round 1 "also-ran" observations (lower rank, not separately reproduced)

- **Name collision:** `symai.clients.openai.responses.ResponseError` (a pydantic *model*) vs `symai.clients.openai.errors.ResponseError` (an *exception*), both reachable from the package facade — `responses.py:396-398` vs `errors.py:27-31`.
- **`AssistantOutputMessage` constructs empty:** it omits `AssistantMessage`'s "content or reasoning required" validator — `models.py:200-224` vs `:363-369`.
- **`models.py` depends on pydantic's private `model_rebuild(_types_namespace=…)`** — `:141-143` (see CX-01).
- **No CI:** nothing runs the 448 tests or `ruff` on push (no `.github/workflows`).
- **`Function.batch()` is a sequential for-loop** — `components.py:150-170` — no concurrency and no provider batch API, so "batch" is a naming promise the implementation doesn't keep.
