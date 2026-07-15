# SymbolicAI `refactor/cleanup` — Implementation Sequence

This document is the single source of truth for implementation order, dependencies, acceptance, and release readiness. [`FINDINGS.md`](./FINDINGS.md) owns the evidence; [`README.md`](./README.md) summarizes it; [`SYMBOL_REDESIGN.md`](./SYMBOL_REDESIGN.md) defines the approved value/operation design.

---

## 1. Ratified decisions

1. **Runtime and Function are explicit.** No `current_runtime()` or ambient `ContextVar` discovery survives.
2. **Configured engine instances have names.** Provider/model is not identity; the same provider/model may be configured multiple times with different credentials and transports.
3. **Every configured engine owns a separate `httpx.Client`.** Clients are not pooled or deduplicated across engine instances.
4. **Synchronous Runtime has one owner thread.** A future async API is a separately designed `AsyncRuntime`.
5. **Function returns `LanguageModelResponse`.** Typed output conversion is a separate `Decoder[T]` stage; `return_type` does not define Function.
6. **Symbol remains as a shallow-immutable ergonomic value DSL.** It has no Runtime dependency, mutable context, graph, persistence, embedding cache, or semantic mode.
7. **Semantic operations are explicit free functions.** They take Runtime and Symbol and return new Symbols.
8. **Expression and Result are removed.** Ordinary callables compose operations.
9. **Static/dynamic context concepts are removed.** Function instructions and call values/messages are sufficient.
10. **Built-in persistence is removed.** There is no pickle compatibility API.
11. **Tool/function calling is unsupported and outside the roadmap.** No tool request/output contract or payload-less `tool_calls` normalization is added.
12. **The result is one clean major-version cutover.** No forwarding shims preserve the old god-object surface.

---

## 2. Global invariants

### Ownership and identity

- An engine instance is identified only by its configured name within a Runtime.
- Names are globally unique within one Runtime and never contain or derive from secrets.
- Equal provider/model configurations remain distinct instances.
- Each instance exclusively owns its engine handle, provider client, transport settings, credentials, and HTTP client.
- No raw engine, client, or ownership-bearing handle escapes Runtime.
- Construction resolves all configurations before allocating any transport.
- Partial construction closes completed resources in reverse order.
- Runtime close attempts every cleanup exactly once and reports all failures.

### Execution

- Runtime is passed explicitly to Function and semantic operations.
- Engine selection is explicit when ambiguous.
- A synchronous Runtime may execute only on the thread that entered it.
- Native Symbol operations never execute a provider request.
- Semantic operation names never accept provider/model selection; they forward only a configured engine name.

### Contracts

- Incoming provider data is tolerant of additive fields but preserves required semantic invariants.
- Requested model identity and provider-returned model identity are recorded separately when they differ.
- Terminal filtered/refused output is representable without fabricated text.
- Execution, provider response normalization, output decoding, and ergonomic Symbol wrapping are separate stages.
- Errors remain structured and safe: no secret, prompt, full response body, or arbitrary provider payload enters routine exception messages or logs.

### Scope

- Only language-model and embedding capabilities are designed here.
- Provider implementation registration and configured engine-instance identity are separate mechanisms.
- No speculative N-capability runtime, tool calling, streaming, async execution, or persistence codec is smuggled into this sequence.

---

## 3. Dependency order

```text
Provider-path correctness and credential safety
                    │
                    ▼
Named engine instances + explicit Runtime ownership
                    │
                    ▼
Function execution / Decoder separation
                    │
                    ▼
Immutable Symbol + explicit semantic operations
                    │
                    ▼
Inbound contract and per-model hardening
                    │
                    ▼
Shared client/adapter mechanics + lazy provider loading
                    │
                    ▼
Coverage, live canaries, documentation, major release
```

Provider-path correctness can proceed independently of the user-facing redesign. The Runtime boundary must settle before Function and operations are migrated. Function/decoder separation must settle before the Symbol surface is cut over.

---

## 4. Restore provider-path correctness and credential safety

### Response model identity

Remove exact requested/returned model equality as a success-path rejection. Do **not** replace it with arbitrary `startswith()`: `gpt-4.1-mini-*` also starts with `gpt-4.1`.

Preserve both facts:

- configured/requested model;
- provider-returned resolved model, when present.

A successful response is normalized unless the provider response is internally malformed. If provider-specific canonical identity validation is later required, it needs an explicit alias/snapshot relation, not string-prefix coincidence.

**Addresses:** BUG-05/API-01 and BUG-06.

**Acceptance:** dated-snapshot responses normalize; a different but prefix-sharing model is not silently treated as proof of identity; metadata exposes the returned model without losing the configured model.

### Filtered/refused terminal output

Represent terminal states directly. A provider may return `content: null`, a content-filter/refusal finish reason, and no refusal text. The normalized contract must not require fabricated content merely to satisfy a validator.

Prefer an output state model whose invariant permits:

- successful content/reasoning;
- explicit refusal text when supplied;
- empty provider text only when a terminal filtered/refused status explains it.

**Addresses:** BUG-07.

**Acceptance:** a content-filtered response is distinguishable from transport/schema failure even when no refusal string exists.

### API-key boundary

Validate credential header safety at the shared client/header construction boundary, not only in `ProviderEngineConfig`, because direct client users otherwise bypass the check. Reject control characters and unsafe whitespace without including the raw key in a validation error.

**Addresses:** SEC-01.

**Acceptance:** malformed keys never reach httpcore; direct clients and Runtime-created clients behave identically; credential validation has no credential-derived exception text or arguments. Tests compare failures from distinct invalid credentials rather than requiring arbitrary credential bytes to be absent from traceback boilerplate, because a credential can equal structural text such as `ValueError\n`.

---

## 5. Introduce named engine instances and explicit Runtime ownership

### Configuration

Replace scalar capability slots with immutable named instance collections:

```python
RuntimeConfig(
    language_models=(
        NamedEngineConfig(
            name="tenant-a",
            provider=Provider.OPENAI,
            model="gpt-5.4",
            api_key=key_a,
        ),
        NamedEngineConfig(
            name="tenant-b",
            provider=Provider.OPENAI,
            model="gpt-5.4",
            api_key=key_b,
        ),
    ),
    embeddings=(...),
    default_language_model="tenant-a",
    default_embedding=None,
)
```

Validate before allocation:

- at least one configured instance;
- nonempty globally unique names;
- provider/capability/model support;
- defaults exist and match their capability;
- no silent normalization or overwrite of duplicate names.

Keep names opaque, case-sensitive application identifiers unless a real interoperability requirement justifies a stricter syntax.

### Selection

Add keyword-only engine selection to typed execution overloads:

```python
runtime.execute(language_request, engine="tenant-b")
runtime.execute(embedding_request, engine="embed-primary")
```

Resolution rules:

1. An explicit name must exist and match the request capability.
2. Otherwise use the configured capability default.
3. Otherwise use the sole matching engine.
4. Otherwise raise an ambiguity error listing safe engine names.
5. If no engine provides the request capability, preserve `UnsupportedCapabilityError`.

Use distinct structured errors for unknown name, wrong capability, and ambiguity. Never expose credentials.

### Ownership and transport

Construct one `httpx.Client` per named instance. Preserve each instance's independent key, timeout, retry, pool, and teardown. Do not cache or deduplicate clients by provider/model.

Internally Runtime owns a read-only name-to-tagged-handle map and a reverse construction order. It may expose immutable instance metadata later, but never raw handles or clients.

### Explicit lifecycle

Remove `_CURRENT_RUNTIME` and `current_runtime()`. `with Runtime` controls lifecycle only; consumers receive Runtime explicitly.

Record the owner thread when entry succeeds. Reject foreign-thread `execute`, active `close`, or `__exit__` before touching state or handles. Once affinity is enforced, synchronous same-Runtime overlap is illegal and the `Condition`, `_in_flight`, `CLOSING` drain, and `EngineHandle` lock can be removed.

Close must:

- mark/detach state before external cleanup;
- remain idempotent;
- attempt every handle in reverse order;
- aggregate cleanup failures;
- work for a constructed Runtime that was never entered.

**Addresses:** BUG-01, BUG-02, BUG-03, CX-04, CX-05, EXT-02's scalar-slot aspect, and the newly recorded same-model-instance gap.

**Acceptance:** two same-provider/same-model language instances with different keys create distinct clients, select deterministically by name, and close exactly once; ambiguous omission fails; foreign-thread access fails clearly; independent runtimes can run on independent threads.

---

## 6. Separate Function execution from typed decoding

### Function

Make Function non-generic. Remove `return_type`, `sym_return_type`, `default`, `limit`, `preview`, and `return_metadata` from execution.

- `request(...) -> LanguageModelRequest` builds without I/O.
- calling Function with an explicit Runtime returns `LanguageModelResponse`.
- `execute_many(...)` returns responses in stable input order and is documented as sequential execution.
- response metadata is always available on the response.

Function holds immutable instructions and normalized request options. It has no static/dynamic context fields.

### Decoding

Add a separate generic `Decoder[T]` protocol and `decode_output(...) -> T`.

Standard decoding must cover:

- text normalization;
- constructor-based scalar conversion;
- booleans with explicit accepted forms;
- Pydantic models;
- nested parameterized containers through `TypeAdapter`;
- deterministic output-index selection;
- defaults that catch only decode failures;
- deterministic list/tuple/dict limiting;
- set pass-through.

Do not infer a Runtime decoder from `Function[T]`, `__orig_class__`, or a redundant generic plus `return_type` pair.

**Addresses:** BUG-10's set failure, BUG-11's surprising recursive literal coercion, the current Function typing conflation, and part of TST-04/TST-05.

**Acceptance:** Function has one response shape; typed inference is demonstrated by a real static-check fixture; nested containers and Pydantic models decode without bare-class loss; execution errors are never converted into defaults.

---

## 7. Cut over to immutable Symbol and explicit operations

Apply [`SYMBOL_REDESIGN.md`](./SYMBOL_REDESIGN.md) as one coherent public-surface change.

### Keep

- shallow-immutable `Symbol[T].value`;
- deterministic native operators;
- mixed Pythonic result contract: comparisons/casts are raw, arithmetic/indexing return new Symbols;
- explicit semantic namespaces that take Runtime and Symbol and return new Symbols.

### Remove

- Expression and Result;
- graph/linker/root/node/edge/result retention;
- semantic flags and implicit fallback;
- `.sem` / `.syn`;
- static/dynamic/global context and `adapt`/`clear`;
- embedding cache on Symbol;
- `sym_return_type`;
- `__setitem__` / `__delitem__` and other Symbol mutation;
- broad wrapped-value `__getattr__` forwarding;
- Symbol persistence and pickle loading;
- operation mixins as Symbol bases.

Make Symbol unhashable. Invalid native indexing/operators propagate original Python exceptions.

Semantic operations require configured engine names only when Runtime selection would otherwise be ambiguous. They do not accept provider/model options and never mutate inputs.

**Addresses:** BUG-09 through BUG-12, TST-01/TST-02's retained surface, the graph-retention observation, and the newly recorded hash/persistence/exception defects.

**Acceptance:** Symbol has no runtime/provider/client import or mutable internal state; no native operator performs I/O; every retained operator has a result-type matrix test; removed APIs have no aliases or forwarding shims; old pickle artifacts are not loaded by the library.

---

## 8. Harden inbound contracts and per-model behavior

### Tolerant inbound parsing

Use tolerant models for provider-owned additive response objects. Preserve unknown values when callers may need diagnostics; do not turn every open provider string into a closed enum parse failure.

Handle reasoning-only truncation and multi-phase output without requiring exactly one assistant message. Required normalized invariants still fail closed.

**Addresses:** CLI-01, CLI-02, CLI-03, CLI-05, BUG-08, and API-03.

### Provider capability corrections

Align only behavior the normalized product claims to support:

- correct per-model OpenAI reasoning efforts;
- correct Cerebras vision and reasoning controls per model;
- remove nonexistent finish-reason mappings;
- decide whether parameters a provider silently ignores should be rejected as ineffective rather than advertised as supported;
- validate only universal model limits, not account-tier entitlements unavailable locally.

`finish_reason="tool_calls"` remains unsupported. Since no tool request/output contract exists, do not add a normalized finish reason without its payload. If it appears unexpectedly, raise a precise unsupported-response error.

**Addresses:** API-02 and API-05 through API-13 after removing tool-only/non-actionable claims from the defect tally.

---

## 9. Improve safe error handling and operability

### Structured provider errors

Parse bounded provider error fields into typed metadata:

- HTTP status;
- provider error code;
- safe provider message;
- parameter name when safe;
- request ID;
- retryability classification.

Do not copy complete response bodies, prompts, credentials, or arbitrary provider payloads into routine exception strings or logs. Bound stored body size if raw diagnostics are retained behind an explicit opt-in.

Distinguish authentication, permission, invalid request, rate limit, provider failure, transport failure, and invalid response when callers react differently.

### Retry policy

Do not add generic automatic retries for POST requests. Any retry layer must first define:

- operation idempotency or an idempotency key;
- retryable failure classes;
- bounded attempts and total elapsed time;
- exponential backoff with jitter;
- `Retry-After` precedence and clamp;
- cancellation/deadline behavior;
- attempt metadata.

Until that contract exists, surface retry metadata to applications rather than risking duplicate execution or billing.

### Telemetry

Library logging is structured, bounded, and opt-in at normal Python logging levels. Never log request/response bodies, prompts, credentials, or arbitrary error bodies. Stable safe fields include provider, configured engine name, model, status, request ID, duration, and attempt number when retries exist.

**Addresses:** CLI-04, UX-01 through UX-03, FP-12, and FP-14 after correcting their policy framing.

---

## 10. Consolidate mechanics without coupling provider schemas

Extract only mechanics whose divergence is a defect:

- bearer-header construction and credential validation;
- HTTP status/error classification;
- bounded response parsing;
- transport/error envelopes;
- retry-header parsing;
- engine execution/error translation;
- common OpenAI-compatible chat mechanics for Cerebras and DeepSeek.

Keep provider-owned request/response schemas and capability policy separate. OpenAI Responses remains structurally separate from OpenAI-compatible Chat Completions.

Move composition out of the runtime core to break the runtime/backend cycle. Lazy provider loading may use immutable lightweight descriptors, but every configured engine resolution constructs a fresh engine, provider client, and HTTP client. Registration metadata may be shared; live instances never are.

Provider implementation registration is a separate concern from named configured instances. Do not expose `EngineHandle` as an extension API. If bring-your-own-engine becomes supported, define an explicit factory and ownership-transfer contract first.

**Addresses:** SOC-01 through SOC-07, CLI-06, CLI-07, EXT-01, EXT-03, EXT-04, and the response-error name collision.

**Acceptance:** adding an OpenAI-compatible provider supplies a small schema/policy delta and registration descriptor; importing `symai` does not eagerly compile every provider schema; resolving two equal descriptors still creates two independently owned live instances.

---

## 11. Performance work after ownership is stable

Collapse redundant full-payload validation/copy passes in embedding normalization and remeasure the documented max-batch case.

Do not implement the former shared-client recommendation. Per-instance HTTP clients are intentional ownership isolation, not a performance defect.

**Addresses:** PERF-01. PERF-02 is withdrawn/reclassified as deliberate isolation.

**Acceptance:** embedding normalization materially improves under a retained benchmark without weakening finite-float, dimension, count, or index validation.

---

## 12. Verification and release gate

This section is the only release-gate source across the audit documents.

### Deterministic tests

- dated/resolved model identity and metadata;
- null filtered/refused terminal responses;
- credential redaction through direct and Runtime-created clients;
- named engine duplicate/default/unknown/wrong-capability/ambiguity paths;
- two same-model instances with different keys and distinct clients;
- owner-thread enforcement and independent-runtime parallelism;
- reverse cleanup and grouped cleanup failure;
- Function request/response shape and Decoder static inference;
- immutable Symbol operator/result matrix;
- removed Expression/context/persistence/implicit-semantic surfaces;
- tolerant inbound additive fields and unknown provider values;
- embedding math, kernels, and normalization benchmark.

### Provider canaries

Fixture tests cannot prove current provider behavior. Run a bounded live canary for every supported provider on a scheduled/manual CI path with secret availability and strict cost limits. A skipped environment-guarded test in ordinary CI is not sufficient evidence by itself.

Retain exact provider-document URLs, access dates, and captured safe schema facts for claims that cannot be exercised without credentials.

### Static and packaging checks

- lint and format checks;
- full pyright check plus dedicated inference fixtures;
- full deterministic suite;
- wheel build and clean-environment install;
- public import contract;
- documentation snippets executed against the final surface;
- version metadata exposed consistently.

### Compatibility and version

Complete the Runtime, Function/decoder, and Symbol cutovers before setting the release version. Publish as a new major version with a migration guide. Do not ship an intermediate major that immediately requires another major for the Symbol redesign.

Remove stale console/config/testing instructions rather than preserving nonfunctional shims. Reassess dependencies after Symbol graph, Box-based Result handling, persistence, and embedding state are removed; keep only dependencies used by the final surface.

### Release-ready means

- every Critical and release-relevant High finding in `FINDINGS.md` is closed or explicitly withdrawn with evidence;
- the named-engine and explicit-runtime contracts work end to end;
- no secret or executable persistence boundary remains;
- no unsupported tool-calling contract is advertised;
- tests, canaries, static checks, packaging, and executable documentation pass;
- the README and reference docs describe the exact shipped public API.

---

## 13. Finding coverage policy

`FINDINGS.md` remains the canonical register. This implementation sequence references finding IDs only to preserve traceability; it does not duplicate severity, status, anchors, or reproduction evidence.

Intentional exclusions and withdrawn claims are not implementation obligations. In particular:

- tool calling is an explicit non-goal, not a feature-parity defect;
- separate HTTP clients are deliberate instance isolation, not a pooling defect;
- account-tier limits are not statically knowable model facts;
- provider-accepted-but-ignored parameters require a product-policy decision, not automatic pass-through;
- future third-capability hazards do not justify a generic runtime before a third capability is approved.
