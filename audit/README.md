# SymbolicAI `refactor/cleanup` — Deep Audit

**What changed:** a 341-file rewrite (+13,837 / −78,800) that replaces the old single-layer engine architecture with three explicit layers: hand-written HTTP clients, provider adapters, and a normalized typed Runtime. Most legacy capabilities and framework machinery were removed.

**Audited:** branch `refactor/cleanup` at `a220d6f` versus merge-base `da28e25`, on 2026-07-15. Provider/model count reduction was treated as intentional. Current code was subsequently re-read while reviewing the audit documents; the additional Symbol/persistence/runtime findings from that review belong in [`FINDINGS.md`](./FINDINGS.md).

**Document authority:**

- [`FINDINGS.md`](./FINDINGS.md) is the evidence register: observed behavior, severity, anchors, and verification limits.
- [`FIXPLAN.md`](./FIXPLAN.md) is the only implementation-order and release-gate source.
- [`SYMBOL_REDESIGN.md`](./SYMBOL_REDESIGN.md) is the approved Symbol/Function design.

---

## Bottom line

The retained runtime/client core is a better foundation than the old implicit global-engine system, but the branch is not ready to publish.

The immediate live-path blocker is the OpenAI response-model equality check: the adapter rejects a successful response when the provider returns a resolved dated model identifier rather than the configured alias. The same validation shape exists in DeepSeek and the OpenAI embedding adapter, although provider-specific echo behavior must be evidenced separately rather than asserted as identical.

The broader problem is contract placement:

- configured model identity is confused with provider-returned model identity;
- terminal filtered/refused responses cannot always be represented without text;
- capability declarations and imperative enforcement disagree;
- Function combines execution, parsing, defaults, limiting, Symbol wrapping, preview, and metadata modes;
- Runtime has one scalar slot per capability and cannot represent two independently keyed instances of the same model;
- Symbol combines value behavior with mutable prompt context, implicit semantic execution, graph state, embedding state, and executable persistence.

The green suite does not disprove these defects. The deterministic suite currently passes 448 tests, but all provider traffic is mocked and several fixtures encode the same assumptions as production. Provider-document claims also need durable URLs/access dates or bounded live canaries; labels such as `docs` are not independently reproducible evidence by themselves.

---

## Verdict by dimension

| Dimension | Verdict | Reason |
|---|---|---|
| Architecture and layering | Strong core, incomplete composition | Client isolation is real; Runtime composition imports back through backend/provider implementations. |
| Provider success path | Failing | OpenAI model-echo validation rejects documented resolved identifiers; other legal terminal shapes are rejected. |
| Runtime ownership | Careful cleanup, wrong instance model | Construction/teardown is disciplined, but scalar slots cannot hold repeated same-model instances and ambient discovery obscures ownership. |
| Symbol value model | Redesign required | Mutable context, graph state, implicit I/O, persistence, and operator behavior coexist on one wrapper. |
| Function typing | Redesign required | `return_type` conflates execution with decoding and cannot faithfully express parameterized containers. |
| Contract coherence | Weak | Dead capability fields, write-only logprobs, unreachable multi-output machinery, and strict inbound submodels. |
| Error/security boundary | Weak | Malformed credentials can leak; provider error handling is opaque; current pickle loading executes trusted-file assumptions silently. |
| Tests | Fast but incomplete | Runtime/client coverage is high; the user-facing operation layer and real provider contracts are not adequately exercised. |
| Packaging/migration | Not publishable | A breaking public API remains at the old version with stale instructions and no settled final major surface. |

---

## What should survive

1. **Client isolation.** Modules under `symai/clients/**` do not depend on the higher-level runtime or Symbol surface.
2. **Pre-resolution and failure cleanup.** Configuration is resolved before transport allocation; partial construction closes completed resources in reverse order.
3. **Explicit Runtime lifecycle.** At-most-once teardown and grouped cleanup failures are a substantial improvement over implicit process-global ownership.
4. **Provider-neutral request/response models.** Strict application-owned request types and normalized outputs are the correct boundary, once inbound provider tolerance and terminal states are corrected.
5. **Caller-owned provider clients.** Each configured engine instance should retain its own key, transport settings, HTTP client, and cleanup.
6. **Deterministic local tests.** The existing suite is fast enough to remain the default inner loop; live canaries complement rather than replace it.

---

## Newly confirmed omissions from the original audit

The documentation review reproduced additional defects in the current Symbol/persistence surface:

- `save(..., replace=False, serialize=True)` can overwrite an existing `.pkl` because collision detection occurs before the extension is added.
- `load()` calls `pickle.load()` on a caller-selected file with no trusted-input warning, permitting arbitrary code execution.
- equal Symbols can produce different hashes, and a Symbol's hash can change after item mutation.
- native indexing/item operations replace `KeyError`, `IndexError`, and `TypeError` with generic `Exception`.
- Runtime configuration cannot hold two language engines for the same provider/model with different API keys.

The approved design removes Symbol persistence and mutation, makes Symbol unhashable, preserves native exception types, and introduces named independently owned engine instances.

---

## Ratified design direction

### Runtime

- Runtime is passed explicitly; ambient `current_runtime()` discovery is removed.
- Named configured engine instances are the selection identity.
- The same provider/model may appear multiple times with different API keys or transports.
- Each instance owns a separate `httpx.Client`; no client pooling/deduplication is planned.
- Omitted selection uses an explicit default or the sole matching engine; otherwise it fails as ambiguous.
- Synchronous Runtime is single-owner-thread. A future async API is designed separately.

### Function and decoding

- Function constructs/executes a request and returns `LanguageModelResponse`.
- Function is not generic and has no `return_type`.
- Typed conversion is an explicit `Decoder[T]` / `decode_output(...) -> T` stage.
- Request preview is a named `request()` method; metadata is always on the response.

### Symbol and operations

- Symbol remains as a shallow-immutable ergonomic value DSL.
- Native deterministic operators never call a model.
- Semantic operations are explicit free functions taking Runtime and Symbol and returning a new Symbol.
- Static/dynamic/global context, `adapt`, `.sem`, graph/linker state, persistence, embedding cache, and implicit fallback are removed.
- Expression and Result are removed; ordinary callables compose operations.

### Explicit non-goals

- Tool/function calling is not supported and is not on the roadmap.
- No payload-less `tool_calls` finish reason is added to the normalized contract.
- No shared HTTP client pool.
- No speculative N-capability runtime without an approved third capability.
- No built-in persistence or pickle compatibility API.

---

## Highest-priority correctness work

The detailed dependency order and acceptance criteria live only in [`FIXPLAN.md`](./FIXPLAN.md). In summary:

1. restore provider success-path correctness and credential safety;
2. introduce named engine instances, explicit Runtime passing, and enforced synchronous ownership;
3. split Function execution from typed decoding;
4. cut over to the immutable Symbol/explicit operation design;
5. harden inbound provider parsing, per-model policy, safe errors, and observability;
6. consolidate shared mechanics without coupling provider schemas or live instance ownership;
7. expand deterministic coverage, add bounded provider canaries, verify packaging/docs, then publish one major release.

There is no intermediate “minimum shippable” subset that publishes the old public version and immediately breaks it again for the Symbol redesign. The final major surface settles first.

---

## Audit method

The original audit used four rounds:

1. architecture and branch-delta reconstruction;
2. independent correctness, contract, provider, test, packaging, security, extensibility, and performance lenses;
3. provider-document comparison;
4. reproduce-or-refute passes for correctness findings.

The documentation review then cross-checked the proposed fixes against the code and ran targeted reproductions for the omitted persistence/hash/exception defects. The current deterministic suite, Ruff, and Pyright were also run successfully during that review.

### Evidence limits

- No live API key was used during the audit or documentation review.
- Exact provider-returned model strings and network failure behavior were not observed end to end.
- Provider-document claims without retained URLs/access dates need stronger provenance.
- Throwaway reproductions are weaker than checked-in regression tests; release-relevant ones should become deterministic tests.
- Passing line coverage is not proof that fixtures model the real provider contract.

These limits narrow what may be claimed; they do not refute code-side defects that were reproduced locally or exact validation paths confirmed by inspection.
