# R8 · Contracts port — implementation plan

Builds on [`r7-contracts`](./r7-contracts.md) (why contracts are a required pillar + faithful semantics).
Decisions locked in this revision:

- **`Contract[In, Out]` is a clean, modern object that exposes NONE of the legacy surface.**
- **`@contract` is a thin backward-compat shim** — the *only* place the old ergonomics
  (`forward` fallback, `contract_successful`/`contract_result`/`contract_exception`,
  `contract_perf_stats()`, `graceful`) are reconstructed, by adapting a native `Contract` underneath.
- **`LLMDataModel` is carried over** as ordinary Pydantic.

Plan style: intent + interfaces + edge cases + tests. Literal code only for the load-bearing signatures.

## 1. The two surfaces, kept strictly separate

**Native (new, clean).** A `Contract[In, Out]` is an immutable, engine-agnostic *spec*: a fixed
instruction, typed input/output models, `pre`/`act`/`post` hooks, and remediation options. You invoke it
with an explicit engine handle + input. It has **no `forward`, no mutable `self.contract_*` state, no
`contract_perf_stats()`, no `graceful` flag**. It either returns a validated `Out` or raises; a
`.run(...)` variant returns an immutable `ContractResult` for callers who want telemetry / non-raising.
That is the whole surface — nothing legacy.

**Shim (backcompat only).** `@contract(...)` reads an old-style class (`prompt`/`pre`/`act`/`post`/`forward`
+ annotations), builds a native `Contract` from it, and returns a wrapper that **re-creates the legacy
behaviors on top**: the always-run-`forward` fallback, the mutable `contract_successful` /
`contract_result` / `contract_exception` attributes, `contract_perf_stats()`, and mapping the legacy
`graceful=True` to the native `.run()` path. All legacy concepts live here and nowhere else.

## 2. Where it lives (module map + layering)

```
symai/contract/
  __init__.py     (empty, per repo convention)
  models.py       LLMDataModel (ported: prompt-render + json-schema helpers)
  contract.py     Contract[In, Out], ContractResult, ContractOptions, RetryParams, ContractViolation
  validation.py   type validation + semantic-condition checker + remedy-prompt builders
  remedy.py       the bounded retry/backoff remediation loop
  decorator.py    @contract shim + the legacy wrapper (all backcompat lives here)
```

Altitude = same as `symai/ops`. **Imports allowed:** `symai.function.Function`, `symai.runtime` (the `r5`
handle, `LanguageModelRequest`, `JsonSchemaResponseFormat`, response types), `symai.contract.models`.
**Forbidden:** importing `symai.providers.*` — reach the model only through the handle. Nothing in
`function`/`runtime`/`providers` may import `symai.contract` (enforce in `tests/test_import_boundaries.py`).

## 3. Native types (load-bearing signatures) — no legacy anywhere

```python
# symai/contract/contract.py
class ContractViolation(Exception):
    def __init__(self, stage: Literal["pre", "post", "type"], errors: tuple[str, ...]): ...

class RetryParams(BaseModel):            # typed Pydantic, NOT a dict; NO `graceful` (a legacy suppression knob)
    model_config = ConfigDict(frozen=True)
    tries: int      = Field(default=8,     ge=1)
    delay: float    = Field(default=0.015, ge=0)
    max_delay: float = Field(default=0.25, ge=0)
    jitter: float   = Field(default=0.0,   ge=0)
    backoff: float  = Field(default=1.25,  ge=1)

@dataclass(frozen=True, slots=True)
class ContractOptions:
    pre_remedy: bool = False
    post_remedy: bool = True
    accumulate_errors: bool = False
    retry: RetryParams = field(default_factory=RetryParams)

@dataclass(frozen=True, slots=True)
class ContractResult[Out]:               # immutable; returned by .run() only
    value: Out | None
    succeeded: bool
    attempts: int
    errors: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class Contract[In: LLMDataModel, Out: LLMDataModel]:
    instruction: str
    input_type: type[In]
    output_type: type[Out]
    pre:  Callable[[In], None]  = _noop         # Q1=A: pure predicate, raises (with a message) on violation
    act:  Callable[[In], In]    = _identity     # the only mutation point
    post: Callable[[Out], None] = _noop         # Q1=A: pure predicate, raises on violation
    semantic_conditions: tuple[str, ...] = ()   # Q2=A: first-class; LLM-judged, distinct from post
    options: ContractOptions = field(default_factory=ContractOptions)

    # raises ContractViolation when remediation is exhausted:
    def __call__(self, engine: LanguageModel, input: In, *, remedy: LanguageModel | None = None) -> Out: ...
    # never raises for a contract failure — telemetry/graceful path:
    def run(self, engine: LanguageModel, input: In, *, remedy: LanguageModel | None = None) -> ContractResult[Out]: ...
```

The clean design expresses "graceful" as a *method choice* — `__call__` raises, `.run()` returns a
`ContractResult` — instead of a boolean flag that suppresses exceptions and leaves the object in a
half-valid state. There is no user `forward` in the native object: **generation is intrinsic** to the
Contract, not a method the user overrides.

```python
# symai/contract/models.py
class LLMDataModel(BaseModel):           # ported; ordinary (mutable) Pydantic
    # helpers: render-for-prompt (uses Field descriptions), json schema for response_format
    ...
```

## 4. Native pipeline (faithful semantics, no fallback)

`Contract.run(engine, input, remedy=None)`:
1. **Input** — `input` is typed `In`; run `pre(input)`. On failure: if `options.pre_remedy`, enter the remedy
   loop to LLM-correct the input; else the run fails at the `pre` stage.
2. **Act** — `input = act(input)`.
3. **Generate** — `Function(instruction)` rendering `input` (LLMDataModel prompt helper) with
   `response_format = JsonSchemaResponseFormat(schema=Out.model_json_schema())`; execute on `engine`; parse
   `response.text` → `Out` via Pydantic.
4. **Validate output** — Pydantic type validation + `post(output)` + `semantic_conditions` (an internal
   validator `Function`, ported `_check_semantic_conditions`, on `remedy or engine`). On failure: if
   `options.post_remedy`, run the **remedy loop** — errors → corrective prompt (accumulating if
   `accumulate_errors`), re-generate up to `retry.tries` with exponential backoff
   (`min(max_delay, delay*backoff**n)` ± `jitter`), on the `remedy` handle if given.
5. **Result** — `.run()` returns `ContractResult(value, succeeded, attempts, errors)`; `__call__` returns the
   `Out` or raises `ContractViolation(stage, errors)` when `succeeded` is false. **No `forward` is ever
   called** — that concept does not exist in the native object.

## 5. The shim (`decorator.py`) — where ALL legacy lives

`@contract(pre_remedy=..., post_remedy=..., accumulate_errors=..., graceful=..., remedy_retry_params=..., verbose=...)`
on a class `C`:
- extract `instruction` from `C.prompt`; `In` from the `pre`/`forward` input annotation; `Out` from the
  `forward` **return** annotation (native `LLMDataModel`, or a Python type auto-wrapped into a single-`value`
  model — port the dynamic-wrap helper); bind `pre`/`act`/`post` from methods; build a native `Contract` +
  `ContractOptions` (translating the legacy `remedy_retry_params` dict into `RetryParams`, dropping `graceful`).
- return a **legacy wrapper** whose `__init__(engine, *, remedy=None)` stores the handles, and whose
  `__call__(input)` / `forward(input)`:
  - call `contract.run(engine, input, remedy=remedy)`;
  - store `contract_successful` / `contract_result` / `contract_exception` from the `ContractResult`;
  - if it failed: **run the user's original `forward`** (the legacy fallback) with the (validated-or-original)
    input, honoring the old rule; if legacy `graceful=False`, surface the failure as before;
  - expose `contract_perf_stats()` synthesized from the `r6` observer records for that run.

So today's code keeps working unchanged, and every legacy affordance is confined to this wrapper:

```python
@contract(post_remedy=True)
class Classify:
    @property
    def prompt(self) -> str: return "Classify the sentiment."
    def pre(self, input: Review) -> None: ...
    def post(self, output: Verdict) -> None: ...
    def forward(self, input: Review) -> Verdict: ...      # legacy fallback (shim-only)

with load_runtime(cfg) as rt:
    verdict = Classify(rt.language_model("smart"))(Review(text="..."))
```

The native form has none of that:

```python
classify = Contract(
    instruction="Classify the sentiment.",
    input_type=Review, output_type=Verdict,
    post=check_confidence,
    options=ContractOptions(post_remedy=True),
)
with load_runtime(cfg) as rt:
    verdict: Verdict = classify(rt.language_model("smart"), Review(text="..."))   # or classify.run(...) for telemetry
```

## 6. Interactions & dependencies

- **`r5` handles** — the explicit engine, plus a separate `remedy` handle (the old `dynamic_engine`).
- **`r6` observer seam** — each generation/remedy attempt is one `runtime.execute`, so usage/cost/latency is
  captured there; the shim's `contract_perf_stats()` reads from it. The native Contract does **not**
  self-instrument.
- **Structured-output requests** — depends on `JsonSchemaResponseFormat` + schema-from-`Out`; the
  simplification tail must keep that feature (the `JsonObject`→`pydantic.JsonValue` change is fine).
- **No `Expression`, no ambient engine, no `static/dynamic_context`** — the native object is a pure immutable
  spec invoked with explicit handles.

## 7. Phased implementation

1. **`LLMDataModel`** (`models.py`) + prompt-render/schema helpers; unit-test rendering & schema.
2. **Validation core** (`validation.py`): Pydantic type validation over response text; remedy-prompt builder;
   semantic-condition checker (`Function` judge). Test with a mock engine.
3. **Remedy loop** (`remedy.py`): bounded retry/backoff over an **injected generator fn** (so it's testable
   without a network), `accumulate_errors`.
4. **`Contract[In, Out]`** (`contract.py`): wire 1–3; `__call__` (raises) + `.run()` (`ContractResult`).
5. **`@contract` shim** (`decorator.py`): class introspection → native Contract; legacy wrapper with
   `forward` fallback + `contract_*` attrs + `contract_perf_stats()` + `graceful`/`remedy_retry_params`
   translation; native-type auto-wrap/unwrap.
6. **Boundary + parity tests**: import-boundary guard; port `dev` `tests/contract/*` *behaviors* against the
   shim (backcompat) and add native-API tests.
7. **Docs**: object-first `contracts.md`; decorator shown as backcompat.

## 8. Testing strategy

- **Native API** (new): `__call__` returns `Out` / raises `ContractViolation`; `.run()` returns immutable
  `ContractResult` and never raises; engine-agnostic reuse (one `Contract`, two handles); `post` +
  `semantic_conditions` remediation; retry exhaustion; `accumulate_errors`; static typing proves
  `Contract[In, Out].__call__ -> Out`; observer captures N calls for an N-try heal.
- **Shim** (backcompat): port `dev`'s behaviors — `forward` fallback runs on failure, `contract_successful`/
  `contract_result`/`contract_exception` set correctly, `graceful=True` maps to the non-raising path,
  `contract_perf_stats()` present, native-type wrap/unwrap.
- **Determinism**: the remedy loop takes an injected generator, so no test needs the network. A small opt-in
  live canary complements it.

## 9. Decisions (locked)

1. **`pre`/`post` are pure predicates that raise** on violation (the message feeds the remedy prompt);
   `act` is the only mutation point. **(Q1 = A)**
2. **Semantic conditions are a first-class `semantic_conditions: tuple[str, ...]`** on the native Contract,
   judged by an internal validator `Function` distinct from `post`. **(Q2 = A)**
3. **`RetryParams` is a typed Pydantic frozen model, not a dict**, with the code's defaults
   (`tries=8, delay=0.015, max_delay=0.25, jitter=0.0, backoff=1.25`) + `Field` validation; overridable per
   contract. **(Q3 = B, as a Pydantic type)**
4. **Rendering + output schema come from the `LLMDataModel` type itself** — `Contract` just takes the
   `In`/`Out` `LLMDataModel` subclasses (which already know how to render to a prompt and emit a JSON schema
   via `model_json_schema()`); no separate or pluggable renderer. **(Q4 = A)**

_(The old "always run `forward` fallback" is shim-only; the native Contract hard-fails via `__call__` or
reports via `.run()`.)_
