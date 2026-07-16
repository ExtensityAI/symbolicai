# R5 · Engine-handle ergonomics (selection API)

Design note, authored in the main loop after a usage review surfaced a wart no Round-1/2/3
lens caught: the `r1-08` API-surface lens accepted the `engine="name"` keyword as fine and
never challenged the **two-argument** selection shape. This note proposes a bound **handle**
that keeps every design principle while removing the wart. Feature impact: **keeps-all**.

## 1. The wart

Today every I/O operation takes the runtime **and** a stringly-typed engine name:

```python
with runtime:
    s = text.summarize(runtime, doc, engine="fast")
    a = reason.query(runtime, doc, "thesis?", engine="smart")
```

Two distinct complaints, only one of which is a real defect:

- **`with runtime:` *and* passing `runtime`** — NOT redundancy. `with` is **lifecycle** (activate
  engines, single-owner-thread, close every httpx client on exit); passing `runtime` is the
  **explicit dependency**. The redesign deliberately deleted `current_runtime()`/the ambient
  ContextVar so there is *no* hidden global (two runtimes at once, thread-safe, testable). Keep this.
- **`(runtime, …, engine="name")` — the real wart.** Two arguments express one idea ("where this
  runs"), the name is stringly-typed (no autocomplete; a typo fails at **execute** time, far from the
  config), and the `engine=` kwarg is copy-pasted onto all ~19 I/O ops + `Function.__call__` +
  `execute_many`.

## 2. Why it is a string today (and the precise fix)

The design forwards a *string* because handing out the **engine object** would break the ownership
invariant — `Runtime` owns each engine's key, client, and close order; a loose engine reference
muddies "who closes it, is it usable after the runtime exits."

But the caller never wanted the engine — they wanted a **selector**. A cheap, frozen **handle** that
holds `(runtime, name)` and only *forwards* keeps ownership exactly where it is and fixes all three
problems:

```python
@dataclass(frozen=True, slots=True)
class LanguageModel:
    _runtime: "Runtime"
    _name: str
    def execute(self, request: LanguageModelRequest) -> LanguageModelResponse:
        return self._runtime.execute(request, engine=self._name)

@dataclass(frozen=True, slots=True)
class EmbeddingModel:
    _runtime: "Runtime"
    _name: str
    def execute(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return self._runtime.execute(request, engine=self._name)
```

Acquired from the runtime, which validates **now** (fail fast, next to config):

```python
class Runtime:
    def language_model(self, name: str | None = None) -> LanguageModel: ...
    def embedding(self, name: str | None = None) -> EmbeddingModel: ...
    # name given -> that engine; name omitted -> the sole engine for that capability,
    # otherwise AmbiguousEngineError. Resolution happens eagerly at acquisition.
```

Usage becomes one explicit argument:

```python
with load_runtime(config) as rt:          # __enter__ already returns self
    fast  = rt.language_model("fast")     # UnknownEngineError raised HERE, not at execute
    smart = rt.language_model("smart")
    vec   = rt.embedding()                # sole configured embedding engine

    s = text.summarize(fast, doc)
    a = reason.query(smart, doc, "thesis?")
    e = embed.embed(vec, Symbol(["cat", "dog"]))
```

## 3. What changes in `ops.*` and `Function`

- I/O op signatures go from `(runtime: Runtime, source: Symbol[T], …, *, engine: str | None = None)`
  to `(model: LanguageModel, source: Symbol[T], …)` — **one** parameter, **no** `engine=` kwarg.
  `ops.primitives._execute_language` takes the handle and calls `model_.execute(...)` (via `Function`).
- `Function.__call__(runtime, *values, engine=…)` → `Function.__call__(model, *values)`;
  `execute_many(model, inputs)`; `Function.request(*values)` is unchanged (still no I/O).
- `ops.embed.embed` takes an `EmbeddingModel`.
- **Deterministic ops are untouched** — `embed.similarity/distance/mmd/kernel` and `text.template`
  take no runtime/engine today and still take none.
- Low-level `runtime.execute(request, engine="name")` **stays** for dynamic/config-driven fan-out
  (looping over names). The handle is the ergonomic default, not the only door.

## 4. Bonus: it also fixes a known ambiguity

`r1-05`/`r2-c` flagged that engine names are unique only **per capability** (the same name may live in
both the language and embedding maps, disambiguated only by request type). Capability-specific
accessors `language_model(name)` / `embedding(name)` resolve that at the call site — `rt.language_model("x")`
and `rt.embedding("x")` are unambiguous by construction, so this note quietly closes that gap too.

## 5. Tradeoffs (honest)

- **+** one argument; typos caught at acquisition, next to config; better autocomplete; no `engine=`
  proliferation; still fully explicit; Runtime still owns lifecycle (handle is a view, dead after
  `with` exits — same `RuntimeClosedError` as today); resolves the per-capability name ambiguity.
- **−** adds two small frozen handle types; marginally more indirection; a purely dynamic
  "loop over engine names" reads slightly better with the raw string (kept available via
  `runtime.execute(..., engine=name)`).
- The design doc's line "operations pass `engine=<name>` through without knowing provider details"
  needs a one-word update — ops now pass a *handle* through, which is strictly **more** explicit and
  fully in the spirit of "make every model call explicit."

## 6. Recommendation

Adopt the handle as the primary selection API; keep `runtime.execute(req, engine=name)` as the
low-level escape hatch. Slot it into the plan alongside the naming/placement work (Group D) —
`keeps-all`, effort **S–M** (touches ~19 op signatures + `Function` + their tests, mechanical). It is
a pure ergonomics win that also erases the per-capability name-ambiguity finding.
