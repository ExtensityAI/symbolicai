# R6 · Observability, usage & cost tracking (design)

Forward-looking design note (additive infrastructure, not part of the simplification pass).
Motivated by a product ask — flexible usage + cost tracking and logging — and by the audit's
finding that the current execution path has **zero observability**. Aligns with the planned
stdlib-logging rework (Plane SYMBOLICAI-17). Feature impact: **keeps-all** (purely additive).

## 1. The gap

- Usage **is** already normalized onto every response: `response.metadata.usage` (`TokenUsage`) +
  `metadata.rate_limit`, `provider`, `requested_model`, `response_model`, `request_id`, `status_code`.
- But you only see it via `Function` / `runtime.execute`. The ergonomic `ops.*` layer **discards it** —
  `ops.primitives._execute_language` returns `Symbol(decode_output(response, decoder))` and drops the
  response. So through the recommended API you cannot see usage at all.
- There is **no** cost, **no** aggregation, **no** logging, and **no** hook to add them.

## 2. Principle — one seam at `Runtime.execute`, not in return values

Every call — `ops.*`, `Function`, raw `runtime.execute` — funnels through `Runtime.execute`. That is
the single place to observe. Putting observability there (rather than threading metadata back through
return values) means:

- it works **uniformly** regardless of which layer the caller used — ops don't have to change and keep
  returning `Symbol[T]`;
- it stays **provider-neutral** — the record is built from the already-normalized `ResponseMetadata` /
  `ErrorMetadata`, so provider clients/engines stay unaware (layering preserved);
- the Runtime is **single-owner-thread**, so an accumulating observer needs **no locks**.

## 3. The record

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    engine: str                         # the selected name, e.g. "smart"
    capability: str                     # "language_model" | "embedding"
    provider: str
    requested_model: str
    response_model: str | None
    usage: TokenUsage | None
    rate_limit: RateLimitMetadata | None
    request_id: str | None
    status_code: int | None             # None if transport failed before a response
    duration_s: float
    error: Exception | None             # None on success; the raised runtime error otherwise
```

Deliberately **no request messages** on the record (prompts can be sensitive; keeping them out of the
default observability path prevents accidental log leakage). If a caller wants prompt-size context,
add opt-in counts (message count / estimated tokens), never the raw text, gated behind an explicit flag.

## 4. The hook

```python
from collections.abc import Callable, Sequence
Observer = Callable[[ExecutionRecord], None]

class Runtime:
    def __init__(self, *, observers: Sequence[Observer] = (), ...): ...
```

Placement in `execute()` — the engine call already happens **outside** `_lifecycle_lock` (resolution is
under the lock, `selected.execute(request)` is not), so wrap exactly that call:

```python
        # (engine resolved under the lock as today)
    start = monotonic()
    try:
        response = selected.execute(request)
    except SymbolicAIRuntimeError as error:
        self._emit(_record_from_error(engine, capability, error, monotonic() - start))
        raise
    self._emit(_record_from_response(engine, capability, response, monotonic() - start))
    return response
```

Rules:
- fire on **both** success and error paths;
- observer exceptions are **swallowed and logged** (`logger.exception`) so telemetry can never break a
  real call;
- observers fire in registration order;
- use `time.monotonic()` for `duration_s` (wall-clock-independent).

`_record_from_response` reads `response.metadata`; `_record_from_error` reads `error.metadata`
(`ErrorMetadata`: provider, model, request_id, retry_after) — partial but honest when transport failed
before a response existed.

## 5. Logging (built-in observer)

The library currently logs nothing. Add a module logger and a ready-made logging observer; keep the
library **opt-in** (never configure handlers — the app owns that), following the stdlib-logging rules:

```python
import logging
logger = logging.getLogger("symai.runtime")

def log_executions(record: ExecutionRecord) -> None:
    if record.error is not None:
        logger.error(
            "engine=%s model=%s failed", record.engine, record.requested_model,
            extra={"engine": record.engine, "provider": record.provider,
                   "request_id": record.request_id, "status": record.status_code,
                   "duration_s": record.duration_s},
        )
        return

    logger.info(
        "engine=%s model=%s ok", record.engine, record.response_model or record.requested_model,
        extra={"engine": record.engine, "provider": record.provider,
               "request_id": record.request_id, "duration_s": record.duration_s,
               "prompt_tokens": getattr(record.usage, "prompt_tokens", None),
               "completion_tokens": getattr(record.usage, "completion_tokens", None)},
    )
```

Structured `extra=` keys make it drop-in for JSON log processors. This is the vehicle for
SYMBOLICAI-17: one logger + this observer, no logging scattered through the engines.

## 6. Cost (userland price table, never shipped)

Prices change constantly and are per (provider, model) — keep them out of the library:

```python
from decimal import Decimal

@dataclass(frozen=True, slots=True)
class Price:                       # per 1M tokens
    input: Decimal
    cached_input: Decimal
    output: Decimal

def cost(usage: TokenUsage, price: Price) -> Decimal:
    fresh = usage.prompt_tokens - usage.cached_prompt_tokens
    return (Decimal(fresh) * price.input
            + Decimal(usage.cached_prompt_tokens) * price.cached_input
            + Decimal(usage.completion_tokens) * price.output) / Decimal(1_000_000)

# App-owned:
PRICES: dict[tuple[str, str], Price] = {
    ("openai", "gpt-5.5"): Price(input=Decimal("1.25"), cached_input=Decimal("0.125"), output=Decimal("10")),
}
```

An accumulating meter is just an observer:

```python
class UsageMeter:
    def __init__(self, prices: dict[tuple[str, str], Price]) -> None:
        self._prices = prices
        self.tokens: dict[str, int] = {}
        self.cost: dict[str, Decimal] = {}

    def __call__(self, r: ExecutionRecord) -> None:
        if r.usage is None:
            return
        model = r.response_model or r.requested_model
        self.tokens[r.engine] = self.tokens.get(r.engine, 0) + r.usage.total_tokens
        price = self._prices.get((r.provider, model))
        if price is not None:
            self.cost[r.engine] = self.cost.get(r.engine, Decimal(0)) + cost(r.usage, price)
```

Usage:

```python
meter = UsageMeter(PRICES)
with load_runtime(config, observers=(log_executions, meter)) as rt:
    text.summarize(rt, doc, engine="smart")     # usage captured even though ops return a Symbol
print(meter.tokens, meter.cost)
```

## 7. Where the pieces live (layering)

- `ExecutionRecord`, `Observer`, the emit logic → `symai/runtime/` (runtime-level concern).
- `log_executions` + module logger → `symai/runtime/` (or a small `symai/observability.py`).
- `Price`, `cost`, `UsageMeter` → could ship as an optional `symai/pricing.py` helper, but the **price
  data** is always app-owned.
- Provider clients/engines stay untouched — records are built from normalized metadata only.

## 8. Tradeoffs & scope

- **+** one seam covers ops + Function + raw execute; no ops change; provider-neutral; lock-free
  accumulation; the logging vehicle SYMBOLICAI-17 wants; cost stays flexible/app-owned.
- **−** small new runtime surface (`observers=` + record type); a shared meter across threads/runtimes
  is the user's synchronization problem, not the library's; duration excludes decode time (it measures
  the provider call only — the honest boundary).
- **Out of scope:** streaming/async (none exists); per-token streaming cost; retry accounting (add a
  `retries: int` field later if a retry policy lands).
- **Interacts with `r5`:** if the engine-handle API lands, `RuntimeConfig` keys still name the engines,
  so `ExecutionRecord.engine` is unchanged; a handle could also expose a per-engine `meter` view.

## 9. Recommendation

Add the `observers=` seam + `ExecutionRecord` + the built-in `log_executions` observer to the Runtime
(effort **M**), ship `Price`/`cost`/`UsageMeter` as an optional helper, and keep price data app-owned.
This closes the observability gap, gives flexible usage+cost tracking, and is the home for the planned
logging rework — all additive, `keeps-all`.
