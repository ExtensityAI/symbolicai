from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType

from symai.runtime.models import TokenUsage
from symai.runtime.observability import ExecutionRecord

_ONE_MILLION = Decimal(1_000_000)
_ZERO = Decimal(0)


@dataclass(frozen=True, slots=True)
class Price:
    input: Decimal
    cached_input: Decimal
    output: Decimal

    def __post_init__(self) -> None:
        for field, value in (
            ("input", self.input),
            ("cached_input", self.cached_input),
            ("output", self.output),
        ):
            if not isinstance(value, Decimal):
                msg = f"{field} price must be a Decimal"
                raise TypeError(msg)
            if not value.is_finite() or value < 0:
                msg = f"{field} price must be finite and nonnegative"
                raise ValueError(msg)


def cost(usage: TokenUsage, price: Price) -> Decimal:
    fresh_prompt_tokens = usage.prompt_tokens - usage.cached_prompt_tokens
    if fresh_prompt_tokens < 0:
        msg = "cached prompt tokens must not exceed prompt tokens"
        raise ValueError(msg)
    return (
        Decimal(fresh_prompt_tokens) * price.input
        + Decimal(usage.cached_prompt_tokens) * price.cached_input
        + Decimal(usage.completion_tokens) * price.output
    ) / _ONE_MILLION


class UsageMeter:
    def __init__(self, prices: Mapping[tuple[str, str], Price]) -> None:
        self._prices = MappingProxyType(dict(prices))
        self.tokens: dict[str, int] = {}
        self.cost: dict[str, Decimal] = {}

    def __call__(self, record: ExecutionRecord) -> None:
        usage = record.usage
        if usage is None:
            return

        self.tokens[record.engine] = self.tokens.get(record.engine, 0) + usage.total_tokens
        model = record.response_model or record.requested_model
        if record.provider is None or model is None:
            return
        price = self._prices.get((record.provider, model))
        if price is None:
            return
        self.cost[record.engine] = self.cost.get(record.engine, _ZERO) + cost(usage, price)
