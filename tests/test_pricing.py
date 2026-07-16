from decimal import Decimal

import pytest

from symai.pricing import Price, UsageMeter, cost
from symai.runtime.models import TokenUsage
from symai.runtime.observability import ExecutionRecord


def record(
    *,
    engine: str,
    provider: str = "openai",
    requested_model: str = "requested",
    response_model: str | None = "served",
    usage: TokenUsage | None,
) -> ExecutionRecord:
    return ExecutionRecord(
        engine=engine,
        capability="language_model",
        provider=provider,
        requested_model=requested_model,
        response_model=response_model,
        usage=usage,
        rate_limit=None,
        request_id=None,
        status_code=200,
        duration_s=0.25,
        error=None,
    )


def test_cost_separates_fresh_cached_and_completion_tokens() -> None:
    usage = TokenUsage(
        prompt_tokens=100,
        cached_prompt_tokens=20,
        completion_tokens=10,
        total_tokens=110,
    )
    price = Price(
        input=Decimal("2"),
        cached_input=Decimal("0.5"),
        output=Decimal("10"),
    )

    assert cost(usage, price) == Decimal("0.00027")


@pytest.mark.parametrize(
    "price",
    [
        Price(input=Decimal("0"), cached_input=Decimal("0"), output=Decimal("0")),
        Price(input=Decimal("1.5"), cached_input=Decimal("0.1"), output=Decimal("8")),
    ],
)
def test_price_accepts_finite_nonnegative_decimals(price: Price) -> None:
    assert price.input >= 0
    assert price.cached_input >= 0
    assert price.output >= 0


@pytest.mark.parametrize("value", [Decimal("-0.1"), Decimal("NaN"), Decimal("Infinity")])
def test_price_rejects_invalid_values(value: Decimal) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        Price(input=value, cached_input=Decimal("0"), output=Decimal("0"))


def test_cost_rejects_cached_tokens_exceeding_prompt_tokens() -> None:
    usage = TokenUsage(prompt_tokens=1, cached_prompt_tokens=2, total_tokens=1)
    price = Price(input=Decimal("1"), cached_input=Decimal("1"), output=Decimal("1"))

    with pytest.raises(ValueError, match="cached prompt tokens"):
        cost(usage, price)


def test_usage_meter_aggregates_tokens_and_app_owned_prices_by_engine() -> None:
    prices = {
        ("openai", "served"): Price(
            input=Decimal("2"),
            cached_input=Decimal("1"),
            output=Decimal("4"),
        )
    }
    meter = UsageMeter(prices)
    prices.clear()
    first = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    second = TokenUsage(prompt_tokens=4, completion_tokens=1, total_tokens=5)

    meter(record(engine="smart", usage=first))
    meter(record(engine="smart", usage=second))
    meter(record(engine="unpriced", response_model="other", usage=first))
    meter(record(engine="smart", usage=None))

    assert meter.tokens == {"smart": 20, "unpriced": 15}
    assert meter.cost == {"smart": Decimal("0.000052")}
