from collections.abc import Iterator
from typing import Any, override


def _unwrap_operand(value: object) -> object:
    if isinstance(value, Symbol):
        return value.value
    return value


class Symbol[T]:
    __slots__ = ("_value",)
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __init__(self, value: T) -> None:
        object.__setattr__(self, "_value", value)

    @override
    def __setattr__(self, _name: str, _value: object) -> None:
        msg = f"{type(self).__name__} is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, _name: str) -> None:
        msg = f"{type(self).__name__} is immutable"
        raise AttributeError(msg)

    @property
    def value(self) -> T:
        return self._value

    @override
    def __eq__(self, other: object) -> bool:
        return bool(self._value == _unwrap_operand(other))

    @override
    def __ne__(self, other: object) -> bool:
        return bool(self._value != _unwrap_operand(other))

    def __lt__(self, other: object) -> bool:
        return bool(self._value < _unwrap_operand(other))

    def __le__(self, other: object) -> bool:
        return bool(self._value <= _unwrap_operand(other))

    def __gt__(self, other: object) -> bool:
        return bool(self._value > _unwrap_operand(other))

    def __ge__(self, other: object) -> bool:
        return bool(self._value >= _unwrap_operand(other))

    def __contains__(self, item: object) -> bool:
        return bool(_unwrap_operand(item) in self._value)

    def __add__(self, other: object) -> "Symbol[Any]":
        return Symbol(self._value + _unwrap_operand(other))

    def __radd__(self, other: object) -> "Symbol[Any]":
        return Symbol(_unwrap_operand(other) + self._value)

    def __sub__(self, other: object) -> "Symbol[Any]":
        return Symbol(self._value - _unwrap_operand(other))

    def __rsub__(self, other: object) -> "Symbol[Any]":
        return Symbol(_unwrap_operand(other) - self._value)

    def __mul__(self, other: object) -> "Symbol[Any]":
        return Symbol(self._value * _unwrap_operand(other))

    def __rmul__(self, other: object) -> "Symbol[Any]":
        return Symbol(_unwrap_operand(other) * self._value)

    def __matmul__(self, other: object) -> "Symbol[Any]":
        return Symbol(self._value @ _unwrap_operand(other))

    def __rmatmul__(self, other: object) -> "Symbol[Any]":
        return Symbol(_unwrap_operand(other) @ self._value)

    def __truediv__(self, other: object) -> "Symbol[Any]":
        return Symbol(self._value / _unwrap_operand(other))

    def __rtruediv__(self, other: object) -> "Symbol[Any]":
        return Symbol(_unwrap_operand(other) / self._value)

    def __floordiv__(self, other: object) -> "Symbol[Any]":
        return Symbol(self._value // _unwrap_operand(other))

    def __rfloordiv__(self, other: object) -> "Symbol[Any]":
        return Symbol(_unwrap_operand(other) // self._value)

    def __mod__(self, other: object) -> "Symbol[Any]":
        return Symbol(self._value % _unwrap_operand(other))

    def __rmod__(self, other: object) -> "Symbol[Any]":
        return Symbol(_unwrap_operand(other) % self._value)

    def __divmod__(self, other: object) -> "Symbol[Any]":
        return Symbol(divmod(self._value, _unwrap_operand(other)))

    def __rdivmod__(self, other: object) -> "Symbol[Any]":
        return Symbol(divmod(_unwrap_operand(other), self._value))

    def __pow__(
        self,
        other: object,
        modulo: object | None = None,
    ) -> "Symbol[Any]":
        exponent = _unwrap_operand(other)
        if modulo is None:
            return Symbol(pow(self._value, exponent))

        return Symbol(pow(self._value, exponent, _unwrap_operand(modulo)))

    def __rpow__(
        self,
        other: object,
        modulo: object | None = None,
    ) -> "Symbol[Any]":
        base = _unwrap_operand(other)
        if modulo is None:
            return Symbol(pow(base, self._value))

        return Symbol(pow(base, self._value, _unwrap_operand(modulo)))

    def __and__(self, other: object) -> "Symbol[Any]":
        return Symbol(self._value & _unwrap_operand(other))

    def __rand__(self, other: object) -> "Symbol[Any]":
        return Symbol(_unwrap_operand(other) & self._value)

    def __or__(self, other: object) -> "Symbol[Any]":
        return Symbol(self._value | _unwrap_operand(other))

    def __ror__(self, other: object) -> "Symbol[Any]":
        return Symbol(_unwrap_operand(other) | self._value)

    def __xor__(self, other: object) -> "Symbol[Any]":
        return Symbol(self._value ^ _unwrap_operand(other))

    def __rxor__(self, other: object) -> "Symbol[Any]":
        return Symbol(_unwrap_operand(other) ^ self._value)

    def __lshift__(self, other: object) -> "Symbol[Any]":
        return Symbol(self._value << _unwrap_operand(other))

    def __rlshift__(self, other: object) -> "Symbol[Any]":
        return Symbol(_unwrap_operand(other) << self._value)

    def __rshift__(self, other: object) -> "Symbol[Any]":
        return Symbol(self._value >> _unwrap_operand(other))

    def __rrshift__(self, other: object) -> "Symbol[Any]":
        return Symbol(_unwrap_operand(other) >> self._value)

    def __neg__(self) -> "Symbol[Any]":
        return Symbol(-self._value)

    def __pos__(self) -> "Symbol[Any]":
        return Symbol(+self._value)

    def __abs__(self) -> "Symbol[Any]":
        return Symbol(abs(self._value))

    def __invert__(self) -> "Symbol[Any]":
        return Symbol(~self._value)

    def __getitem__(self, key: object) -> "Symbol[Any]":
        return Symbol(self._value[_unwrap_operand(key)])

    def __iter__(self) -> Iterator["Symbol[Any]"]:
        return map(Symbol, iter(self._value))

    def __len__(self) -> int:
        return len(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)

    @override
    def __str__(self) -> str:
        return str(self._value)

    def __int__(self) -> int:
        return int(self._value)

    def __float__(self) -> float:
        return float(self._value)
