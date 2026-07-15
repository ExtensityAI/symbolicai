import ast
import numbers
import pickle
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, overload

import numpy as np
from pydantic import BaseModel

from symai.decoding import (
    ConstructorDecoder,
    Decoder,
    PydanticDecoder,
    TextDecoder,
    decode_output,
)

from symai.operations import (
    combine_request,
    compare_request,
    contains_request,
    contextualize_language_request,
    convert_request,
    embedding_request,
    endswith_request,
    equals_request,
    extract_request,
    filter_request,
    getitem_request,
    include_request,
    interpret_request,
    invert_request,
    isinstanceof_request,
    logic_request,
    map_request,
    modify_request,
    negate_request,
    parse_embedding_response,
    query_request,
    rank_request,
    replace_request,
    setitem_request,
    startswith_request,
    style_request,
    summarize_request,
    translate_request,
)
from symai.prompts import Prompt
from symai.runtime.models import LanguageModelRequest, ResponseMetadata
from symai.runtime.runtime import current_runtime

_NATIVE_UNSUPPORTED = object()

if TYPE_CHECKING:
    from symai.symbol import Symbol


class Primitive(Protocol):
    """Fixed provider-neutral operations composed into every Symbol."""

    _embedding: np.ndarray | None
    _semantic: bool
    _value: Any

    @property
    def value(self) -> Any: ...

    @property
    def static_context(self) -> str: ...

    @property
    def dynamic_context(self) -> str: ...

    @property
    def _symbol_type(self) -> type["Symbol"]: ...

    def _to_type(
        self,
        value: Any,
        *,
        semantic: bool | None = None,
    ) -> "Symbol": ...


def _decoder_for(return_type: type) -> Decoder[object]:
    if return_type is str:
        return TextDecoder()
    if issubclass(return_type, BaseModel):
        return PydanticDecoder(return_type)
    return ConstructorDecoder(return_type)


def _execute_language_value(
    symbol: Primitive,
    request: LanguageModelRequest,
    *,
    return_type: type = str,
    output_index: int = 0,
    default: object = None,
    limit: int | None = 1,
    literal: bool = False,
) -> object:
    request = contextualize_language_request(
        request,
        static_context=symbol.static_context,
        dynamic_context=symbol.dynamic_context,
    )
    response = current_runtime().execute(request)
    decoder = _decoder_for(type(symbol.value) if literal else return_type)
    return decode_output(
        response,
        decoder,
        output_index=output_index,
        default=default,
        limit=limit,
    )


def _execute_symbol(
    symbol: Primitive,
    request: LanguageModelRequest,
    kwargs: dict[str, object],
    *,
    return_type: type = str,
    default: object = None,
    limit: int | None = 1,
    literal: bool = False,
) -> Any:
    options = kwargs.copy()
    output_index = options.pop("output_index", 0)
    return_metadata = options.pop("return_metadata", False)
    requested_return_type = options.pop("return_type", return_type)
    default = options.pop("default", default)
    requested_limit = options.pop("limit", limit)
    forbidden = {"engine", "model", "provider"}.intersection(options)
    if forbidden:
        names = ", ".join(sorted(forbidden))
        msg = f"Provider/model selection belongs to runtime configuration, not per-call kwargs: {names}"
        raise TypeError(msg)
    if options:
        names = ", ".join(sorted(options))
        msg = f"Unsupported execution options: {names}"
        raise TypeError(msg)
    if not isinstance(output_index, int):
        msg = "output_index must be an integer"
        raise TypeError(msg)
    if not isinstance(return_metadata, bool):
        msg = "return_metadata must be a boolean"
        raise TypeError(msg)
    if not isinstance(requested_return_type, type):
        msg = "return_type must be a type"
        raise TypeError(msg)
    return_type = requested_return_type
    if requested_limit is not None and not isinstance(requested_limit, int):
        msg = "limit must be an integer or None"
        raise TypeError(msg)
    limit = requested_limit

    request = contextualize_language_request(
        request,
        static_context=symbol.static_context,
        dynamic_context=symbol.dynamic_context,
    )
    response = current_runtime().execute(request)
    decoder = _decoder_for(type(symbol.value) if literal else return_type)
    value = decode_output(
        response,
        decoder,
        output_index=output_index,
        default=default,
        limit=limit,
    )
    result = symbol._to_type(value)
    if return_metadata:
        return result, response.metadata
    return result


class OperatorPrimitives(Primitive):
    __hash__ = object.__hash__

    def __try_type_specific_func(
        self,
        other: Any,
        function: Callable[[Any, Any], Any],
        op: str | None = None,
    ) -> Any:
        if not isinstance(other, self._symbol_type):
            other = self._to_type(other)

        semantic_fallback = self._semantic or getattr(other, "_semantic", False)
        if self.value is None or other.value is None:
            if semantic_fallback:
                return _NATIVE_UNSUPPORTED
            msg = (
                f"unsupported operand type(s) for {op}: "
                f"'{type(self.value)}' and '{type(other.value)}'"
            )
            raise TypeError(msg)

        try:
            value = function(self, other)
        except TypeError as error:
            traceback = error.__traceback__
            # Python converts bilateral NotImplemented into TypeError in the
            # operation wrapper; a deeper frame means the operand code failed.
            while traceback is not None and traceback.tb_next is not None:
                traceback = traceback.tb_next
            function_code = getattr(function, "__code__", None)
            if traceback is None or traceback.tb_frame.f_code is not function_code:
                raise
            if semantic_fallback:
                return _NATIVE_UNSUPPORTED
            raise

        if value is NotImplemented:
            if semantic_fallback:
                return _NATIVE_UNSUPPORTED
            operation = "" if op is None else op
            msg = (
                f"unsupported operand type(s) for {operation}: "
                f"'{type(self.value)}' and '{type(other.value)}'"
            )
            raise TypeError(msg)
        return value

    def __bool__(self) -> bool:
        """
        Get the boolean value of the Symbol.
        If the Symbol's value is of type 'bool', the method returns the boolean value, otherwise it returns False.

        Returns:
            bool: The boolean value of the Symbol.
        """
        val = False
        if isinstance(self.value, bool):
            val = self.value
        elif self.value is not None:
            val = bool(self.value)

        return val

    def __contains__(self, other: Any) -> bool:
        """
        Check if a Symbol object is present in another Symbol object.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to check for containment.

        Returns:
            bool: True if the current Symbol contains the 'other' Symbol, otherwise False.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: other.value in self.value, op="in"
        )

        if result is not _NATIVE_UNSUPPORTED:
            return bool(result)

        value = _execute_language_value(
            self,
            contains_request(self, other),
            return_type=bool,
            default=False,
        )
        return bool(value)

    def __eq__(self, other: Any) -> bool:
        """
        Check if the current Symbol is equal to another Symbol.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to check for equality.

        Returns:
            bool: True if the current Symbol is equal to the 'other' Symbol, otherwise False.
        """
        if self is other:
            return True

        result = self.__try_type_specific_func(
            other, lambda self, other: self.value == other.value, op="=="
        )

        if result is not _NATIVE_UNSUPPORTED:
            return bool(result)

        value = _execute_language_value(
            self,
            equals_request(self, other),
            return_type=bool,
            default=False,
        )
        return bool(value)

    def __ne__(self, other: Any) -> bool:
        """
        This method checks if a Symbol object is not equal to another Symbol by using the __eq__ method.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to check for inequality.

        Returns:
            bool: True if the current Symbol is not equal to the 'other' Symbol, otherwise False.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value != other.value, op="!="
        )

        if result is not _NATIVE_UNSUPPORTED:
            return bool(result)

        return not self.__eq__(other)

    def __gt__(self, other: Any) -> bool:
        """
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to compare.

        Returns:
            bool: True if the current Symbol is greater than the 'other' Symbol, otherwise False.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value > other.value, op=">"
        )

        if result is not _NATIVE_UNSUPPORTED:
            return bool(result)

        value = _execute_language_value(
            self,
            compare_request(self, ">", other),
            return_type=bool,
            default=False,
        )
        return bool(value)

    def __lt__(self, other: Any) -> bool:
        """
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to compare.

        Returns:
            bool: True if the current Symbol is less than the 'other' Symbol, otherwise False.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value < other.value, op="<"
        )

        if result is not _NATIVE_UNSUPPORTED:
            return bool(result)

        value = _execute_language_value(
            self,
            compare_request(self, "<", other),
            return_type=bool,
            default=False,
        )
        return bool(value)

    def __le__(self, other: Any) -> bool:
        """
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to compare.

        Returns:
            bool: True if the current Symbol is less than or equal to the 'other' Symbol, otherwise False.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value <= other.value, op="<="
        )

        if result is not _NATIVE_UNSUPPORTED:
            return bool(result)

        value = _execute_language_value(
            self,
            compare_request(self, "<=", other),
            return_type=bool,
            default=False,
        )
        return bool(value)

    def __ge__(self, other: Any) -> bool:
        """
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to compare.

        Returns:
            bool: True if the current Symbol is greater than or equal to the 'other' Symbol, otherwise False.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value >= other.value, op=">="
        )

        if result is not _NATIVE_UNSUPPORTED:
            return bool(result)

        value = _execute_language_value(
            self,
            compare_request(self, ">=", other),
            return_type=bool,
            default=False,
        )
        return bool(value)

    def __neg__(self) -> "Symbol":
        """
        Return the negated value of the Symbol.

        Returns:
            Symbol: The negated value of the Symbol.
        """
        result = self.__try_type_specific_func(False, lambda self, _: -self.value, op="-")

        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        value = _execute_language_value(self, negate_request(self))
        return self._to_type(value)

    def __invert__(self) -> "Symbol":
        """
        Return the inverted value of the Symbol (logical NOT).
        This allows using the ~ operator for semantic inversion.

        Returns:
            Symbol: The negated value of the Symbol.
        """
        if isinstance(self.value, bool):
            return self._to_type(not self.value)

        result = self.__try_type_specific_func(False, lambda self, _: ~self.value, op="~")

        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        value = _execute_language_value(self, invert_request(self))
        return self._to_type(value)

    def __lshift__(self, other: Any) -> "Symbol":
        """
        Add new information to the Symbol.

        Args:
            information (Any): The information to include in the Symbol.

        Returns:
            Symbol: The Symbol with the new information included.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value << other.value, op="<<"
        )

        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        value = _execute_language_value(self, include_request(self, other))
        return self._to_type(value)

    def __rlshift__(self, other: Any) -> "Symbol":
        """
        Add new information to the Symbol.

        Args:
            information (Any): The information to include in the Symbol.

        Returns:
            Symbol: The Symbol with the new information included.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: other.value << self.value, op="<<"
        )

        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        value = _execute_language_value(self, include_request(self, other))
        return self._to_type(value)

    def __ilshift__(self, other: Any) -> Self:
        """
        Add new information to the Symbol.

        Args:
            information (Any): The information to include in the Symbol.

        Returns:
            Symbol: The Symbol with the new information included.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value << other.value, op="<<="
        )

        if result is not _NATIVE_UNSUPPORTED:
            self._value = result
            return self

        self._value = _execute_language_value(self, include_request(self, other))
        return self

    def __rshift__(self, other: Any) -> "Symbol":
        """
        Add new information to the Symbol.

        Args:
            information (Any): The information to include in the Symbol.

        Returns:
            Symbol: The Symbol with the new information included.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value >> other.value, op=">>"
        )

        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        value = _execute_language_value(self, include_request(self, other))
        return self._to_type(value)

    def __rrshift__(self, other: Any) -> "Symbol":
        """
        Add new information to the Symbol.

        Args:
            information (Any): The information to include in the Symbol.

        Returns:
            Symbol: The Symbol with the new information included.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: other.value >> self.value, op=">>"
        )

        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        value = _execute_language_value(self, include_request(self, other))
        return self._to_type(value)

    def __irshift__(self, other: Any) -> Self:
        """
        Add new information to the Symbol.

        Args:
            information (Any): The information to include in the Symbol.

        Returns:
            Symbol: The Symbol with the new information included.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value >> other.value, op=">>="
        )

        if result is not _NATIVE_UNSUPPORTED:
            self._value = result
            return self

        self._value = _execute_language_value(self, include_request(self, other))
        return self

    def __add__(self, other: Any) -> "Symbol":
        """
        Combine the Symbol with another value.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other: The value to combine with the Symbol.

        Returns:
            Symbol: The Symbol combined with the other value.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value + other.value, op="+"
        )

        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        value = _execute_language_value(self, combine_request(self, other))
        return self._to_type(value)

    def __radd__(self, other) -> "Symbol":
        """
        Combine another value with the Symbol.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The value to combine with the Symbol.

        Returns:
            Symbol: The other value combined with the Symbol.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: other.value + self.value, op="+"
        )

        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        value = _execute_language_value(self, combine_request(other, self))
        return self._to_type(value)

    def __iadd__(self, other: Any) -> Self:
        """
        This method adds another value to the Symbol and updates its value with the result.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The value to add to the Symbol.

        Returns:
            Symbol: The updated Symbol with the added value.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value + other.value, op="+="
        )

        if result is not _NATIVE_UNSUPPORTED:
            self._value = result
            return self

        value = self.__add__(other)
        self._value = value.value
        return self

    def __sub__(self, other: Any) -> "Symbol":
        """
        Replace occurrences of a value with another value in the Symbol.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The value to replace in the Symbol.

        Returns:
            Symbol: The Symbol with occurrences of the other value replaced with an empty string.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value - other.value, op="-"
        )

        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        value = _execute_language_value(self, replace_request(self, other, ""))
        return self._to_type(value)

    def __rsub__(self, other: Any) -> "Symbol":
        """
        Subtracts the symbol value from another one and removes the substrings that match the symbol value.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The string to subtract the symbol value from.

        Returns:
            Symbol: A new symbol with the result of the subtraction.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: other.value - self.value, op="-"
        )

        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        other = self._to_type(other)
        value = _execute_language_value(self, replace_request(other, self, ""))
        return self._to_type(value)

    def __isub__(self, other: Any) -> Self:
        """
        In-place subtraction of the symbol value by the other symbol value.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The symbol to subtract from the current symbol.

        Returns:
            Symbol: The current symbol with the updated value.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value - other.value, op="-="
        )

        if result is not _NATIVE_UNSUPPORTED:
            self._value = result
            return self

        value = self.__sub__(other)
        self._value = value.value
        return self

    def __and__(self, other: Any) -> Any:
        """
        Performs a logical AND operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The string to perform the AND operation with the symbol value.

        Returns:
            Symbol: A new symbol with the result of the AND operation.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value & other.value, op="&"
        )

        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        value = _execute_language_value(self, logic_request(self, "and", other))
        return self._to_type(value)

    def __rand__(self, other: Any) -> Any:
        """
        Performs a logical AND operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The string to perform the AND operation with the symbol value.

        Returns:
            Symbol: A new symbol with the result of the AND operation.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: other.value & self.value, op="&"
        )

        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        other = self._to_type(other)
        value = _execute_language_value(self, logic_request(other, "and", self))
        return self._to_type(value)

    def __iand__(self, other: Any) -> Any:
        """
        Performs a logical AND operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The string to perform the AND operation with the symbol value.

        Returns:
            Symbol: A new symbol with the result of the AND operation.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value & other.value, op="&="
        )

        if result is not _NATIVE_UNSUPPORTED:
            self._value = result
            return self

        self._value = _execute_language_value(self, logic_request(self, "and", other))
        return self

    def __or__(self, other: Any) -> Any:
        """
        Performs a logical OR operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The string to perform the OR operation with the symbol value.

        Returns:
            Symbol: A new symbol with the result of the OR operation.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value | other.value, op="|"
        )

        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        value = _execute_language_value(self, logic_request(self, "or", other))
        return self._to_type(value)

    def __ror__(self, other: Any) -> "Symbol":
        """
        Performs a logical OR operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to concatenate.

        Returns:
            Symbol: A new Symbol object with the concatenated value.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value | other.value, op="|"
        )

        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        other = self._to_type(other)
        value = _execute_language_value(self, logic_request(other, "or", self))
        return self._to_type(value)

    def __ior__(self, other: Any) -> Self:
        """
        Performs a logical OR operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to concatenate.

        Returns:
            Symbol: A new Symbol object with the concatenated value.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value | other.value, op="|="
        )

        if result is not _NATIVE_UNSUPPORTED:
            self._value = result
            return self

        self._value = _execute_language_value(self, logic_request(self, "or", other))
        return self

    def __xor__(self, other: Any) -> Any:
        """
        Performs a logical XOR operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The string to perform the XOR operation with the symbol value.

        Returns:
            Symbol: A new symbol with the result of the XOR operation.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value ^ other.value, op="^"
        )

        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        value = _execute_language_value(self, logic_request(self, "xor", other))
        return self._to_type(value)

    def __rxor__(self, other: Any) -> "Symbol":
        """
        Performs a logical XOR operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The string to perform the XOR operation with the symbol value.

        Returns:
            Symbol: A new symbol with the result of the XOR operation.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: other.value ^ self.value, op="^"
        )

        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        value = _execute_language_value(self, logic_request(other, "xor", self))
        return self._to_type(value)

    def __ixor__(self, other: Any) -> Self:
        """
        Performs a logical XOR operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The string to perform the XOR operation with the symbol value.

        Returns:
            Symbol: A new symbol with the result of the XOR operation.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value ^ other.value, op="^="
        )

        if result is not _NATIVE_UNSUPPORTED:
            self._value = result
            return self

        self._value = _execute_language_value(self, logic_request(self, "xor", other))
        return self

    def __matmul__(self, other: Any) -> "Symbol":
        """
        This method concatenates the string representation of two Symbol objects and returns a new Symbol with the concatenated result.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to concatenate.

        Returns:
            Symbol: A new Symbol object with the concatenated value.
        """
        if (isinstance(self.value, str) and isinstance(other, str)) or (
            isinstance(self.value, str)
            and isinstance(other, self._symbol_type)
            and isinstance(other.value, str)
        ):
            other = self._to_type(other)
            return self._to_type(f"{self.value}{other.value}")
        msg = f"This method is only supported for string concatenation! Got {type(self.value)} and {type(other)} instead."
        raise TypeError(msg)

    def __rmatmul__(self, other: Any) -> "Symbol":
        """
        This method concatenates the string representation of two Symbol objects in a reversed order and returns a new Symbol with the concatenated result.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to concatenate.

        Returns:
            Symbol: A new Symbol object with the concatenated value.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self._to_type(self.value).__matmul__(other), op="@"
        )

        return self._to_type(result)

    def __imatmul__(self, other: Any) -> Self:
        """
        This method concatenates the string representation of two Symbol objects and assigns the concatenated result to the value of the current Symbol object.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to concatenate.

        Returns:
            Symbol: The current Symbol object with the concatenated value.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self._to_type(self.value).__matmul__(other), op="@="
        )
        self._value = result

        return self

    def __truediv__(self, other: Any) -> "Symbol":
        """
        Divides the symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value / other.value, op="/"
        )
        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        return self._to_type(str(self).split(str(other)))

    def __rtruediv__(self, other: Any) -> "Symbol":
        """
        Divides the symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: other.value / self.value, op="/"
        )
        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        msg = "Division operation not supported semantically! Might change in the future."
        raise NotImplementedError(msg)

    def __itruediv__(self, other: Any) -> Self:
        """
        Divides the symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value / other.value, op="/="
        )
        if result is not _NATIVE_UNSUPPORTED:
            self._value = result
            return self

        msg = "Division operation is unsupported"
        raise NotImplementedError(msg)

    def __floordiv__(self, other: Any) -> "Symbol":
        """
        Floor divides the symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value // other.value, op="//"
        )
        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        msg = "Floor division operation not supported semantically! Might change in the future."
        raise NotImplementedError(msg)

    def __rfloordiv__(self, other: Any) -> "Symbol":
        """
        Floor divides the symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: other.value // self.value, op="//"
        )
        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        msg = "Floor division operation not supported semantically! Might change in the future."
        raise NotImplementedError(msg)

    def __ifloordiv__(self, other: Any) -> Self:
        """
        Floor divides the symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value // other.value, op="//="
        )
        if result is not _NATIVE_UNSUPPORTED:
            self._value = result
            return self

        msg = "Floor division operation is unsupported"
        raise NotImplementedError(msg)

    def __pow__(self, other: Any) -> "Symbol":
        """
        Power operation on symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value**other.value, op="**"
        )
        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        msg = "Power operation not supported semantically! Might change in the future."
        raise NotImplementedError(msg)

    def __rpow__(self, other: Any) -> "Symbol":
        """
        Power operation on symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: other.value**self.value, op="**"
        )
        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        msg = "Power operation not supported semantically! Might change in the future."
        raise NotImplementedError(msg)

    def __ipow__(self, other: Any) -> Self:
        """
        Power operation on symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value**other.value, op="**="
        )
        if result is not _NATIVE_UNSUPPORTED:
            self._value = result
            return self

        msg = "Power operation is unsupported"
        raise NotImplementedError(msg)

    def __mod__(self, other: Any) -> "Symbol":
        """
        Modulo operation on symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value % other.value, op="%"
        )
        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        msg = "Modulo operation not supported semantically! Might change in the future."
        raise NotImplementedError(msg)

    def __rmod__(self, other: Any) -> "Symbol":
        """
        Modulo operation on symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: other.value % self.value, op="%"
        )
        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        msg = "Modulo operation is unsupported"
        raise NotImplementedError(msg)

    def __imod__(self, other: Any) -> Self:
        """
        Modulo operation on symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value % other.value, op="%="
        )
        if result is not _NATIVE_UNSUPPORTED:
            self._value = result
            return self

        msg = "Modulo operation is unsupported"
        raise NotImplementedError(msg)

    def __mul__(self, other: Any) -> "Symbol":
        """
        Multiply operation on symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value * other.value, op="*"
        )
        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        msg = "Multiply operation not supported semantically! Might change in the future."
        raise NotImplementedError(msg)

    def __rmul__(self, other: Any) -> "Symbol":
        """
        Multiply operation on symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: other.value * self.value, op="*"
        )
        if result is not _NATIVE_UNSUPPORTED:
            return self._to_type(result)

        msg = "Multiply operation not supported semantically! Might change in the future."
        raise NotImplementedError(msg)

    def __imul__(self, other: Any) -> Self:
        """
        Multiply operation on symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        """
        result = self.__try_type_specific_func(
            other, lambda self, other: self.value * other.value, op="*="
        )
        if result is not _NATIVE_UNSUPPORTED:
            self._value = result
            return self

        msg = "Multiply operation is unsupported"
        raise NotImplementedError(msg)


class CastingPrimitives(Primitive):
    """
    This mixin contains functionalities related to casting symbols.
    """

    @property
    def syn(self) -> "Self | Symbol":
        """Return a native-only view that never invokes an active runtime fallback."""

        if not self._semantic:
            return self
        return self._to_type(self.value, semantic=False)

    @property
    def sem(self) -> "Self | Symbol":
        """Return a view that uses the active runtime when a native operation is unsupported."""

        if self._semantic:
            return self
        return self._to_type(self.value, semantic=True)

    def cast(self, as_type: type) -> Any:
        """
        Cast the Symbol's value to a specific type.

        Args:
            as_type (Type): The type to cast the Symbol's value to.

        Returns:
            The Symbol's value casted to the specified type.
        """
        return as_type(self.value)

    def to(self, as_type: type) -> Any:
        """
        Cast the Symbol's value to a specific type.

        Args:
            as_type (Type): The type to cast the Symbol's value to.

        Returns:
            The Symbol's value casted to the specified type.
        """
        return self.cast(as_type)

    def ast(self) -> Any:
        """
        Converts the string representation of the Symbol's value to an abstract syntax tree using 'ast.literal_eval'.

        Returns:
            The abstract syntax tree representation of the Symbol's value.
        """
        return ast.literal_eval(str(self.value))

    def str(self) -> str:
        """
        Get the string representation of the Symbol's value.

        Returns:
            str: The string representation of the Symbol's value.
        """
        return str(self.value)

    def int(self) -> int:
        """
        Get the integer representation of the Symbol's value.

        Returns:
            int: The integer representation of the Symbol's value.
        """
        return int(self.value)

    def float(self) -> float:
        """
        Get the float representation of the Symbol's value.

        Returns:
            float: The float representation of the Symbol's value.
        """
        return float(self.value)

    def bool(self) -> bool:
        """
        Get the boolean representation of the Symbol's value.

        Returns:
            bool: The boolean representation of the Symbol's value.
        """
        return bool(self.value)


# @TODO: We can do much better than asking the model to generate the entire thing. We can come up with a more structured way, e.g.,
#       using a JSON schema (or contracts) and return only the keys where we need to set/del information.
class IterationPrimitives(Primitive):
    """Fixed local collection operations with explicit runtime-backed methods."""

    def __getitem__(self, key: str | int | slice) -> "Symbol":
        """
        Get the item of the Symbol value with the specified key or index.
        If the Symbol value is a list, tuple, or numpy array, the key can be an integer or slice.
        If the Symbol value is a dictionary, the key can be a string or an integer.

        Args:
            key (Union[str, int, slice]): The key or index for the item in the Symbol value.

        Returns:
            Symbol: The item of the Symbol value with the specified key or index.

        Raises:
            KeyError: If the key or index is not found in the Symbol value.
        """
        if not self._semantic:
            try:
                return self.value[key]
            except Exception:
                msg = f"Key {key} not found in {self.value}"
                raise Exception(msg) from None

        value = _execute_language_value(self, getitem_request(self, key))
        return self._to_type(value)

    def __setitem__(
        self,
        key: str | int | slice,
        value: Any,
        *,
        output_index: int = 0,
    ) -> None:
        """
        Set the item of the Symbol value with the specified key or index to the given value.
        If the Symbol value is a list, the key can be an integer or slice.
        If the Symbol value is a dictionary, the key can be a string or an integer.

        Args:
            key (Union[str, int, slice]): The key or index for the item in the Symbol value.
            value: The value to set the item to.

        Raises:
            KeyError: If the key or index is not found in the Symbol value.
        """

        if not isinstance(self.value, (str, dict, list)):
            msg = f"Setting item is not supported for {type(self.value)}. Supported types are str, dict, and list."
            raise TypeError(msg)

        if not self._semantic:
            try:
                self._value[key] = value
                return
            except Exception:
                msg = f"Key {key} not found in {self.value}"
                raise Exception(msg) from None

        self._value = _execute_language_value(
            self,
            setitem_request(self, key, value),
            output_index=output_index,
            limit=None,
            literal=True,
        )

    def __delitem__(self, key: str | int, *, output_index: int = 0) -> None:
        """
        Delete the item of the Symbol value with the specified key or index.
        If the Symbol value is a dictionary, the key can be a string or an integer.

        Args:
            key (Union[str, int]): The key for the item in the Symbol value.

        Raises:
            KeyError: If the key or index is not found in the Symbol value.
        """

        if not isinstance(self.value, (str, dict, list)):
            msg = f"Setting item is not supported for {type(self.value)}. Supported types are str, dict, and list."
            raise TypeError(msg)

        if not self._semantic:
            try:
                del self._value[key]
                return
            except Exception:
                msg = f"Key {key} not found in {self.value}"
                raise Exception(msg) from None

        self._value = _execute_language_value(
            self,
            setitem_request(self, key, None, delete=True),
            output_index=output_index,
            limit=None,
            literal=True,
        )


# @TODO: Add tests for this class
class ValueHandlingPrimitives(Primitive):
    """Local value inspection plus explicit runtime-backed transformations."""

    @property
    def size(self) -> int:
        """
        Get the size of the container of the Symbol's value.

        Returns:
            int: The size of the container of the Symbol's value.
        """
        return len(self.value)

    @property
    def type(self):
        """
        Get the type of the Symbol.

        Returns:
            type: The type of the Symbol.
        """
        return type(self)

    @property
    def value_type(self):
        """
        Get the type of the Symbol's value.

        Returns:
            type: The type of the Symbol's value.
        """
        return type(self.value)

    def index(self, item: str, **kwargs) -> "Symbol":
        """
        Returns the index of a specified item in the symbol value.

        Args:
            item (str): The item to find the index of within the symbol value.

        Returns:
            Symbol: A new symbol with the index of the specified item.
        """

        return _execute_symbol(
            self,
            getitem_request(self, item),
            kwargs,
            return_type=int,
        )


class StringHelperPrimitives(Primitive):
    """
    This mixin contains functions that provide additional help for symbols or their values.
    """

    def split(self, delimiter: str) -> "Symbol":
        """
        Splits the symbol value by a specified delimiter.

        Args:
            delimiter (str): The delimiter to split the symbol value by.

        Returns:
            Symbol: A new symbol with the split value.
        """
        if not isinstance(delimiter, str):
            msg = f"delimiter must be a string, got {type(delimiter)}"
            raise TypeError(msg)
        if not isinstance(self.value, str):
            msg = f"value must be a string, got {type(self.value)}"
            raise TypeError(msg)
        return self._to_type([*self.value.split(delimiter)])

    def join(self, delimiter: str = " ") -> "Symbol":
        """
        Joins the symbol value with a specified delimiter.

        Args:
            delimiter (str, optional): The delimiter to join the symbol value with. Defaults to ' '.

        Returns:
            Symbol: A new symbol with the joined str value.
        """
        if not isinstance(delimiter, str):
            msg = f"delimiter must be a string, got {type(delimiter)}"
            raise TypeError(msg)
        if not isinstance(self.value, Iterable):
            msg = f"value must be an iterable, got {type(self.value)}"
            raise TypeError(msg)
        return self._to_type(delimiter.join(self.value))

    def startswith(self, prefix: str) -> bool:
        """
        Checks if the symbol value starts with a specified prefix.

        Args:
            prefix (str): The prefix to check if the symbol value starts with.

        Returns:
            bool: True if the symbol value starts with the specified prefix, otherwise False.
        """
        if not isinstance(prefix, str):
            msg = f"prefix must be a string, got {type(prefix)}"
            raise TypeError(msg)
        if not isinstance(self.value, str):
            msg = f"value must be a string, got {type(self.value)}"
            raise TypeError(msg)

        if not self._semantic:
            return self.value.startswith(prefix)

        value = _execute_language_value(
            self,
            startswith_request(self, prefix),
            return_type=bool,
            default=False,
        )
        return bool(value)

    def endswith(self, suffix: str) -> bool:
        """
        Checks if the symbol value ends with a specified suffix.

        Args:
            suffix (str): The suffix to check if the symbol value ends with.

        Returns:
            bool: True if the symbol value ends with the specified suffix, otherwise False.
        """
        if not isinstance(suffix, str):
            msg = f"suffix must be a string, got {type(suffix)}"
            raise TypeError(msg)
        if not isinstance(self.value, str):
            msg = f"value must be a string, got {type(self.value)}"
            raise TypeError(msg)

        if not self._semantic:
            return self.value.endswith(suffix)

        value = _execute_language_value(
            self,
            endswith_request(self, suffix),
            return_type=bool,
            default=False,
        )
        return bool(value)


class ComparisonPrimitives(Primitive):
    """
    This mixin is dedicated to functions that perform more complex comparison operations between symbols or symbol values.
    This usually involves additional context, which the builtin overrode (e.g. __eq__) functions lack.
    """

    def equals(self, string: str, context: str = "contextually", **kwargs) -> "Symbol":
        """
        Checks if the symbol value is equal to another string.

        Args:
            string (str): The string to compare with the symbol value.
            context (str, optional): The context in which to compare the strings. Defaults to 'contextually'.

        Returns:
            Symbol: A new symbol indicating whether the two strings are equal or not.
        """

        return _execute_symbol(
            self,
            equals_request(self, string, context=context),
            kwargs,
            return_type=bool,
            default=False,
        )

    def contains(self, element: Any, **kwargs) -> Any:
        """

        Args:
            element (Any): The element to be checked for containment.

        Returns:
            bool: True if the symbol's value contains the element, False otherwise.
        """

        value = _execute_symbol(
            self,
            contains_request(self, element),
            kwargs,
            return_type=bool,
            default=False,
        )
        if isinstance(value, tuple):
            result, metadata = value
            return bool(result.value), metadata
        return bool(value.value)

    def isinstanceof(self, query: str, **kwargs) -> Any:
        """
        Check if the current Symbol is an instance of a specific type.

        Args:
            query (str): The type to check if the Symbol is an instance of.

        Returns:
            bool: True if the current Symbol is an instance of the specified type, otherwise False.
        """

        value = _execute_symbol(
            self,
            isinstanceof_request(self, query),
            kwargs,
            return_type=bool,
            default=False,
        )
        if isinstance(value, tuple):
            result, metadata = value
            return bool(result.value), metadata
        return bool(value.value)


class ExpressionHandlingPrimitives(Primitive):
    """
    This mixin consists of functions that handle symbolic expressions - evaluations, parsing, computation and more.
    Future functionalities in this mixin might include operations to manipulate expressions, more complex evaluation techniques, etc.
    """

    def init_results(self) -> None:
        """Ensure accumulated expression values are initialized."""

        if not hasattr(self, "_accumulated_results"):
            self._accumulated_results = []

    def get_results(self) -> list[Any]:
        self.init_results()
        return self._accumulated_results

    def clear_results(self) -> None:
        self.init_results()
        self._accumulated_results = []

    def interpret(
        self,
        prompt: str | None = "Evaluate the symbolic expressions and return only the result:\n",
        accumulate: bool = False,
        **kwargs,
    ) -> "Symbol | tuple[Symbol, ResponseMetadata]":
        """
        Evaluates simple symbolic expressions.

        Args:
            prompt (Optional[str]): The prompt to evaluate. Defaults to the symbol value.
            accumulate (bool): If True, stores results for later retrieval. Defaults to False.

        Returns:
            Symbol: A new symbol with the result of the expression evaluation.
        """
        # Propagate original input
        input_value = getattr(self, "_input", self) if hasattr(self, "_input") else self

        execution = _execute_symbol(
            self,
            interpret_request(self, prompt=prompt or ""),
            kwargs,
        )
        if isinstance(execution, tuple):
            result, metadata = execution
        else:
            result = execution
            metadata = None

        if accumulate:
            input_value.init_results()
            input_value._accumulated_results.append(result.value)

        result._input = input_value
        if metadata is not None:
            return result, metadata
        return result


class DataHandlingPrimitives(Primitive):
    """
    This mixin houses functions that clean, summarize and outline symbols or their values.
    Future implementations in this mixin may include various other cleaning and summarization techniques, error detection/correction in symbols, complex filtering, bulk modifications, or other types of condition-based manipulations on symbols, etc.
    """

    def summarize(self, context: str | None = None, **kwargs) -> "Symbol":
        """
        Summarizes the symbol value.

        Args:
            context (Optional[str]): The context to be used for summarization. Defaults to None.

        Returns:
            Symbol: A new symbol with the summarized value.
        """

        return _execute_symbol(
            self,
            summarize_request(self, context=context),
            kwargs,
        )

    def filter(self, criteria: str, include: bool | None = False, **kwargs) -> "Symbol":
        """
        Filters the symbol value based on a specified criteria.

        Args:
            criteria (str): The criteria to filter the symbol value by.
            include (Optional[bool]): Whether to include or exclude items based on the criteria. Defaults to False.

        Returns:
            Symbol: A new symbol with the filtered value.
        """

        return _execute_symbol(
            self,
            filter_request(self, criteria, include=bool(include)),
            kwargs,
        )

    def map(self, instruction: str, **kwargs) -> "Symbol":
        """
        Applies a semantic transformation instruction to each element in an iterable.
        This method transforms each element based on the provided instruction while preserving
        elements that don't match the transformation criteria.

        Args:
            instruction (str): The semantic instruction to apply to each element
            **kwargs: Additional keyword arguments for the transformation

        Returns:
            Symbol: A Symbol object containing the transformed elements

        Raises:
            AssertionError: If the Symbol's value is not iterable or is an unsupported type
        """
        try:
            iter(self.value)
        except TypeError:
            msg = "Map can only be applied to iterable objects"
            raise AssertionError(msg) from None

        return _execute_symbol(
            self,
            map_request(self, instruction),
            kwargs,
            literal=True,
        )

    def modify(self, changes: str, **kwargs) -> "Symbol":
        """
        Modifies the symbol value based on the specified changes.

        Args:
            changes (str): The changes to apply to the symbol value.

        Returns:
            Symbol: A new symbol with the modified value.
        """

        return _execute_symbol(
            self,
            modify_request(self, changes),
            kwargs,
        )

    def replace(self, old: str, new: str, **kwargs) -> "Symbol":
        """
        Replaces one value in the symbol value with another.

        Args:
            old (str): The value to be replaced in the symbol value.
            new (str): The value to replace the existing value with.

        Returns:
            Symbol: A new symbol with the replaced value.
        """

        return _execute_symbol(
            self,
            replace_request(self, old, new),
            kwargs,
        )

    def remove(self, information: str, **kwargs) -> "Symbol":
        """
        Removes a specified piece of information from the symbol value.

        Args:
            information (str): The information to remove from the symbol value.

        Returns:
            Symbol: A new symbol with the removed information.
        """

        return _execute_symbol(
            self,
            replace_request(self, information, ""),
            kwargs,
        )

    def include(self, information: str, **kwargs) -> "Symbol":
        """
        Includes a specified piece of information in the symbol value.

        Args:
            information (str): The information to include in the symbol value.

        Returns:
            Symbol: A new symbol with the included information.
        """

        return _execute_symbol(
            self,
            include_request(self, information),
            kwargs,
        )

    def combine(self, information: str, **kwargs) -> "Symbol":
        """
        Combines the current symbol value with another string.

        Args:
            information (str): The information to combine with the symbol value.

        Returns:
            Symbol: A new symbol with the combined value.
        """

        return _execute_symbol(
            self,
            combine_request(self, information),
            kwargs,
        )


class PatternMatchingPrimitives(Primitive):
    """
    This mixin houses functions that deal with ranking symbols, extracting details based on patterns, and correcting symbols.
    It will house future functionalities that involve sorting, complex pattern detections, advanced correction techniques etc.
    """

    def rank(
        self, measure: str | None = "alphanumeric", order: str | None = "desc", **kwargs
    ) -> "Symbol":
        """
        Ranks the symbol value based on a measure and a sort order.

        Args:
            measure (Optional[str]): The measure to rank the symbol value by. Defaults to 'alphanumeric'.
            order (Optional[str]): The sort order for ranking. Defaults to 'desc'.

        Returns:
            Symbol: A new symbol with the ranked value.
        """

        return _execute_symbol(
            self,
            rank_request(self, measure, order=order or "desc"),
            kwargs,
            literal=True,
        )

    def extract(self, pattern: str, **kwargs) -> "Symbol":
        """
        Extracts data from the symbol value based on a pattern.

        Args:
            pattern (str): The pattern to use for data extraction.

        Returns:
            Symbol: A new symbol with the extracted data.
        """

        return _execute_symbol(
            self,
            extract_request(self, pattern),
            kwargs,
        )

    def translate(self, language: str | None = "English", **kwargs) -> "Symbol":
        """
        Translates the symbol value to the specified language.

        Args:
            language (Optional[str]): The language to translate the value to. Defaults to 'English'.

        Returns:
            Symbol: The translated value as a Symbol.
        """

        return _execute_symbol(
            self,
            translate_request(self, language or "English"),
            kwargs,
            default="Sorry, I do not understand the given language.",
        )


class QueryHandlingPrimitives(Primitive):
    """
    This mixin helps in transforming, preparing, and executing queries, and it is designed to be extendable as new ways of handling queries are developed.
    Future methods could potentially include query optimization, enhanced query formatting, multi-level query execution, query error handling, etc.
    """

    def query(
        self,
        context: str,
        prompt: str | None = None,
        examples: list[Prompt] | None = None,
        **kwargs,
    ) -> "Symbol":
        """
        Queries the symbol value based on a specified context.

        Args:
            context (str): The context used for the query.
            prompt (Optional[str]): The prompt for the query. Defaults to None.
            examples (Optional[List[Prompt]]): The examples for the query. Defaults to None.

        Returns:
            Symbol: The result of the query as a Symbol.
        """

        normalized_examples = tuple(
            value
            for example in (examples or ())
            for value in (example.value if isinstance(example, Prompt) else (str(example),))
        )
        return _execute_symbol(
            self,
            query_request(self, context, prompt=prompt, examples=normalized_examples),
            kwargs,
        )

    def convert(self, format: str, **kwargs) -> "Symbol":
        """
        Converts the symbol value to the specified format.

        Args:
            format (str): The format to convert the value to.

        Returns:
            Symbol: The converted value as a Symbol.
        """

        return _execute_symbol(
            self,
            convert_request(self, format),
            kwargs,
        )


class TemplateStylingPrimitives(Primitive):
    """
    This mixin includes functionalities for stylizing symbols and applying templates.
    Future functionalities might include a variety of new stylizing methods, application of more complex templates, etc.
    """

    def template(
        self,
        template: str,
        placeholder: str = "{{placeholder}}",
    ) -> "Symbol":
        """Apply a local text template to the Symbol value."""

        return self._to_type(template.replace(placeholder, str(self)))

    def style(self, description: str, libraries: list | None = None, **kwargs) -> "Symbol":
        """
        Applies a style to the Symbol.
        It is useful for providing structure and style to the Symbol's value.

        Args:
            description (str): The description of the style to apply.
            libraries (Optional[List]): A list of libraries that may be included in the style. Defaults to an empty list.

        Returns:
            Symbol: A Symbol object with the style applied.
        """
        if libraries is None:
            libraries = []

        return _execute_symbol(
            self,
            style_request(self, description, libraries=libraries),
            kwargs,
        )


class EmbeddingPrimitives(Primitive):
    """Text embeddings and local numeric similarity operations."""

    @staticmethod
    def calculate_mmd(x, y, kernel="rbf", kernel_mul=2.0, kernel_num=5, fix_sigma=None, eps=1e-9):
        def gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma):
            n_samples = source.shape[0] + target.shape[0]
            total = np.concatenate([source, target], axis=0)
            total0 = np.expand_dims(total, 0)
            total1 = np.expand_dims(total, 1)
            l2_distance = np.sum((total0 - total1) ** 2, axis=2)

            bandwidth = fix_sigma or np.sum(l2_distance) / (n_samples**2 - n_samples + eps)
            bandwidth /= kernel_mul ** (kernel_num // 2)
            bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
            kernel_val = [
                np.exp(-l2_distance / (bandwidth_temp + eps)) for bandwidth_temp in bandwidth_list
            ]
            return np.sum(kernel_val, axis=0)

        def linear_mmd2(f_of_x, f_of_y):
            delta = f_of_x.mean(axis=0) - f_of_y.mean(axis=0)
            return np.dot(delta, delta.T)

        if kernel == "linear":
            return linear_mmd2(x, y)
        if kernel == "rbf":
            batch_size = x.shape[0]
            kernels = gaussian_kernel(
                x, y, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma
            )
            xx = np.mean(kernels[:batch_size, :batch_size])
            yy = np.mean(kernels[batch_size:, batch_size:])
            xy = np.mean(kernels[:batch_size, batch_size:])
            yx = np.mean(kernels[batch_size:, :batch_size])
            return xx + yy - xy - yx
        return None

    @overload
    def embed(
        self,
        *,
        dimensions: int | None = None,
        user: str | None = None,
        return_metadata: Literal[False] = False,
        **options: object,
    ) -> "Symbol": ...

    @overload
    def embed(
        self,
        *,
        dimensions: int | None = None,
        user: str | None = None,
        return_metadata: Literal[True],
        **options: object,
    ) -> "tuple[Symbol, ResponseMetadata]": ...
    def embed(
        self,
        *,
        dimensions: int | None = None,
        user: str | None = None,
        return_metadata: bool = False,
        **options: object,
    ) -> "Symbol | tuple[Symbol, ResponseMetadata]":
        """Embed one text value or a sequence of text through the active runtime.

        Binary and other non-text inputs are rejected before runtime execution.
        """

        values = tuple(self.value) if isinstance(self.value, (list, tuple)) else (self.value,)
        if not values or any(not isinstance(value, str) for value in values):
            msg = "Embedding inputs must be non-empty text values"
            raise TypeError(msg)
        if options:
            names = ", ".join(sorted(options))
            msg = f"Unsupported embedding options: {names}"
            raise TypeError(msg)

        response = current_runtime().execute(
            embedding_request(values, dimensions=dimensions, user=user)
        )
        result = self._to_type(parse_embedding_response(response))
        if return_metadata:
            return result, response.metadata
        return result

    @property
    def embedding(self) -> np.ndarray:
        """Return the cached numeric representation as a NumPy array."""

        if self._embedding is None:
            if isinstance(self.value, np.ndarray):
                self._embedding = np.asarray(self.value)
            elif isinstance(self.value, (list, tuple)):
                if not self.value:
                    msg = "Cannot compute embedding of empty list"
                    raise ValueError(msg)
                if all(isinstance(value, self._symbol_type) for value in self.value):
                    self._embedding = np.asarray([value.embedding for value in self.value])
                elif all(isinstance(value, (int, float, bool, np.number)) for value in self.value):
                    self._embedding = np.asarray(self.value)
                else:
                    self._embedding = np.asarray(self.embed().value)
            elif isinstance(self.value, (int, float, bool, np.number)):
                self._embedding = np.asarray(self.value)
            else:
                self._embedding = np.asarray(self.embed().value)

        return self._embedding

    def _ensure_numpy_format(self, x, cast=False):
        # if it is a Symbol, get its value
        if not isinstance(x, (np.ndarray, list)):
            if not isinstance(
                x, self._symbol_type
            ):  # @NOTE: enforce Symbol to avoid circular import
                if not cast:
                    msg = f"Cannot compute similarity with type {type(x)}"
                    raise TypeError(msg)
                x = self._symbol_type(x)
            # evaluate the Symbol as an embedding
            x = x.embedding
        # if it is a list, convert it to numpy
        if isinstance(x, (list, tuple)):
            if not x:
                msg = "Cannot compute similarity with empty list"
                raise ValueError(msg)
            x = np.asarray(x)
        else:
            x = np.asarray(x)

        x = np.squeeze(x)
        if x.ndim == 0:
            x = x[None]

        return x[:, None]

    def _prepare_embedding_operand(self, operand):
        if isinstance(operand, (list, tuple)):
            if self._is_numeric_sequence(operand):
                return self._ensure_numpy_format(operand, cast=True)
            formatted = [self._ensure_numpy_format(item, cast=True) for item in operand]
            return np.concatenate(formatted, axis=1)
        return self._ensure_numpy_format(operand, cast=True)

    def _is_numeric_sequence(self, operand: Iterable):
        for item in operand:
            if isinstance(item, (list, tuple, np.ndarray, self._symbol_type)):
                return False
            if isinstance(item, (numbers.Real, np.generic)):
                continue
            return False
        return True

    def _get_similarity_handler(self, metric, eps, kwargs):
        def _cosine_similarity(lhs, rhs):
            return lhs.T @ rhs / (np.sqrt(lhs.T @ lhs) * np.sqrt(rhs.T @ rhs) + eps)

        def _angular_cosine_similarity(lhs, rhs):
            c = kwargs.get("c", 1)
            return 1 - (
                c
                * np.arccos(lhs.T @ rhs / (np.sqrt(lhs.T @ lhs) * np.sqrt(rhs.T @ rhs) + eps))
                / np.pi
            )

        def _product_similarity(lhs, rhs):
            return lhs.T @ rhs

        def _manhattan_similarity(lhs, rhs):
            return np.abs(lhs - rhs).sum(axis=0, keepdims=True)

        def _euclidean_similarity(lhs, rhs):
            return np.sqrt(np.sum((lhs - rhs) ** 2, axis=0, keepdims=True))

        def _minkowski_similarity(lhs, rhs):
            p = kwargs.get("p", 3)
            return np.sum(np.abs(lhs - rhs) ** p, axis=0, keepdims=True) ** (1 / p)

        def _jaccard_similarity(lhs, rhs):
            intersection = np.minimum(lhs, rhs)
            union = np.maximum(lhs, rhs)
            return np.sum(intersection, axis=0, keepdims=True) / (
                np.sum(union, axis=0, keepdims=True) + eps
            )

        metric_handlers = {
            "cosine": _cosine_similarity,
            "angular-cosine": _angular_cosine_similarity,
            "product": _product_similarity,
            "manhattan": _manhattan_similarity,
            "euclidean": _euclidean_similarity,
            "minkowski": _minkowski_similarity,
            "jaccard": _jaccard_similarity,
        }

        handler = metric_handlers.get(metric)
        if handler is None:
            msg = (
                f"Similarity metric {metric} not implemented. Available metrics: "
                "'cosine', 'angular-cosine', 'product', 'manhattan', 'euclidean', 'minkowski', 'jaccard'"
            )
            raise NotImplementedError(msg)
        return handler

    def _get_kernel_handler(self, kernel):
        kernel_handlers = {
            "gaussian": self._kernel_gaussian,
            "rbf": self._kernel_rbf,
            "laplacian": self._kernel_laplacian,
            "polynomial": self._kernel_polynomial,
            "sigmoid": self._kernel_sigmoid,
            "linear": self._kernel_linear,
            "cauchy": self._kernel_cauchy,
            "t-distribution": self._kernel_t_distribution,
            "inverse-multiquadric": self._kernel_inverse_multiquadric,
            "cosine": self._kernel_cosine,
            "angular-cosine": self._kernel_angular_cosine,
            "mmd": self._kernel_mmd,
        }

        handler = kernel_handlers.get(kernel)
        if handler is None:
            msg = f"Kernel function {kernel} not implemented. Available functions: 'gaussian'"
            raise NotImplementedError(msg)
        return handler

    def _kernel_gaussian(self, lhs, rhs, _eps, kwargs):
        gamma = kwargs.get("gamma", 1)
        return np.exp(-gamma * np.sum((lhs - rhs) ** 2, axis=0))

    def _kernel_rbf(self, lhs, rhs, _eps, kwargs):
        bandwidth = kwargs.get("bandwidth")
        gamma = kwargs.get("gamma", 1)
        distance_sq = np.sum((lhs - rhs) ** 2, axis=0)
        if bandwidth is not None:
            val = 0
            for a in bandwidth:
                gamma = 1.0 / (2 * a)
                val += np.exp(-gamma * distance_sq)
            return val
        return np.exp(-gamma * distance_sq)

    def _kernel_laplacian(self, lhs, rhs, _eps, kwargs):
        gamma = kwargs.get("gamma", 1)
        return np.exp(-gamma * np.sum(np.abs(lhs - rhs), axis=0))

    def _kernel_polynomial(self, lhs, rhs, _eps, kwargs):
        gamma = kwargs.get("gamma", 1)
        degree = kwargs.get("degree", 3)
        coef = kwargs.get("coef", 1)
        return (gamma * np.sum((lhs * rhs), axis=0) + coef) ** degree

    def _kernel_sigmoid(self, lhs, rhs, _eps, kwargs):
        gamma = kwargs.get("gamma", 1)
        coef = kwargs.get("coef", 1)
        return np.tanh(gamma * np.sum((lhs * rhs), axis=0) + coef)

    def _kernel_linear(self, lhs, rhs, _eps, _kwargs):
        return np.sum((lhs * rhs), axis=0)

    def _kernel_cauchy(self, lhs, rhs, _eps, kwargs):
        gamma = kwargs.get("gamma", 1)
        return 1 / (1 + np.sum((lhs - rhs) ** 2, axis=0) / gamma)

    def _kernel_t_distribution(self, lhs, rhs, _eps, kwargs):
        gamma = kwargs.get("gamma", 1)
        degree = kwargs.get("degree", 1)
        return 1 / (1 + (np.sum((lhs - rhs) ** 2, axis=0) / (gamma * degree)) ** (degree + 1) / 2)

    def _kernel_inverse_multiquadric(self, lhs, rhs, _eps, kwargs):
        gamma = kwargs.get("gamma", 1)
        return 1 / np.sqrt(np.sum((lhs - rhs) ** 2, axis=0) / gamma**2 + 1)

    def _kernel_cosine(self, lhs, rhs, eps, _kwargs):
        numerator = np.sum(lhs * rhs, axis=0)
        denominator = np.sqrt(np.sum(lhs**2, axis=0)) * np.sqrt(np.sum(rhs**2, axis=0)) + eps
        return 1 - (numerator / denominator)

    def _kernel_angular_cosine(self, lhs, rhs, eps, kwargs):
        c = kwargs.get("c", 1)
        numerator = np.sum(lhs * rhs, axis=0)
        denominator = np.sqrt(np.sum(lhs**2, axis=0)) * np.sqrt(np.sum(rhs**2, axis=0)) + eps
        return c * np.arccos(numerator / denominator) / np.pi

    def _kernel_mmd(self, lhs, rhs, eps, _kwargs):
        return self.calculate_mmd(lhs.T, rhs.T, eps=eps)

    def similarity(
        self,
        other: "Symbol | list[Any] | np.ndarray",
        metric: Literal[
            "cosine", "angular-cosine", "product", "manhattan", "euclidean", "minkowski", "jaccard"
        ] = "cosine",
        eps: float = 1e-8,
        normalize: Callable | None = None,
        **kwargs,
    ) -> float:
        """
        Calculates the similarity between two Symbol objects using a specified metric.
        This method compares the values of two Symbol objects and calculates their similarity according to the specified metric.
        It supports the 'cosine' metric, and raises a NotImplementedError for other metrics.

        Args:
            other (Symbol): The other Symbol object to calculate the similarity with.
            metric (Optional[str]): The metric to use for calculating the similarity. Defaults to 'cosine'.
            eps (float): A small value to avoid division by zero.
            normalize (Optional[Callable]): A function to normalize the Symbol's value before calculating the similarity. Defaults to None.

        Returns:
            float: The similarity value between the two Symbol objects.

        Raises:
            TypeError: If any of the Symbol objects is not of type np.ndarray or Symbol.
            NotImplementedError: If the given metric is not supported.
        """
        v = self._ensure_numpy_format(self)
        o = self._prepare_embedding_operand(other)
        handler = self._get_similarity_handler(metric, eps, kwargs)
        val = handler(v, o)

        # get the similarity value(s)
        shape = val.shape
        if len(shape) >= 2 and min(shape) > 1:
            val = val.diagonal()
        elif len(shape) < 1 or shape[0] <= 1:
            val = val.item()
        if normalize is not None:
            val = normalize(val)

        return val

    def distance(
        self,
        other: "Symbol | list[Any] | np.ndarray",
        kernel: Literal[
            "gaussian",
            "rbf",
            "laplacian",
            "polynomial",
            "sigmoid",
            "linear",
            "cauchy",
            "t-distribution",
            "inverse-multiquadric",
            "cosine",
            "angular-cosine",
            "mmd",
        ] = "gaussian",
        eps: float = 1e-8,
        normalize: Callable | None = None,
        **kwargs,
    ) -> float:
        """
        Calculates the kernel between two Symbol objects.

        Args:
            other (Symbol): The other Symbol object to calculate the kernel with.
            kernel (Optional[str]): The function to use for calculating the kernel. Defaults to 'gaussian'.
            normalize (Optional[Callable]): A function to normalize the Symbol's value before calculating the kernel. Defaults to None.
            **kwargs: Additional keyword arguments for the kernel arguments (e.g. gamma, coef).

        Returns:
            float: The kernel value between the two Symbol objects.

        Raises:
            TypeError: If any of the Symbol objects is not of type np.ndarray or Symbol.
            NotImplementedError: If the given kernel is not supported.
        """
        v = self._ensure_numpy_format(self)
        o = self._prepare_embedding_operand(other)
        handler = self._get_kernel_handler(kernel)
        val = handler(v, o, eps, kwargs)

        # get the kernel value(s)
        shape = val.shape
        val = val if len(shape) >= 1 and shape[0] > 1 else val.item()
        if normalize is not None:
            val = normalize(val)
        return val

    def zip(
        self,
        *,
        dimensions: int | None = None,
        user: str | None = None,
    ) -> list[tuple[str, Any, dict[str, str]]]:
        """Pair text values with embeddings and stable query records."""
        if isinstance(self.value, str):
            self._value = [self.value]
        elif isinstance(self.value, list):
            pass
        else:
            msg = f"Expected id to be a string, got {type(self.value)}"
            raise ValueError(msg)

        embeds = self.embed(dimensions=dimensions, user=user).value
        idx = [str(uuid.uuid4()) for _ in range(len(self.value))]
        query = [{"text": str(self.value[i])} for i in range(len(self.value))]

        # convert embeds to list if it is a numpy array
        if isinstance(embeds, np.ndarray):
            embeds = embeds.tolist()

        return list(zip(idx, embeds, query, strict=False))


# @TODO: add tests


class PersistencePrimitives(Primitive):
    """
    This mixin contains functionalities related to expanding symbols and saving/loading symbols to/from disk.
    Future functionalities in this mixin might include different ways of serialization and deserialization, or more complex expansion techniques etc.
    """

    def save(self, path: str, replace: bool | None = False, serialize: bool | None = True) -> None:
        """
        Save the current Symbol to a file.

        Args:
            path (str): The filepath of the saved file.
            replace (Optional[bool]): Whether to replace the file if it already exists. Defaults to False.
            serialize (Optional[bool]): Whether to serialize the object via pickle instead of writing the string. Defaults to True.

        Returns:
            Symbol: The current Symbol.
        """
        file_path = Path(path)

        if not replace:
            cnt = 0
            candidate = file_path
            while candidate.exists():
                candidate = candidate.with_name(f"{file_path.stem}_{cnt}{file_path.suffix}")
                cnt += 1
            file_path = candidate

        if serialize:
            # serialize the object via pickle instead of writing the string
            path_str = str(file_path)
            pickle_path = Path(path_str if path_str.endswith(".pkl") else f"{path_str}.pkl")
            with pickle_path.open("wb") as f:
                pickle.dump(self, file=f)
        else:
            with file_path.open("w") as f:
                f.write(str(self))

    def load(self, path: str) -> Any:
        """
        Load a Symbol from a file.

        Args:
            path (str): The filepath of the saved file.

        Returns:
            Symbol: The loaded Symbol.
        """
        with Path(path).open("rb") as f:
            return pickle.load(f)


# @TODO: add tests
