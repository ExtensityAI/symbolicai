import ast
import logging
import numbers
import pickle
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union

import numpy as np

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
    parse_literal_or_text_output,
    parse_stripped_output,
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
from symai.runtime.models import LanguageModelRequest
from symai.runtime.runtime import current_runtime
from symai.utils import Extra, missing_dependency

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from symai.symbol import Symbol


def _execute_language_value(
    symbol: "Symbol",
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
    if literal:
        return parse_literal_or_text_output(
            response,
            index=output_index,
            default=default,
            limit=limit,
        )
    return parse_stripped_output(
        response,
        return_type,
        index=output_index,
        default=default,
        limit=limit,
    )


def _execute_symbol(
    symbol: "Symbol",
    request: LanguageModelRequest,
    kwargs: dict[str, object],
    *,
    return_type: type = str,
    default: object = None,
    limit: int | None = 1,
    literal: bool = False,
):
    output_index = kwargs.pop("output_index", 0)
    return_metadata = kwargs.pop("return_metadata", False)
    requested_return_type = kwargs.pop("return_type", return_type)
    default = kwargs.pop("default", default)
    requested_limit = kwargs.pop("limit", limit)
    forbidden = {"engine", "model", "provider"}.intersection(kwargs)
    if forbidden:
        names = ", ".join(sorted(forbidden))
        msg = f"Provider/model selection belongs to runtime configuration, not per-call kwargs: {names}"
        raise TypeError(msg)
    if kwargs:
        names = ", ".join(sorted(kwargs))
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
    if literal:
        value = parse_literal_or_text_output(
            response,
            index=output_index,
            default=default,
            limit=limit,
        )
    else:
        value = parse_stripped_output(
            response,
            return_type,
            index=output_index,
            default=default,
            limit=limit,
        )
    result = symbol._to_type(value)
    if return_metadata:
        return result, response.metadata
    return result


class Primitive:
    # DO NOT use by default neuro-symbolic iterations for mixins to avoid unwanted side effects
    __semantic__ = False
    # disable the entire NeSy engine access
    __disable_nesy_engine__ = False
    # disable None shortcut
    __disable_none_shortcut__ = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # by default, disable shortcut matches and neuro-symbolic iterations
        self.__semantic__ = self.__semantic__ or Primitive.__semantic__
        self.__disable_nesy_engine__ = (
            self.__disable_nesy_engine__ or Primitive.__disable_nesy_engine__
        )
        self.__disable_none_shortcut__ = (
            self.__disable_none_shortcut__ or Primitive.__disable_none_shortcut__
        )


class OperatorPrimitives(Primitive):
    __hash__ = None

    def __try_type_specific_func(self, other, func, op: str | None = None):
        if not isinstance(other, self._symbol_type):
            other = self._to_type(other)
        # None shortcut
        if not self.__disable_none_shortcut__ and (self.value is None or other.value is None):
            msg = f"unsupported {self._symbol_type.__class__} value operand type(s) for {op}: '{type(self.value)}' and '{type(other.value)}'"
            raise TypeError(msg)
        # try type specific function
        try:
            # try type specific function
            value = func(self, other)
            if value is NotImplemented:
                operation = "" if op is None else op
                msg = f"unsupported {self._symbol_type.__class__} value operand type(s) for {operation}: '{type(self.value)}' and '{type(other.value)}'"
                raise TypeError(msg)
            return value
        except Exception as ex:
            self._metadata._error = ex
            pass
        return None

    def __throw_error_on_nesy_engine_call(self, func):
        """
        This function raises an error if the neuro-symbolic engine is disabled.
        """
        if self.__disable_nesy_engine__:
            msg = f"unsupported {self.__class__} value operand type(s) for {func.__name__}: '{type(self.value)}'"
            raise TypeError(msg)

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

    """
    This mixin contains functions that perform arithmetic operations on symbols or symbol values.
    The functions in this mixin are bound to the 'neurosymbolic' engine for evaluation.
    """

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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return result

        self.__throw_error_on_nesy_engine_call(self.__contains__)

        value = _execute_language_value(
            self,
            contains_request(self, other),
            return_type=bool,
            default=False,
        )
        return self._to_type(value)

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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__eq__)

        value = _execute_language_value(
            self,
            equals_request(self, other),
            return_type=bool,
            default=False,
        )
        return self._to_type(value)

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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return result

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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__gt__)

        value = _execute_language_value(
            self,
            compare_request(self, ">", other),
            return_type=bool,
            default=False,
        )
        return self._to_type(value)

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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__lt__)

        value = _execute_language_value(
            self,
            compare_request(self, "<", other),
            return_type=bool,
            default=False,
        )
        return self._to_type(value)

    def __le__(self, other) -> bool:
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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__le__)

        value = _execute_language_value(
            self,
            compare_request(self, "<=", other),
            return_type=bool,
            default=False,
        )
        return self._to_type(value)

    def __ge__(self, other) -> bool:
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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__ge__)

        value = _execute_language_value(
            self,
            compare_request(self, ">=", other),
            return_type=bool,
            default=False,
        )
        return self._to_type(value)

    def __neg__(self) -> "Symbol":
        """
        Return the negated value of the Symbol.

        Returns:
            Symbol: The negated value of the Symbol.
        """
        result = self.__try_type_specific_func(False, lambda self, _: -self.value, op="-")

        if not self.__semantic__:
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__neg__)

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

        if not self.__semantic__:
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__invert__)

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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__lshift__)

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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__rlshift__)

        value = _execute_language_value(self, include_request(self, other))
        return self._to_type(value)

    def __ilshift__(self, other: Any) -> "Symbol":
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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            self._value = result
            return self

        self.__throw_error_on_nesy_engine_call(self.__ilshift__)

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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__rshift__)

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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__rrshift__)

        value = _execute_language_value(self, include_request(self, other))
        return self._to_type(value)

    def __irshift__(self, other: Any) -> "Symbol":
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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            self._value = result
            return self

        self.__throw_error_on_nesy_engine_call(self.__irshift__)

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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__add__)

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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__radd__)

        value = _execute_language_value(self, combine_request(other, self))
        return self._to_type(value)

    def __iadd__(self, other: Any) -> "Symbol":
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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            self._value = result
            return self
        other = self._to_type(other)
        self._value = self.__add__(other)

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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__sub__)

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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__rsub__)

        other = self._to_type(other)
        value = _execute_language_value(self, replace_request(other, self, ""))
        return self._to_type(value)

    def __isub__(self, other: Any) -> "Symbol":
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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            self._value = result
            return self
        val = self.__sub__(other)
        self._value = val.value

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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__and__)

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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__rand__)

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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            self._value = result
            return self

        self.__throw_error_on_nesy_engine_call(self.__iand__)

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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__or__)

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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__ror__)

        other = self._to_type(other)
        value = _execute_language_value(self, logic_request(other, "or", self))
        return self._to_type(value)

    def __ior__(self, other: Any) -> "Symbol":
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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            self._value = result
            return self
        result = self._to_type(str(self) + str(other))

        self.__throw_error_on_nesy_engine_call(self.__ior__)

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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__xor__)

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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            return self._to_type(result)

        self.__throw_error_on_nesy_engine_call(self.__rxor__)

        value = _execute_language_value(self, logic_request(other, "xor", self))
        return self._to_type(value)

    def __ixor__(self, other: Any) -> "Symbol":
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

        if not self.__semantic__ and not getattr(other, "__semantic__", False):
            self._value = result
            return self

        self.__throw_error_on_nesy_engine_call(self.__ixor__)

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

    def __imatmul__(self, other: Any) -> "Symbol":
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
        if result is not None:
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
        if result is not None:
            return self._to_type(result)

        msg = "Division operation not supported semantically! Might change in the future."
        raise NotImplementedError(msg)

    def __itruediv__(self, other: Any) -> "Symbol":
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
        if result is not None:
            self._value = result
            return self

        msg = "Division operation not supported semantically! Might change in the future."
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
        if result is not None:
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
        if result is not None:
            return self._to_type(result)

        msg = "Floor division operation not supported semantically! Might change in the future."
        raise NotImplementedError(msg)

    def __ifloordiv__(self, other: Any) -> "Symbol":
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
        if result is not None:
            self._value = result
            return self

        msg = "Floor division operation not supported semantically! Might change in the future."
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
        if result is not None:
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
        if result is not None:
            return self._to_type(result)

        msg = "Power operation not supported semantically! Might change in the future."
        raise NotImplementedError(msg)

    def __ipow__(self, other: Any) -> "Symbol":
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
        if result is not None:
            self._value = result
            return self

        msg = "Power operation not supported semantically! Might change in the future."
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
        if result is not None:
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
        if result is not None:
            return self._to_type(result)

        msg = "Modulo operation not supported! Might change in the future."
        raise NotImplementedError(msg) from self._metadata._error

    def __imod__(self, other: Any) -> "Symbol":
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
        if result is not None:
            self._value = result
            return self

        msg = "Modulo operation not supported semantically! Might change in the future."
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
        if result is not None:
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
        if result is not None:
            return self._to_type(result)

        msg = "Multiply operation not supported semantically! Might change in the future."
        raise NotImplementedError(msg)

    def __imul__(self, other: Any) -> "Symbol":
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
        if result is not None:
            self._value = result
            return self

        msg = "Multiply operation not supported semantically! Might change in the future."
        raise NotImplementedError(msg)


class CastingPrimitives(Primitive):
    """
    This mixin contains functionalities related to casting symbols.
    """

    @property
    def syn(self) -> "Symbol":
        """
        Return a syntactic (non-semantic) view of this Symbol.
        """
        if not getattr(self, "__semantic__", False):
            return self
        return self._to_type(self.value, semantic=False)

    @property
    def sem(self) -> "Symbol":
        """
        Return a semantic view of this Symbol.
        (Useful after calling `.syn` in a chain.)
        """
        if getattr(self, "__semantic__", False):
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
    """
    This mixin contains functions that perform iteration operations on symbols or symbol values.
    The functions in this mixin are bound to the 'neurosymbolic' engine for evaluation.
    """

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
        if not self.__semantic__:
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

        if not self.__semantic__:
            try:
                self.value[key] = value
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

        if not self.__semantic__:
            try:
                del self.value[key]
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
    """
    This mixin includes functions responsible for handling symbol values - tokenization, type retrieval, value casting, indexing, etc.
    Future functions might include different methods of processing or manipulating the values of symbols, working with metadata of values, etc.
    """

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

    def split(self, delimiter: str, **_kwargs) -> "Symbol":
        """
        Splits the symbol value by a specified delimiter.

        Args:
            delimiter (str): The delimiter to split the symbol value by.

        Returns:
            Symbol: A new symbol with the split value.
        """
        assert isinstance(delimiter, str), f"delimiter must be a string, got {type(delimiter)}"
        assert isinstance(self.value, str), f"self.value must be a string, got {type(self.value)}"
        return self._to_type([*self.value.split(delimiter)])

    def join(self, delimiter: str = " ", **_kwargs) -> "Symbol":
        """
        Joins the symbol value with a specified delimiter.

        Args:
            delimiter (str, optional): The delimiter to join the symbol value with. Defaults to ' '.

        Returns:
            Symbol: A new symbol with the joined str value.
        """
        assert isinstance(delimiter, str), f"delimiter must be a string, got {type(delimiter)}"
        assert isinstance(self.value, Iterable), (
            f"value must be an iterable, got {type(self.value)}"
        )
        return self._to_type(delimiter.join(self.value))

    def startswith(self, prefix: str, **_kwargs) -> bool:
        """
        Checks if the symbol value starts with a specified prefix.

        Args:
            prefix (str): The prefix to check if the symbol value starts with.

        Returns:
            bool: True if the symbol value starts with the specified prefix, otherwise False.
        """
        assert isinstance(prefix, str), f"prefix must be a string, got {type(prefix)}"
        assert isinstance(self.value, str), f"self.value must be a string, got {type(self.value)}"

        if not self.__semantic__:
            return self.value.startswith(prefix)

        value = _execute_language_value(
            self,
            startswith_request(self, prefix),
            return_type=bool,
            default=False,
        )
        return bool(value)

    def endswith(self, suffix: str, **_kwargs) -> bool:
        """
        Checks if the symbol value ends with a specified suffix.

        Args:
            suffix (str): The suffix to check if the symbol value ends with.

        Returns:
            bool: True if the symbol value ends with the specified suffix, otherwise False.
        """
        assert isinstance(suffix, str), f"suffix must be a string, got {type(suffix)}"
        assert isinstance(self.value, str), f"self.value must be a string, got {type(self.value)}"

        if not self.__semantic__:
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_results(self):
        """Ensures _accumulated_results exists, initializing if needed."""
        if not hasattr(self, "_accumulated_results"):
            self._accumulated_results = []

    def get_results(self) -> list["Symbol"]:
        """
        Retrieves accumulated results from previous interpretations.

        Returns:
            List[Symbol]: List of accumulated results
        """
        self.init_results()
        return self._accumulated_results

    def clear_results(self):
        """Clears the accumulated results"""
        self.init_results()
        self._accumulated_results = []

    def interpret(
        self,
        prompt: str | None = "Evaluate the symbolic expressions and return only the result:\n",
        accumulate: bool = False,
        **kwargs,
    ) -> "Symbol":
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
        that, template: str, placeholder: str | None = "{{placeholder}}", **_kwargs
    ) -> "Symbol":
        """
        Applies a template to the Symbol.
        It is useful for providing structure to the Symbol's value.

        Args:
            template (str): The template to apply to the Symbol.
            placeholder (Optional[str]): The placeholder in the template to be replaced with the Symbol's value. Defaults to '{{placeholder}}'.

        Returns:
            Symbol: A Symbol object with a template applied.
        """

        def _func(self):
            res = template.replace(placeholder, str(self))
            return that._to_type(res)

        return _func(that)

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
    """
    This mixin contains functionalities that deal with embedding symbol values.
    New functionalities in this mixin might include different types of embedding methods, similarity and distance measures etc.
    """

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on a
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on a
                representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        try:
            from scipy import linalg  # noqa: PLC0415
        except ImportError:
            raise missing_dependency(Extra.CLUSTER, "scipy") from None

        mu1 = np.atleast_1d(mu1).squeeze()
        mu2 = np.atleast_1d(mu2).squeeze()

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, (
            "Training and test covariances have different dimensions"
        )

        diff = mu1 - mu2

        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        if not np.isfinite(covmean).all():
            msg = (
                f"fid calculation produces singular product; adding {eps} "
                "to diagonal of cov estimates"
            )
            logger.warning(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                msg = f"Imaginary component {m}"
                raise ValueError(msg)
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

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

    def embed(self, **kwargs) -> "Symbol":
        """
        Generates embeddings for the Symbol's value.
        If the value is not a list, it is converted to a list.

        Supports multimodal inputs: text (str), images (bytes/Part), and mixed content (Content).
        Each engine handles unpacking based on its capabilities.

        Note: When passing raw bytes, MIME type detection is handled internally via the filetype
        library (supports 100+ formats). Users can also pass Part objects for explicit MIME type control.

        Args:

        Returns:
            Symbol: A Symbol object with its value embedded.
        """
        values = self.value if isinstance(self.value, list) else [self.value]
        if any(isinstance(value, (bytes, bytearray)) for value in values):
            msg = "Embedding inputs must be text"
            raise TypeError(msg)
        inputs = tuple(str(value) for value in values)
        dimensions = kwargs.pop("dimensions", None)
        user = kwargs.pop("user", None)
        return_metadata = kwargs.pop("return_metadata", False)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            msg = f"Unsupported embedding options: {names}"
            raise TypeError(msg)
        response = current_runtime().execute(
            embedding_request(inputs, dimensions=dimensions, user=user)
        )
        result = self._to_type(parse_embedding_response(response))
        if return_metadata:
            return result, response.metadata
        return result

    @property
    def embedding(self) -> np.array:
        """
        Get the embedding as a numpy array.

        Returns:
            Any: The embedding of the symbol.
        """
        # if the embedding is not yet computed, compute it
        if self._metadata.embedding is None:
            if (
                isinstance(self.value, (list, tuple))
                and all(isinstance(x, (int, float, bool)) for x in self.value)
            ) or isinstance(self.value, np.ndarray):
                if isinstance(self.value, (list, tuple)):
                    assert len(self.value) > 0, "Cannot compute embedding of empty list"
                    symbol_type = self._symbol_type
                    if isinstance(self.value[0], symbol_type):
                        # convert each element to numpy array
                        self._metadata.embedding = np.asarray([x.embedding for x in self.value])
                    elif isinstance(self.value[0], str):
                        # embed each string
                        self._metadata.embedding = np.asarray(
                            [symbol_type(x).embedding for x in self.value]
                        )
                    else:
                        # convert to numpy array
                        self._metadata.embedding = np.asarray(self.value)
                else:
                    # convert to numpy array
                    self._metadata.embedding = np.asarray(self.value)
            else:
                # compute the embedding and store as numpy array
                self._metadata.embedding = np.asarray(self.embed().value)
        if isinstance(self._metadata.embedding, list):
            self._metadata.embedding = np.asarray(self._metadata.embedding)
        # return the embedding
        return self._metadata.embedding

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
            assert len(x) > 0, "Cannot compute similarity with empty list"
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
            "frechet": self._kernel_frechet,
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

    def _kernel_frechet(self, lhs, rhs, eps, kwargs):
        sigma1 = kwargs.get("sigma1")
        sigma2 = kwargs.get("sigma2")
        assert sigma1 is not None and sigma2 is not None, (
            "Frechet distance requires covariance matrices for both inputs"
        )
        return self.calculate_frechet_distance(lhs.T, sigma1, rhs.T, sigma2, eps)

    def _kernel_mmd(self, lhs, rhs, eps, _kwargs):
        return self.calculate_mmd(lhs.T, rhs.T, eps=eps)

    def similarity(
        self,
        other: Union["Symbol", list, np.ndarray],
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
        other: Union["Symbol", list, np.ndarray],
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
            "frechet",
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

    def zip(self, **kwargs) -> list[tuple[str, list, dict]]:
        """
        Zips the Symbol's value with its embeddings and a query containing the value.
        This method zips the Symbol's value along with its embeddings and a query containing the value.

        Args:
            **kwargs: Additional keyword arguments for the `embed` method.

        Returns:
            List[Tuple[str, List, Dict]]: A list of tuples containing a unique ID, the value's embeddings, and a query containing the value.

        Raises:
            ValueError: If the Symbol's value is not a string or list of strings.
        """
        if isinstance(self.value, str):
            self._value = [self.value]
        elif isinstance(self.value, list):
            pass
        else:
            msg = f"Expected id to be a string, got {type(self.value)}"
            raise ValueError(msg)

        embeds = self.embed(**kwargs).value
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
