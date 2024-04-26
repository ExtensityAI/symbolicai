import ast
import os
import pickle
import uuid
import torch
import numpy as np

from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple,
                    Type, Union)

from .measures import calculate_frechet_distance, calculate_mmd
from .. import core
from .. import core_ext
from ..prompts import Prompt

if TYPE_CHECKING:
    from ..symbol import Expression, Symbol


class Primitive:
    # smart defaults to prefer type specific functions over neuro-symbolic iterations
    __disable_shortcut_matches__   = False
    # DO NOT use by default neuro-symbolic iterations for mixins to avoid unwanted side effects
    __nesy_iteration_primitives__  = False
    # disable the entire NeSy engine access
    __disable_nesy_engine__        = False
    # disable None shortcut
    __disable_none_shortcut__      = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # by default, disable shortcut matches and neuro-symbolic iterations
        self.__disable_shortcut_matches__  = self.__disable_shortcut_matches__ or Primitive.__disable_shortcut_matches__
        self.__nesy_iteration_primitives__ = self.__nesy_iteration_primitives__ or Primitive.__nesy_iteration_primitives__
        self.__disable_nesy_engine__       = self.__disable_nesy_engine__ or Primitive.__disable_nesy_engine__
        self.__disable_none_shortcut__     = self.__disable_none_shortcut__ or Primitive.__disable_none_shortcut__

    @staticmethod
    def _is_iterable(value):
        return isinstance(value, (list, tuple, set, dict, bytes, bytearray, range, torch.Tensor, np.ndarray))


class ArithmeticPrimitives(Primitive):
    def __try_type_specific_func(self, other, func, op: str = None):
        if self.__disable_shortcut_matches__:
            return None
        if not isinstance(other, self._symbol_type):
            other = self._to_symbol(other)
        # None shortcut
        if not self.__disable_none_shortcut__:
            if  self.value is None or other.value is None:
                raise TypeError(f"unsupported {self._symbol_type.__class__} value operand type(s) for {op}: '{type(self.value)}' and '{type(other.value)}'")
        # try type specific function
        try:
            # try type specific function
            value = func(self, other)
            if value is NotImplemented:
                operation = '' if op is None else op
                raise TypeError(f"unsupported {self._symbol_type.__class__} value operand type(s) for {operation}: '{type(self.value)}' and '{type(other.value)}'")
            return value
        except Exception as ex:
            self._metadata._error = ex
            pass
        return None

    def __throw_error_on_nesy_engine_call(self, func):
        '''
        This function raises an error if the neuro-symbolic engine is disabled.
        '''
        if self.__disable_nesy_engine__:
            raise TypeError(f"unsupported {self.__class__} value operand type(s) for {func.__name__}: '{type(self.value)}'")

    '''
    This mixin contains functions that perform arithmetic operations on symbols or symbol values.
    The functions in this mixin are bound to the 'neurosymbolic' engine for evaluation.
    '''
    def __contains__(self, other: Any) -> bool:
        '''
        Check if a Symbol object is present in another Symbol object.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to check for containment.

        Returns:
            bool: True if the current Symbol contains the 'other' Symbol, otherwise False.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: other.value in self.value, op='in')
        # verify the result and return if found return
        if result is not None and result is not False:
            return result
        # allow for fuzzy matches between types
        other = self._to_symbol(other)
        if type(self.value) == str and \
            (type(other.value) == int or \
             type(other.value) == float or \
             type(other.value) == bool):
            result = str(other.value) in self.value
            if result:
                return result
        # verify if fuzzy matches are enabled in general
        # DO NOT use by default neuro-symbolic iterations for mixins to avoid unwanted side effects
        # check if value is iterable
        # except for str
        if type(self.value) != str and (not self.__nesy_iteration_primitives__ or Primitive._is_iterable(self.value)):
            return result

        self.__throw_error_on_nesy_engine_call(self.__contains__)

        @core.contains()
        def _func(_, other) -> bool:
            pass

        return _func(self, other)

    def __eq__(self, other: Any) -> bool:
        '''
        Check if the current Symbol is equal to another Symbol.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to check for equality.

        Returns:
            bool: True if the current Symbol is equal to the 'other' Symbol, otherwise False.
        '''
        # First verify if not identical (same object)
        if self is other:
            return True
        # Then verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value == other.value, op='==')
        # verify the result and return if found return
        if result is not None and result is not False:
            return result

        self.__throw_error_on_nesy_engine_call(self.__eq__)

        @core.equals()
        def _func(_, other) -> bool:
            pass

        return _func(self, other)

    def __ne__(self, other: Any) -> bool:
        '''
        This method checks if a Symbol object is not equal to another Symbol by using the __eq__ method.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to check for inequality.

        Returns:
            bool: True if the current Symbol is not equal to the 'other' Symbol, otherwise False.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other:  self.value != other.value, op='!=')
        # verify the result and return if found return
        if result is not None and result is not False:
            return result
        return not self.__eq__(other)

    def __gt__(self, other: Any) -> bool:
        '''
        This method checks if a Symbol object is greater than another Symbol using the @core.compare() decorator with the '>' operator.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to compare.

        Returns:
            bool: True if the current Symbol is greater than the 'other' Symbol, otherwise False.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value > other.value, op='>')
        # verify the result and return if found return
        if result is not None and result is not False:
            return result

        self.__throw_error_on_nesy_engine_call(self.__gt__)

        @core.compare(operator = '>')
        def _func(_, other) -> bool:
            pass
        return _func(self, other)

    def __lt__(self, other: Any) -> bool:
        '''
        This method checks if a Symbol object is less than another Symbol using the @core.compare() decorator with the '<' operator.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to compare.

        Returns:
            bool: True if the current Symbol is less than the 'other' Symbol, otherwise False.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value < other.value, op='<')
        # verify the result and return if found return
        if result is not None and result is not False:
            return result

        self.__throw_error_on_nesy_engine_call(self.__lt__)

        @core.compare(operator = '<')
        def _func(_, other) -> bool:
            pass
        return _func(self, other)

    def __le__(self, other) -> bool:
        '''
        This method checks if a Symbol object is less than or equal to another Symbol using the @core.compare() decorator with the '<=' operator.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to compare.

        Returns:
            bool: True if the current Symbol is less than or equal to the 'other' Symbol, otherwise False.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value <= other.value, op='<=')
        # verify the result and return if found return
        if result is not None and result is not False:
            return result

        self.__throw_error_on_nesy_engine_call(self.__le__)

        @core.compare(operator = '<=')
        def _func(_, other) -> bool:
            pass
        return _func(self, other)

    def __ge__(self, other) -> bool:
        '''
        This method checks if a Symbol object is greater than or equal to another Symbol using the @core.compare() decorator with the '>=' operator.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to compare.

        Returns:
            bool: True if the current Symbol is greater than or equal to the 'other' Symbol, otherwise False.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value >= other.value, op='>=')
        # verify the result and return if found return
        if result is not None and result is not False:
            return result

        self.__throw_error_on_nesy_engine_call(self.__ge__)

        @core.compare(operator = '>=')
        def _func(_, other) -> bool:
            pass
        return _func(self, other)

    def __neg__(self) -> 'Symbol':
        '''
        Return the negated value of the Symbol.
        The method uses the @core.negate decorator to compute the negation of the Symbol value.

        Returns:
            Symbol: The negated value of the Symbol.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(False, lambda self, _: -self.value, op='-')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)

        self.__throw_error_on_nesy_engine_call(self.__neg__)

        @core.negate()
        def _func(_):
            pass
        return self._to_symbol(_func(self))

    def __not__(self) -> 'Symbol':
        '''
        Return the negated value of the Symbol.
        The method uses the @core.negate decorator to compute the negation of the Symbol value.

        Returns:
            Symbol: The negated value of the Symbol.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(False, lambda self, _: not self.value, op='not')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)

        self.__throw_error_on_nesy_engine_call(self.__not__)

        @core.negate()
        def _func(_):
            pass
        return self._to_symbol(_func(self))

    def __invert__(self) -> 'Symbol':
        '''
        Return the inverted value of the Symbol.
        The method uses the @core.invert decorator to compute the inversion of the Symbol value.

        Returns:
            Symbol: The inverted value of the Symbol.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(False, lambda self, _: ~self.value, op='~')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)

        self.__throw_error_on_nesy_engine_call(self.__invert__)

        @core.invert()
        def _func(_):
            pass
        return self._to_symbol(_func(self))

    def __lshift__(self, other: Any) -> 'Symbol':
        '''
        Add new information to the Symbol.
        The method uses the @core.include decorator to incorporate information into the Symbol.

        Args:
            information (Any): The information to include in the Symbol.

        Returns:
            Symbol: The Symbol with the new information included.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value << other.value, op='<<')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)

        self.__throw_error_on_nesy_engine_call(self.__lshift__)

        @core.include()
        def _func(_, information: str):
            pass
        return self._to_symbol(_func(self, other))

    def __rlshift__(self, other: Any) -> 'Symbol':
        '''
        Add new information to the Symbol.
        The method uses the @core.include decorator to incorporate information into the Symbol.

        Args:
            information (Any): The information to include in the Symbol.

        Returns:
            Symbol: The Symbol with the new information included.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: other.value << self.value, op='<<')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)

        self.__throw_error_on_nesy_engine_call(self.__rlshift__)

        @core.include()
        def _func(_, information: str):
            pass
        return self._to_symbol(_func(self, other))

    def __ilshift__(self, other: Any) -> 'Symbol':
        '''
        Add new information to the Symbol.
        The method uses the @core.include decorator to incorporate information into the Symbol.

        Args:
            information (Any): The information to include in the Symbol.

        Returns:
            Symbol: The Symbol with the new information included.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value << other.value, op='<<=')
        # verify the result and return if found return
        if result is not None and result is not False:
            self._value = result
            return self

        self.__throw_error_on_nesy_engine_call(self.__ilshift__)

        @core.include()
        def _func(_, information: str):
            pass
        self._value = _func(self, other)
        return self

    def __rshift__(self, other: Any) -> 'Symbol':
        '''
        Add new information to the Symbol.
        The method uses the @core.include decorator to incorporate information into the Symbol.

        Args:
            information (Any): The information to include in the Symbol.

        Returns:
            Symbol: The Symbol with the new information included.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value >> other.value, op='>>')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)

        self.__throw_error_on_nesy_engine_call(self.__rshift__)

        @core.include()
        def _func(_, information: str):
            pass
        return self._to_symbol(_func(self, other))

    def __rrshift__(self, other: Any) -> 'Symbol':
        '''
        Add new information to the Symbol.
        The method uses the @core.include decorator to incorporate information into the Symbol.

        Args:
            information (Any): The information to include in the Symbol.

        Returns:
            Symbol: The Symbol with the new information included.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: other.value >> self.value, op='>>')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)

        self.__throw_error_on_nesy_engine_call(self.__rrshift__)

        @core.include()
        def _func(_, information: str):
            pass
        return self._to_symbol(_func(self, other))

    def __irshift__(self, other: Any) -> 'Symbol':
        '''
        Add new information to the Symbol.
        The method uses the @core.include decorator to incorporate information into the Symbol.

        Args:
            information (Any): The information to include in the Symbol.

        Returns:
            Symbol: The Symbol with the new information included.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value >> other.value, op='>>=')
        # verify the result and return if found return
        if result is not None and result is not False:
            self._value = result
            return self

        self.__throw_error_on_nesy_engine_call(self.__irshift__)

        @core.include()
        def _func(_, information: str):
            pass
        self._value = _func(self, other)
        return self

    def __add__(self, other: Any) -> 'Symbol':
        '''
        Combine the Symbol with another value.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.
        The method uses the @core.combine decorator to merge the Symbol and the other value.

        Args:
            other: The value to combine with the Symbol.

        Returns:
            Symbol: The Symbol combined with the other value.
        '''
        # prefer nesy engine over type specific functions for str since the default string concatenation operator in SymbolicAI is '|'
        if (isinstance(self.value, str) and isinstance(other, str)) or \
            (isinstance(self.value, str) and isinstance(other, self._symbol_type) and isinstance(other.value, str)):
            result = None
        else:
            # Otherwise verify for specific type support
            result = self.__try_type_specific_func(other, lambda self, other: self.value + other.value, op='+')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)

        self.__throw_error_on_nesy_engine_call(self.__add__)

        @core.combine()
        def _func(_, a: str, b: str):
            pass
        return self._to_symbol(_func(self, other))

    def __radd__(self, other) -> 'Symbol':
        '''
        Combine another value with the Symbol.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.
        The method uses the @core.combine decorator to merge the other value and the Symbol.

        Args:
            other (Any): The value to combine with the Symbol.

        Returns:
            Symbol: The other value combined with the Symbol.
        '''
        # prefer nesy engine over type specific functions for str since the default string concatenation operator in SymbolicAI is '|'
        if (isinstance(self.value, str) and isinstance(other, str)) or \
            (isinstance(self.value, str) and isinstance(other, self._symbol_type) and isinstance(other.value, str)):
            result = None
        else:
            # Otherwise verify for specific type support
            result = self.__try_type_specific_func(other, lambda self, other: other.value + self.value, op='+')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)

        self.__throw_error_on_nesy_engine_call(self.__radd__)

        @core.combine()
        def _func(_, a: str, b: str):
            pass
        other = self._to_symbol(other)
        return self._to_symbol(_func(other, self))

    def __iadd__(self, other: Any) -> 'Symbol':
        '''
        This method adds another value to the Symbol and updates its value with the result.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The value to add to the Symbol.

        Returns:
            Symbol: The updated Symbol with the added value.
        '''
        # prefer nesy engine over type specific functions for str since the default string concatenation operator in SymbolicAI is '|'
        if (isinstance(self.value, str) and isinstance(other, str)) or \
            (isinstance(self.value, str) and isinstance(other, self._symbol_type) and isinstance(other.value, str)):
            result = None
        else:
            # Otherwise verify for specific type support
            result = self.__try_type_specific_func(other, lambda self, other: self.value + other.value, op='+=')
        # verify the result and return if found return
        if result is not None and result is not False:
            self._value = result
            return self
        other = self._to_symbol(other)
        self._value = self.__add__(other)
        return self

    def __sub__(self, other: Any) -> 'Symbol':
        '''
        Replace occurrences of a value with another value in the Symbol.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.
        The method uses the @core.replace decorator to replace occurrences of the other value with an empty string in the Symbol.

        Args:
            other (Any): The value to replace in the Symbol.

        Returns:
            Symbol: The Symbol with occurrences of the other value replaced with an empty string.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value - other.value, op='-')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)

        self.__throw_error_on_nesy_engine_call(self.__sub__)

        @core.replace()
        def _func(_, text: str, replace: str, value: str):
            pass
        return self._to_symbol(_func(self, other, ''))

    def __rsub__(self, other: Any) -> 'Symbol':
        '''
        Subtracts the symbol value from another one and removes the substrings that match the symbol value.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.
        Using the core.replace decorator, this function creates a _func method to remove matching substrings.

        Args:
            other (Any): The string to subtract the symbol value from.

        Returns:
            Symbol: A new symbol with the result of the subtraction.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: other.value - self.value, op='-')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)

        self.__throw_error_on_nesy_engine_call(self.__rsub__)

        @core.replace()
        def _func(_, text: str, replace: str, value: str):
            pass
        other = self._to_symbol(other)
        return self._to_symbol(_func(other, self, ''))

    def __isub__(self, other: Any) -> 'Symbol':
        '''
        In-place subtraction of the symbol value by the other symbol value.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The symbol to subtract from the current symbol.

        Returns:
            Symbol: The current symbol with the updated value.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value - other.value, op='-=')
        # verify the result and return if found return
        if result is not None and result is not False:
            self._value = result
            return self
        val         = self.__sub__(other)
        self._value = val.value
        return self

    def __and__(self, other: Any) -> Any:
        '''
        Primary concatenation operator for Symbol objects if types are string. Concatenates them without a space in between.

        Otherwise, performs a logical AND operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.
        Uses the core.logic decorator with operator='and' to create a _func method for the AND operation.

        Args:
            other (Any): The string to perform the AND operation with the symbol value.

        Returns:
            Symbol: A new symbol with the result of the AND operation.
        '''
        # Special case for string concatenation with AND (no space)
        if isinstance(self.value, str) and isinstance(other, str) or \
            isinstance(self.value, str) and isinstance(other, self._symbol_type) and isinstance(other.value, str):
            other = self._to_symbol(other)
            return self._to_symbol(f'{self.value}{other.value}')

        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value and other.value, op='&')
        # verify the result and return if found return
        if result is not None and result is not False:
            return result

        self.__throw_error_on_nesy_engine_call(self.__and__)

        @core.logic(operator='and')
        def _func(_, a: str, b: str):
            pass
        return self._to_symbol(_func(self, other))

    def __rand__(self, other: Any) -> Any:
        '''
        Primary concatenation operator for Symbol objects if types are string. Concatenates them without a space in between.

        Otherwise, performs a logical AND operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.
        Uses the core.logic decorator with operator='and' to create a _func method for the AND operation.

        Args:
            other (Any): The string to perform the AND operation with the symbol value.

        Returns:
            Symbol: A new symbol with the result of the AND operation.
        '''
        # Special case for string concatenation with AND (no space)
        if isinstance(self.value, str) and isinstance(other, str) or \
            isinstance(self.value, str) and isinstance(other, self._symbol_type) and isinstance(other.value, str):
            other = self._to_symbol(other)
            return self._to_symbol(f'{other.value}{self.value}')

        other = self._to_symbol(other)
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other:  other.value and self.value, op='&')
        # verify the result and return if found return
        if result is not None and result is not False:
            return result

        self.__throw_error_on_nesy_engine_call(self.__rand__)

        @core.logic(operator='and')
        def _func(_, a: str, b: str):
            pass
        other = self._to_symbol(other)
        return self._to_symbol(_func(other, self))

    def __iand__(self, other: Any) -> Any:
        '''
        Primary concatenation operator for Symbol objects if types are string. Concatenates them without a space in between.

        Otherwise, performs a logical AND operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.
        Uses the core.logic decorator with operator='and' to create a _func method for the AND operation.

        Args:
            other (Any): The string to perform the AND operation with the symbol value.

        Returns:
            Symbol: A new symbol with the result of the AND operation.
        '''
        # Special case for string concatenation with AND (no space)
        if isinstance(self.value, str) and isinstance(other, str) or \
            isinstance(self.value, str) and isinstance(other, self._symbol_type) and isinstance(other.value, str):
            other       = self._to_symbol(other)
            self._value = f'{self.value}{other.value}'
            return self

        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value and other.value, op='&=')
        # verify the result and return if found return
        if result is not None and result is not False:
            self._value = result
            return self

        self.__throw_error_on_nesy_engine_call(self.__iand__)

        @core.logic(operator='and')
        def _func(_, a: str, b: str):
            pass
        self._value = _func(self, other)
        return self

    def __or__(self, other: Any) -> Any:
        '''
        Primary concatenation operator for Symbol objects if types are string. Concatenates them with a space in between.

        Otherwise, performs a logical OR operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.
        Uses the core.logic decorator with operator='or' to create a _func method for the OR operation.

        Args:
            other (Any): The string to perform the OR operation with the symbol value.

        Returns:
            Symbol: A new symbol with the result of the OR operation.
        '''
        # exclude the evaluation for the Aggregator class
        from ..collect.stats import Aggregator
        if isinstance(other, Aggregator):
            return NotImplemented

        # Special case for string concatenation with OR
        if isinstance(self.value, str) and isinstance(other, str) or \
            isinstance(self.value, str) and isinstance(other, self._symbol_type) and isinstance(other.value, str):
            other = self._to_symbol(other)
            return self._to_symbol(f'{self.value} {other.value}')

        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value or other.value, op='|')
        # verify the result and return if found return
        if result is not None and result is not False:
            return result

        self.__throw_error_on_nesy_engine_call(self.__or__)

        @core.logic(operator='or')
        def _func(_, a: str, b: str):
            pass
        return self._to_symbol(_func(self, other))

    def __ror__(self, other: Any) -> 'Symbol':
        '''
        Primary concatenation operator for Symbol objects if types are string. Concatenates them with a space in between.

        Otherwise, performs a logical OR operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to concatenate.

        Returns:
            Symbol: A new Symbol object with the concatenated value.
        '''
        # exclude the evaluation for the Aggregator class
        from ..collect.stats import Aggregator
        if isinstance(other, Aggregator):
            return NotImplemented

        if self.__disable_shortcut_matches__:
            # Special case for string concatenation with OR
            if isinstance(self.value, str) and isinstance(other, str) or \
                isinstance(self.value, str) and isinstance(other, self._symbol_type) and isinstance(other.value, str):
                other = self._to_symbol(other)
                return self._to_symbol(f'{other.value} {self.value}')

        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value | other.value, op='|')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)

        self.__throw_error_on_nesy_engine_call(self.__ror__)

        @core.logic(operator='or')
        def _func(a: str, b: str):
            pass
        other = self._to_symbol(other)
        return self._to_symbol(_func(other, self))

    def __ior__(self, other: Any) -> 'Symbol':
        '''
        Primary concatenation operator for Symbol objects if types are string. Concatenates them with a space in between.

        Otherwise, performs a logical OR operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to concatenate.

        Returns:
            Symbol: A new Symbol object with the concatenated value.
        '''
        # exclude the evaluation for the Aggregator class
        from ..collect.stats import Aggregator
        if isinstance(other, Aggregator):
            return NotImplemented

        if self.__disable_shortcut_matches__:
            # Special case for string concatenation with OR
            if isinstance(self.value, str) and isinstance(other, str) or \
                isinstance(self.value, str) and isinstance(other, self._symbol_type) and isinstance(other.value, str):
                other       = self._to_symbol(other)
                self._value = f'{self.value} {other.value}'
                return self

        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value | other.value, op='|=')
        # verify the result and return if found return
        if result is not None and result is not False:
            self._value = result
            return self
        result = self._to_symbol(str(self) + str(other))

        self.__throw_error_on_nesy_engine_call(self.__ior__)

        @core.logic(operator='or')
        def _func(_, a: str, b: str):
            pass
        self._value = _func(self, other)
        return self

    def __xor__(self, other: Any) -> Any:
        '''
        Performs a logical XOR operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.
        Uses the core.logic decorator with operator='xor' to create a _func method for the XOR operation.

        Args:
            other (Any): The string to perform the XOR operation with the symbol value.

        Returns:
            Symbol: A new symbol with the result of the XOR operation.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value ^ other.value, op='^')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)

        self.__throw_error_on_nesy_engine_call(self.__xor__)

        @core.logic(operator='xor')
        def _func(_, a: str, b: str):
            pass
        return self._to_symbol(_func(self, other))

    def __rxor__(self, other: Any) -> 'Symbol':
        '''
        Performs a logical XOR operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.
        Uses the core.logic decorator with operator='xor' to create a _func method for the XOR operation.

        Args:
            other (Any): The string to perform the XOR operation with the symbol value.

        Returns:
            Symbol: A new symbol with the result of the XOR operation.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: other.value ^ self.value, op='^')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)

        self.__throw_error_on_nesy_engine_call(self.__rxor__)

        @core.logic(operator='xor')
        def _func(_, a: str, b: str):
            pass
        other = self._to_symbol(other)
        return self._to_symbol(_func(other, self))

    def __ixor__(self, other: Any) -> 'Symbol':
        '''
        Performs a logical XOR operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.
        Uses the core.logic decorator with operator='xor' to create a _func method for the XOR operation.

        Args:
            other (Any): The string to perform the XOR operation with the symbol value.

        Returns:
            Symbol: A new symbol with the result of the XOR operation.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value ^ other.value, op='^=')
        # verify the result and return if found return
        if result is not None and result is not False:
            self._value = result
            return self

        self.__throw_error_on_nesy_engine_call(self.__ixor__)

        @core.logic(operator='xor')
        def _func(_, a: str, b: str):
            pass
        self._value = _func(self, other)
        return self

    def __matmul__(self, other: Any) -> 'Symbol':
        '''
        This method concatenates the string representation of two Symbol objects and returns a new Symbol with the concatenated result.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to concatenate.

        Returns:
            Symbol: A new Symbol object with the concatenated value.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value.__matmul__(other.value), op='@')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)
        raise NotImplementedError('Matrix multiplication not supported! Might change in the future.') from self._metadata._error

    def __rmatmul__(self, other: Any) -> 'Symbol':
        '''
        This method concatenates the string representation of two Symbol objects in a reversed order and returns a new Symbol with the concatenated result.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to concatenate.

        Returns:
            Symbol: A new Symbol object with the concatenated value.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other:  self.value.__rmatmul__(other.value), op='@')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)
        raise NotImplementedError('Matrix multiplication not supported! Might change in the future.') from self._metadata._error

    def __imatmul__(self, other: Any) -> 'Symbol':
        '''
        This method concatenates the string representation of two Symbol objects and assigns the concatenated result to the value of the current Symbol object.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to concatenate.

        Returns:
            Symbol: The current Symbol object with the concatenated value.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value.__imatmul__(other.value), op='@=')
        # verify the result and return if found return
        if result is not None and result is not False:
            self._value = result
            return self
        raise NotImplementedError('Matrix multiplication not supported! Might change in the future.') from self._metadata._error

    def __truediv__(self, other: Any) -> 'Symbol':
        '''
        Divides the symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value / other.value, op='/')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)
        return self._to_symbol(str(self).split(str(other)))

    def __rtruediv__(self, other: Any) -> 'Symbol':
        '''
        Divides the symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: other.value / self.value, op='/')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)
        raise NotImplementedError('Division operation not supported! Might change in the future.') from self._metadata._error

    def __itruediv__(self, other: Any) -> 'Symbol':
        '''
        Divides the symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value / other.value, op='/=')
        # verify the result and return if found return
        if result is not None and result is not False:
            self._value = result
            return self
        raise NotImplementedError('Division operation not supported! Might change in the future.') from self._metadata._error

    def __floordiv__(self, other: Any) -> 'Symbol':
        '''
        Floor divides the symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value // other.value, op='//')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)
        return self._to_symbol(str(self).split(str(other)))

    def __rfloordiv__(self, other: Any) -> 'Symbol':
        '''
        Floor divides the symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: other.value // self.value, op='//')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)
        raise NotImplementedError('Floor division operation not supported! Might change in the future.') from self._metadata._error

    def __ifloordiv__(self, other: Any) -> 'Symbol':
        '''
        Floor divides the symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value // other.value, op='//=')
        # verify the result and return if found return
        if result is not None and result is not False:
            self._value = result
            return self
        raise NotImplementedError('Floor division operation not supported! Might change in the future.') from self._metadata._error

    def __pow__(self, other: Any) -> 'Symbol':
        '''
        Power operation on symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value ** other.value, op='**')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)
        raise NotImplementedError('Power operation not supported! Might change in the future.') from self._metadata._error

    def __rpow__(self, other: Any) -> 'Symbol':
        '''
        Power operation on symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: other.value ** self.value, op='**')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)
        raise NotImplementedError('Power operation not supported! Might change in the future.') from self._metadata._error

    def __ipow__(self, other: Any) -> 'Symbol':
        '''
        Power operation on symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value ** other.value, op='**=')
        # verify the result and return if found return
        if result is not None and result is not False:
            self._value = result
            return self
        raise NotImplementedError('Power operation not supported! Might change in the future.') from self._metadata._error

    def __mod__(self, other: Any) -> 'Symbol':
        '''
        Modulo operation on symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value % other.value, op='%')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)
        raise NotImplementedError('Modulo operation not supported! Might change in the future.') from self._metadata._error

    def __rmod__(self, other: Any) -> 'Symbol':
        '''
        Modulo operation on symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: other.value % self.value, op='%')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)
        raise NotImplementedError('Modulo operation not supported! Might change in the future.') from self._metadata._error

    def __imod__(self, other: Any) -> 'Symbol':
        '''
        Modulo operation on symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value % other.value, op='%=')
        # verify the result and return if found return
        if result is not None and result is not False:
            self._value = result
            return self
        raise NotImplementedError('Modulo operation not supported! Might change in the future.') from self._metadata._error

    def __mul__(self, other: Any) -> 'Symbol':
        '''
        Multiply operation on symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value * other.value, op='*')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)
        raise NotImplementedError('Multiply operation not supported! Might change in the future.') from self._metadata._error

    def __rmul__(self, other: Any) -> 'Symbol':
        '''
        Multiply operation on symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: other.value * self.value, op='*')
        # verify the result and return if found return
        if result is not None and result is not False:
            return self._to_symbol(result)
        raise NotImplementedError('Multiply operation not supported! Might change in the future.') from self._metadata._error

    def __imul__(self, other: Any) -> 'Symbol':
        '''
        Multiply operation on symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        '''
        # First verify for specific type support
        result = self.__try_type_specific_func(other, lambda self, other: self.value * other.value, op='*=')
        # verify the result and return if found return
        if result is not None and result is not False:
            self._value = result
            return self
        raise NotImplementedError('Multiply operation not supported! Might change in the future.') from self._metadata._error


class CastingPrimitives(Primitive):
    '''
    This mixin contains functionalities related to casting symbols.
    '''
    def cast(self, as_type: Type) -> Any:
        '''
        Cast the Symbol's value to a specific type.

        Args:
            as_type (Type): The type to cast the Symbol's value to.

        Returns:
            The Symbol's value casted to the specified type.
        '''
        return as_type(self.value)

    def to(self, as_type: Type) -> Any:
        '''
        Cast the Symbol's value to a specific type.

        Args:
            as_type (Type): The type to cast the Symbol's value to.

        Returns:
            The Symbol's value casted to the specified type.
        '''
        return self.cast(as_type)

    def ast(self) -> Any:
        '''
        Converts the string representation of the Symbol's value to an abstract syntax tree using 'ast.literal_eval'.

        Returns:
            The abstract syntax tree representation of the Symbol's value.
        '''
        return ast.literal_eval(str(self.value))

    def str(self) -> str:
        '''
        Get the string representation of the Symbol's value.

        Returns:
            str: The string representation of the Symbol's value.
        '''
        return str(self.value)

    def int(self) -> int:
        '''
        Get the integer representation of the Symbol's value.

        Returns:
            int: The integer representation of the Symbol's value.
        '''
        return int(self.value)

    def float(self) -> float:

        '''
        Get the float representation of the Symbol's value.

        Returns:
            float: The float representation of the Symbol's value.
        '''
        return float(self.value)

    def bool(self) -> bool:
        '''
        Get the boolean representation of the Symbol's value.

        Returns:
            bool: The boolean representation of the Symbol's value.
        '''
        return bool(self.value)


class IterationPrimitives(Primitive):
    '''
    This mixin contains functions that perform iteration operations on symbols or symbol values.
    The functions in this mixin are bound to the 'neurosymbolic' engine for evaluation.
    '''
    def __getitem__(self, key: Union[str, int, slice]) -> 'Symbol':
        '''
        Get the item of the Symbol value with the specified key or index.
        If the Symbol value is a list, tuple, or numpy array, the key can be an integer or slice.
        If the Symbol value is a dictionary, the key can be a string or an integer.
        If the direct item retrieval fails, the method falls back to using the @core.getitem decorator, which retrieves and returns the item using core functions.

        Args:
            key (Union[str, int, slice]): The key or index for the item in the Symbol value.

        Returns:
            Symbol: The item of the Symbol value with the specified key or index.

        Raises:
            KeyError: If the key or index is not found in the Symbol value.
        '''
        try:
            if  (isinstance(key, int) or isinstance(key, slice)) and \
                (isinstance(self.value, list) or \
                 isinstance(self.value, tuple) or \
                 isinstance(self.value, np.ndarray)):
                return self.value[key]
            elif (isinstance(key, str) or isinstance(key, int)) and \
                  isinstance(self.value, dict):
                return self.value[key]
        except KeyError:
            pass
        # verify if fuzzy matches are enabled in general
        if not self.__nesy_iteration_primitives__ or Primitive._is_iterable(self.value):
            raise KeyError(f'Key {key} not found in {self.value}')

        @core.getitem()
        def _func(_, index: str):
            pass

        return self._to_symbol(_func(self, key))

    def __setitem__(self, key: Union[str, int, slice], value: Any) -> None:
        '''
        Set the item of the Symbol value with the specified key or index to the given value.
        If the Symbol value is a list, tuple, or numpy array, the key can be an integer or slice.
        If the Symbol value is a dictionary, the key can be a string or an integer.
        If the direct item setting fails, the method falls back to using the @core.setitem decorator, which sets the item using core functions.

        Args:
            key (Union[str, int, slice]): The key or index for the item in the Symbol value.
            value: The value to set the item to.

        Raises:
            KeyError: If the key or index is not found in the Symbol value.
        '''
        try:
            if (isinstance(key, int) or isinstance(key, slice)) and (isinstance(self.value, list) or isinstance(self.value, tuple) or isinstance(self.value, np.ndarray)):
                self.value[key] = value
                return
            elif (isinstance(key, str) or isinstance(key, int)) and isinstance(self.value, dict):
                self.value[key] = value
                return
        except KeyError:
            raise KeyError(f'Key {key} not found in {self.value}')

        if not self.__nesy_iteration_primitives__ or Primitive._is_iterable(self.value):
            raise KeyError(f'Key {key} not found in {self.value}')

        @core.setitem()
        def _func(_, index: str, value: str):
            pass

        self._value = self._to_symbol(_func(self, key, value)).value

    def __delitem__(self, key: Union[str, int]) -> None:
        '''
        Delete the item of the Symbol value with the specified key or index.
        If the Symbol value is a dictionary, the key can be a string or an integer.
        If the direct item deletion fails, the method falls back to using the @core.delitem decorator, which deletes the item using core functions.

        Args:
            key (Union[str, int]): The key for the item in the Symbol value.

        Raises:
            KeyError: If the key or index is not found in the Symbol value.
        '''
        try:
            if (isinstance(key, str) or isinstance(key, int)) and isinstance(self.value, dict):
                del self.value[key]
                return
        except KeyError:
            raise KeyError(f'Key {key} not found in {self.value}')

        if not self.__nesy_iteration_primitives__ or Primitive._is_iterable(self.value):
            raise KeyError(f'Key {key} not found in {self.value}')

        @core.delitem()
        def _func(_, index: str):
            pass

        self._value = self._to_symbol(_func(self, key)).value


class ValueHandlingPrimitives(Primitive):
    '''
    This mixin includes functions responsible for handling symbol values - tokenization, type retrieval, value casting, indexing, etc.
    Future functions might include different methods of processing or manipulating the values of symbols, working with metadata of values, etc.
    '''
    @property
    def size(self) -> int:
        '''
        Get the size of the container of the Symbol's value.

        Returns:
            int: The size of the container of the Symbol's value.
        '''
        return len(self.value)

    @property
    def tokens(self) -> int:
        '''
        Tokenize the Symbol's value using the tokenizer method.
        The tokenizer method is bound to the 'neurosymbolic' engine using the @decorator.bind() decorator.

        Returns:
            int: The tokenized value of the Symbol.
        '''
        return self.tokenizer().encode(str(self))

    @core_ext.bind(engine='neurosymbolic', property='tokenizer')
    def tokenizer(self) -> Callable:
        '''
        The tokenizer method.
        This method is bound to the 'neurosymbolic' engine using the @decorator.bind() decorator.

        Returns:
            Callable: The tokenizer.
        '''
        pass

    @property
    def type(self):
        '''
        Get the type of the Symbol.

        Returns:
            type: The type of the Symbol.
        '''
        return type(self)

    @property
    def value_type(self):
        '''
        Get the type of the Symbol's value.

        Returns:
            type: The type of the Symbol's value.
        '''
        return type(self.value)

    def index(self, item: str, **kwargs) -> 'Symbol':
        '''
        Returns the index of a specified item in the symbol value.
        Uses the core.getitem decorator to create a _func method that finds the index of the item.

        Args:
            item (str): The item to find the index of within the symbol value.

        Returns:
            Symbol: A new symbol with the index of the specified item.
        '''
        @core.getitem(**kwargs)
        def _func(_, item: str) -> int:
            pass
        return self._to_symbol(_func(self, item))


class StringHelperPrimitives(Primitive):
    '''
    This mixin contains functions that provide additional help for symbols or their values.
    '''
    def split(self, delimiter: str, **kwargs) -> 'Symbol':
        '''
        Splits the symbol value by a specified delimiter.
        Uses the core.split decorator to create a _func method that splits the symbol value by the specified delimiter.

        Args:
            delimiter (str): The delimiter to split the symbol value by.

        Returns:
            Symbol: A new symbol with the split value.
        '''
        assert isinstance(delimiter, str),  f'delimiter must be a string, got {type(delimiter)}'
        assert isinstance(self.value, str), f'self.value must be a string, got {type(self.value)}'
        symbols = self.symbols(*self.value.split(delimiter))
        return symbols

    def join(self, delimiter: str = ' ', **kwargs) -> 'Symbol':
        '''
        Joins the symbol value with a specified delimiter.

        Args:
            delimiter (str, optional): The delimiter to join the symbol value with. Defaults to ' '.

        Returns:
            Symbol: A new symbol with the joined str value.
        '''
        if isinstance(self.value, str):
            # Special case for string joining to forward the original join method
            return self.value.join(delimiter)

        assert isinstance(self.value, Iterable),  f'value must be an iterable, got {type(self.value)}'
        return self._to_symbol(delimiter.join(self.value))

    def startswith(self, prefix: str, **kwargs) -> bool:
        '''
        Checks if the symbol value starts with a specified prefix.
        Uses the core.startswith decorator to create a _func method that checks if the symbol value starts with the specified prefix.

        Args:
            prefix (str): The prefix to check if the symbol value starts with.

        Returns:
            bool: True if the symbol value starts with the specified prefix, otherwise False.
        '''
        assert isinstance(prefix, str),  f'prefix must be a string, got {type(prefix)}'
        assert isinstance(self.value, str), f'self.value must be a string, got {type(self.value)}'
        return self.value.startswith(prefix)

    def endswith(self, suffix: str, **kwargs) -> bool:
        '''
        Checks if the symbol value ends with a specified suffix.
        Uses the core.endswith decorator to create a _func method that checks if the symbol value ends with the specified suffix.

        Args:
            suffix (str): The suffix to check if the symbol value ends with.

        Returns:
            bool: True if the symbol value ends with the specified suffix, otherwise False.
        '''
        assert isinstance(suffix, str),  f'suffix must be a string, got {type(suffix)}'
        assert isinstance(self.value, str), f'self.value must be a string, got {type(self.value)}'
        return self.value.endswith(suffix)

class ComparisonPrimitives(Primitive):
    '''
    This mixin is dedicated to functions that perform more complex comparison operations between symbols or symbol values.
    This usually involves additional context, which the builtin overrode (e.g. __eq__) functions lack.
    '''
    def equals(self, string: str, context: str = 'contextually', **kwargs) -> 'Symbol':
        '''
        Checks if the symbol value is equal to another string.
        Uses the core.equals decorator to create a _func method that checks for equality in a specific context.

        Args:
            string (str): The string to compare with the symbol value.
            context (str, optional): The context in which to compare the strings. Defaults to 'contextually'.

        Returns:
            Symbol: A new symbol indicating whether the two strings are equal or not.
        '''
        @core.equals(context=context, **kwargs)
        def _func(_, string: str) -> bool:
            pass

        return self._to_symbol(_func(self, string))

    def contains(self, element: Any, **kwargs) -> bool:
        '''
        Uses the @core.contains decorator, checks whether the symbol's value contains the element.

        Args:
            element (Any): The element to be checked for containment.
            **kwargs: Additional keyword arguments to pass to the core.contains decorator.

        Returns:
            bool: True if the symbol's value contains the element, False otherwise.
        '''
        @core.contains(**kwargs)
        def _func(_, other) -> bool:
            pass

        return _func(self, element)

    def isinstanceof(self, query: str, **kwargs) -> bool:
        '''
        Check if the current Symbol is an instance of a specific type.

        Args:
            query (str): The type to check if the Symbol is an instance of.
            **kwargs: Any additional kwargs for @core.isinstanceof() decorator.

        Returns:
            bool: True if the current Symbol is an instance of the specified type, otherwise False.
        '''
        @core.isinstanceof()
        def _func(_, query: str, **kwargs) -> bool:
            pass

        return _func(self, query, **kwargs)


class ExpressionHandlingPrimitives(Primitive):
    '''
    This mixin consists of functions that handle symbolic expressions - evaluations, parsing, computation and more.
    Future functionalities in this mixin might include operations to manipulate expressions, more complex evaluation techniques, etc.
    '''
    def interpret(self, expr: Optional[str] = None, **kwargs) -> 'Symbol':
        '''
        Evaluates a symbolic expression using the provided engine.
        Uses the core.expression decorator to create a _func method that evaluates the given expression.

        Args:
            expr (Optional[str]): The expression to evaluate. Defaults to the symbol value.

        Returns:
            Symbol: A new symbol with the result of the expression evaluation.
        '''
        if expr is None:
            expr = self.value

        @core.interpret(**kwargs)
        def _func(_, expr: str):
            pass

        return self._to_symbol(_func(self, expr))


class DataHandlingPrimitives(Primitive):
    '''
    This mixin houses functions that clean, summarize and outline symbols or their values.
    Future implementations in this mixin may include various other cleaning and summarization techniques, error detection/correction in symbols, complex filtering, bulk modifications, or other types of condition-based manipulations on symbols, etc.
    '''
    def clean(self, **kwargs) -> 'Symbol':
        '''
        Cleans the symbol value.
        Uses the core.clean decorator to create a _func method that cleans the symbol value.

        Returns:
            Symbol: A new symbol with the cleaned value.
        '''
        @core.clean(**kwargs)
        def _func(_) -> str:
            pass

        return self._to_symbol(_func(self))

    def summarize(self, context: Optional[str] = None, **kwargs) -> 'Symbol':
        '''
        Summarizes the symbol value.
        Uses the core.summarize decorator with an optional context to create a _func method that summarizes the symbol value.

        Args:
            context (Optional[str]): The context to be used for summarization. Defaults to None.

        Returns:
            Symbol: A new symbol with the summarized value.
        '''
        @core.summarize(context=context, **kwargs)
        def _func(_) -> str:
            pass

        return self._to_symbol(_func(self))

    def outline(self, **kwargs) -> 'Symbol':
        '''
        Creates an outline of the symbol value.
        Uses the core.outline decorator to create a _func method that generates an outline of the symbol value.

        Returns:
            Symbol: A new symbol with the outline of the value.
        '''
        @core.outline(**kwargs)
        def _func(_) -> str:
            pass

        return self._to_symbol(_func(self))

    def filter(self, criteria: str, include: Optional[bool] = False, **kwargs) -> 'Symbol':
        '''
        Filters the symbol value based on a specified criteria.
        Uses the core.filtering decorator with the provided criteria and include flag to create a _func method to filter the symbol value.

        Args:
            criteria (str): The criteria to filter the symbol value by.
            include (Optional[bool]): Whether to include or exclude items based on the criteria. Defaults to False.

        Returns:
            Symbol: A new symbol with the filtered value.
        '''
        @core.filtering(criteria=criteria, include=include, **kwargs)
        def _func(_) -> str:
            pass

        return self._to_symbol(_func(self))

    def modify(self, changes: str, **kwargs) -> 'Symbol':
        '''
        Modifies the symbol value based on the specified changes.
        Uses the core.modify decorator with the provided changes to create a _func method to modify the symbol value.

        Args:
            changes (str): The changes to apply to the symbol value.

        Returns:
            Symbol: A new symbol with the modified value.
        '''
        @core.modify(changes=changes, **kwargs)
        def _func(_) -> str:
            pass

        return self._to_symbol(_func(self))

    def replace(self, old: str, new: str, **kwargs) -> 'Symbol':
        '''
        Replaces one value in the symbol value with another.
        Uses the core.replace decorator to create a _func method that replaces the values in the symbol value.

        Args:
            old (str): The value to be replaced in the symbol value.
            new (str): The value to replace the existing value with.

        Returns:
            Symbol: A new symbol with the replaced value.
        '''
        @core.replace(**kwargs)
        def _func(_, old: str, new: str):
            pass

        return self._to_symbol(_func(self, old, new))

    def remove(self, information: str, **kwargs) -> 'Symbol':
        '''
        Removes a specified piece of information from the symbol value.
        Uses the core.replace decorator to create a _func method that removes the specified information.

        Args:
            information (str): The information to remove from the symbol value.

        Returns:
            Symbol: A new symbol with the removed information.
        '''
        @core.replace(**kwargs)
        def _func(_, text: str, replace: str, value: str):
            pass

        return self._to_symbol(_func(self, information, ''))

    def include(self, information: str, **kwargs) -> 'Symbol':
        '''
        Includes a specified piece of information in the symbol value.
        Uses the core.include decorator to create a _func method that includes the specified information.

        Args:
            information (str): The information to include in the symbol value.

        Returns:
            Symbol: A new symbol with the included information.
        '''
        @core.include(**kwargs)
        def _func(_, information: str):
            pass

        return self._to_symbol(_func(self, information))

    def combine(self, information: str, **kwargs) -> 'Symbol':
        '''
        Combines the current symbol value with another string.
        Uses the core.combine decorator to create a _func method that combines the symbol value with another string.

        Args:
            information (str): The information to combine with the symbol value.

        Returns:
            Symbol: A new symbol with the combined value.
        '''
        @core.combine(**kwargs)
        def _func(_, a: str, b: str):
            pass

        return self._to_symbol(_func(self, information))


class UniquenessPrimitives(Primitive):
    '''
    This mixin includes functions that work with unique aspects of symbol values, like extracting unique information or composing new unique symbols.
    Future functionalities might include finding duplicate information, defining levels of uniqueness, etc.
    '''
    def unique(self, keys: Optional[List[str]] = [], **kwargs) -> 'Symbol':
        '''
        Extracts unique information from the symbol value, using provided keys.
        Uses the core.unique decorator with a list of keys to create a _func method that extracts unique data from the symbol value.

        Args:
            keys (Optional[List[str]]): The list of keys to extract unique data. Defaults to [].

        Returns:
            Symbol: A new symbol with the unique information.
        '''
        @core.unique(keys=keys, **kwargs)
        def _func(_) -> str:
            pass

        return self._to_symbol(_func(self))

    def compose(self, **kwargs) -> 'Symbol':
        '''
        Composes a text based on the symbol value.
        Uses the core.compose decorator to create a _func method that composes a text using the symbol value.

        Returns:
            Symbol: A new symbol with the composed text.
        '''
        @core.compose(**kwargs)
        def _func(_) -> str:
            pass

        return self._to_symbol(_func(self))


class PatternMatchingPrimitives(Primitive):
    '''
    This mixin houses functions that deal with ranking symbols, extracting details based on patterns, and correcting symbols.
    It will house future functionalities that involve sorting, complex pattern detections, advanced correction techniques etc.
    '''
    def rank(self, measure: Optional[str] = 'alphanumeric', order: Optional[str] = 'desc', **kwargs) -> 'Symbol':
        '''
        Ranks the symbol value based on a measure and a sort order.
        Uses the core.rank decorator with the specified measure and order to create a _func method that ranks the symbol value.

        Args:
            measure (Optional[str]): The measure to rank the symbol value by. Defaults to 'alphanumeric'.
            order (Optional[str]): The sort order for ranking. Defaults to 'desc'.
            **kwargs: Additional keyword arguments to pass to the core.rank decorator.

        Returns:
            Symbol: A new symbol with the ranked value.
        '''
        @core.rank(order=order, **kwargs)
        def _func(_, measure: str) -> str:
            pass

        return self._to_symbol(_func(self, measure))

    def extract(self, pattern: str, **kwargs) -> 'Symbol':
        '''
        Extracts data from the symbol value based on a pattern.
        Uses the core.extract decorator with the specified pattern to create a _func method that extracts data from the symbol value.

        Args:
            pattern (str): The pattern to use for data extraction.

        Returns:
            Symbol: A new symbol with the extracted data.
        '''
        @core.extract(**kwargs)
        def _func(_, pattern: str) -> str:
            pass

        return self._to_symbol(_func(self, pattern))

    def correct(self, context: str, **kwargs) -> 'Symbol':
        '''
        Corrects the symbol value based on a specified context.
        Uses the @core.correct decorator, corrects the value of the symbol based on the given context.

        Args:
            context (str): The context used to correct the value of the symbol.
            **kwargs: Additional keyword arguments to pass to the core.correct decorator.

        Returns:
            Symbol: The corrected value as a Symbol.
        '''
        @core.correct(context=context, **kwargs)
        def _func(_) -> str:
            pass

        return self._to_symbol(_func(self))

    def translate(self, language: Optional[str] = 'English', **kwargs) -> 'Symbol':
        '''
        Translates the symbol value to the specified language.
        Uses the @core.translate decorator to translate the symbol's value to the specified language.

        Args:
            language (Optional[str]): The language to translate the value to. Defaults to 'English'.
            **kwargs: Additional keyword arguments to pass to the core.translate decorator.

        Returns:
            Symbol: The translated value as a Symbol.
        '''
        @core.translate(language=language, **kwargs)
        def _func(_) -> str:
            pass

        return self._to_symbol(_func(self))

    def choice(self, cases: List[str], default: str, **kwargs) -> 'Symbol':
        '''
        Chooses one of the cases based on the symbol value.
        Uses the @core.case decorator, selects one of the cases based on the symbol's value.

        Args:
            cases (List[str]): The list of possible cases.
            default (str): The default case if none of the cases match the symbol's value.
            **kwargs: Additional keyword arguments to pass to the core.case decorator.

        Returns:
            Symbol: The chosen case as a Symbol.
        '''
        @core.case(enum=cases, default=default, **kwargs)
        def _func(_) -> str:
            pass

        return self._to_symbol(_func(self))


class QueryHandlingPrimitives(Primitive):
    '''
    This mixin helps in transforming, preparing, and executing queries, and it is designed to be extendable as new ways of handling queries are developed.
    Future methods could potentially include query optimization, enhanced query formatting, multi-level query execution, query error handling, etc.
    '''
    def query(self, context: str, prompt: Optional[str] = None, examples: Optional[List[Prompt]] = None, **kwargs) -> 'Symbol':
        '''
        Queries the symbol value based on a specified context.
        Uses the @core.query decorator, queries based on the context, prompt, and examples.

        Args:
            context (str): The context used for the query.
            prompt (Optional[str]): The prompt for the query. Defaults to None.
            examples (Optional[List[Prompt]]): The examples for the query. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the core.query decorator.

        Returns:
            Symbol: The result of the query as a Symbol.
        '''
        @core.query(context=context, prompt=prompt, examples=examples, **kwargs)
        def _func(_) -> str:
            pass

        return self._to_symbol(_func(self))

    def convert(self, format: str, **kwargs) -> 'Symbol':
        '''
        Converts the symbol value to the specified format.
        Uses the @core.convert decorator, converts the symbol's value to the specified format.

        Args:
            format (str): The format to convert the value to.
            **kwargs: Additional keyword arguments to pass to the core.convert decorator.

        Returns:
            Symbol: The converted value as a Symbol.
        '''
        @core.convert(format=format, **kwargs)
        def _func(_) -> str:
            pass

        return self._to_symbol(_func(self))

    def transcribe(self, modify: str, **kwargs) -> 'Symbol':
        '''
        Transcribes the symbol value based on a specified modification.
        Uses the @core.transcribe decorator, modifies the symbol's value based on the modify parameter.

        Args:
            modify (str): The modification to be applied to the value.
            **kwargs: Additional keyword arguments to pass to the core.transcribe decorator.

        Returns:
            Symbol: The modified value as a Symbol.
        '''
        @core.transcribe(modify=modify, **kwargs)
        def _func(_) -> str:
            pass

        return self._to_symbol(_func(self))


class ExecutionControlPrimitives(Primitive):
    '''
    This mixin represents the core methods for dealing with symbol execution.
    Possible future methods could potentially include async execution, pipeline chaining, execution profiling, improved error handling, version management, embedding more complex execution control structures etc.
    '''
    def analyze(self, exception: Exception, query: Optional[str] = '', **kwargs) -> 'Symbol':
        '''Uses the @core.analyze decorator, analyzes an exception and returns a symbol.

        Args:
            exception (Exception): The exception to be analyzed.
            query (Optional[str]): An additional query to provide context during analysis. Defaults to ''.
            **kwargs: Additional keyword arguments to pass to the core.analyze decorator.

        Returns:
            Symbol: The analyzed result as a Symbol.
        '''
        @core.analyze(exception=exception, query=query, **kwargs)
        def _func(_) -> str:
            pass

        return self._to_symbol(_func(self))

    def execute(self, **kwargs) -> 'Symbol':
        '''
        Executes the symbol's expression using the @core.execute decorator.

        Args:
            **kwargs: Additional keyword arguments to pass to the core.execute decorator.

        Returns:
            Symbol: The result of the executed expression as a Symbol.
        '''
        @core.execute(**kwargs)
        def _func(_):
            pass

        return _func(self)

    def fexecute(self, **kwargs) -> 'Symbol':
        '''
        Executes the symbol's expression using the fallback execute method (ftry).

        Args:
            **kwargs: Additional keyword arguments to pass to the core.execute decorator.

        Returns:
            Symbol: The result of the executed expression as a Symbol.
        '''
        def _func(sym: 'Symbol', **kargs):
            return sym.execute(**kargs)

        return self.ftry(_func, **kwargs)

    def simulate(self, **kwargs) -> 'Symbol':
        '''
        Uses the @core.simulate decorator, simulates the value of the symbol. Used for hypothesis testing or code simulation.

        Args:
            **kwargs: Additional keyword arguments to pass to the core.simulate decorator.

        Returns:
            Symbol: The simulated value as a Symbol.
        '''
        @core.simulate(**kwargs)
        def _func(_):
            pass

        return self._to_symbol(_func(self))

    def sufficient(self, query: str, **kwargs) -> 'Symbol':
        '''
        Uses the @core.sufficient decorator and checks if the symbol's value is sufficient based on the query.

        Args:
            query (str): The query to verify if the symbol's value is sufficient.
            **kwargs: Additional keyword arguments to pass to the core.sufficient decorator.

        Returns:
            Symbol: The sufficiency result as a Symbol.
        '''
        @core.sufficient(query=query, **kwargs)
        def _func(_) -> bool:
            pass

        return self._to_symbol(_func(self))

    def list(self, condition: str, **kwargs) -> 'Symbol':  #@TODO: can't filter directly handle this case?
        '''
        Uses the @core.listing decorator and lists elements based on the condition.

        Args:
            condition (str): The condition to filter the elements in the list.
            **kwargs: Additional keyword arguments to pass to the core.listing decorator.

        Returns:
            Symbol: The filtered list as a Symbol.
        '''
        @core.listing(condition=condition, **kwargs)
        def _func(_) -> list:
            pass

        return self._to_symbol(_func(self))

    def foreach(self, condition: str, apply: str, **kwargs) -> 'Symbol':
        '''
        Uses the @core.foreach decorator, iterates through the symbol's value, and applies the provided functionality.

        Args:
            condition (str): The condition to filter the elements in the list.
            apply (str): The functionality to be applied to each element in the list.
            **kwargs: Additional keyword arguments to pass to the core.foreach decorator.

        Returns:
            Symbol: The result of the iterative application of the function as a Symbol.
        '''
        @core.foreach(condition=condition, apply=apply, **kwargs)
        def _func(_):
            pass

        return self._to_symbol(_func(self))

    def stream(self, expr: 'Expression', token_ratio: Optional[float] = 0.6, **kwargs) -> 'Symbol':
        '''
        Streams the Symbol's value through an Expression object.
        This method divides the Symbol's value into chunks and processes each chunk through the given Expression object.
        It is useful for processing large strings in smaller parts.

        Args:
            expr (Expression): The Expression object to evaluate the Symbol's chunks.
            token_ratio (Optional[float]): The ratio between input-output tokens for calculating max_chars. Defaults to 0.6.
            **kwargs: Additional keyword arguments for the given Expression.

        Returns:
            Symbol: A Symbol object containing the evaluated chunks.

        Raises:
            ValueError: If the Expression object exceeds the maximum allowed tokens.
        '''
        @core_ext.bind(engine='neurosymbolic', property='max_tokens')
        def _max_tokens(_): pass

        max_ctxt_tokens = int(_max_tokens(self) * token_ratio)
        prev = expr(self, preview=True, **kwargs)
        prev = str(prev)

        if len(prev) > _max_tokens(self):
            n_splits = (len(prev) // max_ctxt_tokens) + 1

            for i in range(n_splits):
                tokens_sliced = self.tokens[i * max_ctxt_tokens: (i + 1) * max_ctxt_tokens]
                r = self._to_symbol(self.tokenizer().decode(tokens_sliced))

                yield expr(r, **kwargs)

        else:
            yield expr(self, **kwargs)

    def ftry(self, expr: 'Expression', retries: Optional[int] = 1, **kwargs) -> 'Symbol':
        # TODO: find a way to pass on the constraints and behavior from the self.expr to the corrected code
        '''
        Tries to evaluate a Symbol using a given Expression.
        This method evaluates a Symbol using a given Expression.
        If it fails, it retries the evaluation a specified number of times.

        Args:
            expr (Expression): The Expression object to evaluate the Symbol.
            retries (Optional[int]): The number of retries if the evaluation fails. Defaults to 1.
            **kwargs: Additional keyword arguments for the given Expression.

        Returns:
            Symbol: A Symbol object with the evaluated result.

        Raises:
            Exception: If the evaluation fails after all retries.
        '''
        prompt = {'out_msg': ''}

        def output_handler(input_):
            prompt['out_msg'] = input_

        kwargs['output_handler'] = output_handler
        retry_cnt: int = 0
        code = self # original input

        if hasattr(expr, 'prompt'):
            prompt['prompt_instruction'] = expr.prompt

        sym = self # used for getting passed from one iteration to the next
        while True:
            try:
                sym = expr(sym, **kwargs) # run the expression
                retry_cnt = 0

                return sym

            except Exception as e:
                retry_cnt += 1
                if retry_cnt > retries:
                    raise e
                else:
                    # analyze the error
                    payload = f'[ORIGINAL_USER_PROMPT]\n{prompt["prompt_instruction"]}\n\n' if 'prompt_instruction' in prompt else ''
                    payload = payload + f'[ORIGINAL_USER_DATA]\n{code}\n\n[ORIGINAL_GENERATED_OUTPUT]\n{prompt["out_msg"]}'
                    probe   = sym.analyze(query="What is the issue in this expression?", payload=payload, exception=e)
                    # attempt to correct the error
                    payload = f'[ORIGINAL_USER_PROMPT]\n{prompt["prompt_instruction"]}\n\n' if 'prompt_instruction' in prompt else ''
                    payload = payload + f'[ANALYSIS]\n{probe}\n\n'
                    context = f'Try to correct the error of the original user request based on the analysis above: \n [GENERATED_OUTPUT]\n{prompt["out_msg"]}\n\n'
                    constraints = expr.constraints if hasattr(expr, 'constraints') else []

                    if hasattr(expr, 'post_processor'):
                        post_processor = expr.post_processor
                        sym = code.correct(
                            context=context,
                            exception=e,
                            payload=payload,
                            constraints=constraints,
                            post_processor=post_processor
                        )
                    else:
                        sym = code.correct(
                            context=context,
                            exception=e,
                            payload=payload,
                            constraints=constraints
                        )


class DictHandlingPrimitives(Primitive):
    '''
    This mixin hosts functions that deal with dictionary operations on symbol values.
    It can be extended in the future with more advanced dictionary methods and operations.
    '''
    def dict(self, context: str, **kwargs) -> 'Symbol':
        '''
        Maps related content together under a common abstract topic as a dictionary of the Symbol value.
        This method uses the @core.dictionary decorator to apply the given context to the Symbol.
        It is useful for applying additional context to the symbol.

        Args:
            context (str): The context to apply to the Symbol.
            **kwargs: Additional keyword arguments for the @core.dictionary decorator.

        Returns:
            Symbol: A Symbol object with a dictionary applied.
        '''
        @core.dictionary(context=context, **kwargs)
        def _func(_):
            pass

        return self._to_symbol(_func(self))

    def map(self, **kwargs) -> 'Symbol':
        '''
        Transforms the keys of the dictionary value of a Symbol object to be unique.
        This function asserts that the Symbol's value is a dictionary and creates a new dictionary with the same values but unique keys.
        It is useful for ensuring that there are no duplicate keys in a dictionary.

        Args:
            **kwargs: Additional keyword arguments for the `unique` method.

        Returns:
            Symbol: A Symbol object with its value being the transformed dictionary with unique keys.

        Raises:
            AssertionError: If the Symbol's value is not a dictionary.
        '''
        assert isinstance(self.value, dict), 'Map can only be applied to a dictionary'

        map_ = {}
        keys = []
        for v in self.value.values():
            k = self._to_symbol(v).unique(keys, **kwargs)
            keys.append(k.value)
            map_[k.value] = v

        return self._to_symbol(map_)


class TemplateStylingPrimitives(Primitive):
    '''
    This mixin includes functionalities for stylizing symbols and applying templates.
    Future functionalities might include a variety of new stylizing methods, application of more complex templates, etc.
    '''
    def template(that, template: str, placeholder: Optional[str] = '{{placeholder}}', **kwargs) -> 'Symbol':
        '''
        Applies a template to the Symbol.
        This method uses the @core.template decorator to apply the given template and placeholder to the Symbol.
        It is useful for providing structure to the Symbol's value.

        Args:
            template (str): The template to apply to the Symbol.
            placeholder (Optional[str]): The placeholder in the template to be replaced with the Symbol's value. Defaults to '{{placeholder}}'.
            **kwargs: Additional keyword arguments for the @core.template decorator.

        Returns:
            Symbol: A Symbol object with a template applied.
        '''
        def _func(self):
            res = template.replace(placeholder, str(self))
            return that._to_symbol(res)

        return _func(that)

    def style(self, description: str, libraries: Optional[List] = [], **kwargs) -> 'Symbol':
        '''
        Applies a style to the Symbol.
        This method uses the @core.style decorator to apply the given style description, libraries, and placeholder to the Symbol.
        It is useful for providing structure and style to the Symbol's value.

        Args:
            description (str): The description of the style to apply.
            libraries (Optional[List]): A list of libraries that may be included in the style. Defaults to an empty list.
            **kwargs: Additional keyword arguments for the @core.style decorator.

        Returns:
            Symbol: A Symbol object with the style applied.
        '''
        @core.style(description=description, libraries=libraries, **kwargs)
        def _func(_):
            pass

        return self._to_symbol(_func(self))


class DataClusteringPrimitives(Primitive):
    '''
    This mixin contains functionalities that deal with clustering symbol values or generating embeddings.
    New functionalities in this mixin might include different types of clustering and embedding methods, dimensionality reduction techniques, etc.
    '''
    def cluster(self, **kwargs) -> 'Symbol':
        '''
        Creates a cluster from the Symbol's value.
        This method uses the @core.cluster decorator to create a cluster from the Symbol's value.
        It is useful for grouping values in the Symbol.

        Args:
            **kwargs: Additional keyword arguments for the @core.cluster decorator.

        Returns:
            Symbol: A Symbol object with its value clustered.
        '''
        @core.cluster(entries=self.value, **kwargs)
        def _func(_):
            pass

        return self._to_symbol(_func(self))


class EmbeddingPrimitives(Primitive):
    '''
    This mixin contains functionalities that deal with embedding symbol values.
    New functionalities in this mixin might include different types of embedding methods, similarity and distance measures etc.
    '''
    def embed(self, **kwargs) -> 'Symbol':
        '''
        Generates embeddings for the Symbol's value.
        This method uses the @core.embed decorator to generate embeddings for the Symbol's value.
        If the value is not a list, it is converted to a list.

        Args:
            **kwargs: Additional keyword arguments for the @core.embed decorator.

        Returns:
            Symbol: A Symbol object with its value embedded.
        '''
        value = self.value
        if not isinstance(value, list):
            # must convert to list of str for embedding
            value = [str(value)]
        # ensure that all values are strings
        value = [str(v) for v in value]

        @core.embed(entries=value, **kwargs)
        def _func(_) -> list:
            pass

        return self._to_symbol(_func(self))

    @property
    def embedding(self) -> np.array:
        '''
        Get the embedding as a numpy array.

        Returns:
            Any: The embedding of the symbol.
        '''
        # if the embedding is not yet computed, compute it
        if self._metadata.embedding is None:
            if ((isinstance(self.value, list) or isinstance(self.value, tuple)) and all([type(x) == int or type(x) == float or type(x) == bool for x in self.value])) \
                or isinstance(self.value, np.ndarray):
                if isinstance(self.value, list) or isinstance(self.value, tuple):
                    assert len(self.value) > 0, 'Cannot compute embedding of empty list'
                    if isinstance(self.value[0], Symbol):
                        # convert each element to numpy array
                        self._metadata.embedding = np.asarray([x.embedding for x in self.value])
                    elif isinstance(self.value[0], str):
                        # embed each string
                        self._metadata.embedding = np.asarray([Symbol(x).embedding for x in self.value])
                    else:
                        # convert to numpy array
                        self._metadata.embedding = np.asarray(self.value)
                else:
                    # convert to numpy array
                    self._metadata.embedding = np.asarray(self.value)
            elif isinstance(self.value, torch.Tensor):
                self._metadata.embedding = self.value.detach().cpu().numpy()
            else:
                # compute the embedding and store as numpy array
                self._metadata.embedding = np.asarray(self.embed().value)
        if isinstance(self._metadata.embedding, list):
            self._metadata.embedding = np.asarray(self._metadata.embedding)
        # return the embedding
        return self._metadata.embedding

    def _ensure_numpy_format(self, x, cast=False):
        # if it is a Symbol, get its value
        if not isinstance(x, np.ndarray) or not isinstance(x, torch.Tensor) or not isinstance(x, list):
            if not isinstance(x, self._symbol_type): #@NOTE: enforce Symbol to avoid circular import
                if not cast:
                    raise TypeError(f'Cannot compute similarity with type {type(x)}')
                x = self._symbol_type(x)
            # evaluate the Symbol as an embedding
            x = x.embedding
        # if it is a list, convert it to numpy
        if isinstance(x, list) or isinstance(x, tuple):
            assert len(x) > 0, 'Cannot compute similarity with empty list'
            x = np.asarray(x)
        # if it is a tensor, convert it to numpy
        elif isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x.squeeze()[:, None]

    def similarity(self, other: Union['Symbol', list, np.ndarray, torch.Tensor], metric: Union['cosine', 'angular-cosine', 'product', 'manhattan', 'euclidean', 'minkowski', 'jaccard'] = 'cosine', eps: float = 1e-8, normalize: Optional[Callable] = None, **kwargs) -> float:
        '''
        Calculates the similarity between two Symbol objects using a specified metric.
        This method compares the values of two Symbol objects and calculates their similarity according to the specified metric.
        It supports the 'cosine' metric, and raises a NotImplementedError for other metrics.

        Args:
            other (Symbol): The other Symbol object to calculate the similarity with.
            metric (Optional[str]): The metric to use for calculating the similarity. Defaults to 'cosine'.
            eps (float): A small value to avoid division by zero.
            normalize (Optional[Callable]): A function to normalize the Symbol's value before calculating the similarity. Defaults to None.
            **kwargs: Additional keyword arguments for the @core.similarity decorator.

        Returns:
            float: The similarity value between the two Symbol objects.

        Raises:
            TypeError: If any of the Symbol objects is not of type np.ndarray or Symbol.
            NotImplementedError: If the given metric is not supported.
        '''
        v = self._ensure_numpy_format(self)
        if isinstance(other, list) or isinstance(other, tuple):
            o = []
            for i in range(len(other)):
                o.append(self._ensure_numpy_format(other[i], cast=True))
            o = np.concatenate(o, axis=1)
        else:
            o = self._ensure_numpy_format(other, cast=True)

        if   metric == 'cosine':
            val     = (v.T@o / (np.sqrt(v.T@v) * np.sqrt(o.T@o) + eps))
        elif metric == 'angular-cosine':
            c       = kwargs.get('c', 1)
            val     = 1 - (c * np.arccos((v.T@o / (np.sqrt(v.T@v) * np.sqrt(o.T@o) + eps))) / np.pi)
        elif metric == 'product':
            val     = (v.T@o / (v.shape[0] + eps))
        elif metric == 'manhattan':
            val     = (np.abs(v - o).sum(axis=0) / (v.shape[0] + eps))
        elif metric == 'euclidean':
            val     = (np.sqrt(np.sum((v - o)**2, axis=0)) / (v.shape[0] + eps))
        elif metric == 'minkowski':
            p       = kwargs.get('p', 3)
            val     = (np.sum(np.abs(v - o)**p, axis=0)**(1/p) / (v.shape[0] + eps))
        elif metric == 'jaccard':
            val     = (np.sum(np.minimum(v, o)) / np.sum(np.maximum(v, o) + eps))
        else:
            raise NotImplementedError(f"Similarity metric {metric} not implemented. Available metrics: 'cosine'")

        # get the similarity value(s)
        if len(val.shape) >= 1 and val.shape[0] > 1:
            val = val.diagonal()
        else:
            val = val.item()

        if normalize is not None:
            val = normalize(val)
        return val

    def distance(self, other: Union['Symbol', list, np.ndarray, torch.Tensor], kernel: Union['gaussian', 'rbf', 'laplacian', 'polynomial', 'sigmoid', 'linear', 'cauchy', 't-distribution', 'inverse-multiquadric', 'cosine', 'angular-cosine', 'frechet', 'mmd'] = 'gaussian',  eps: float = 1e-8, normalize: Optional[Callable] = None, **kwargs) -> float:
        '''
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
        '''
        v = self._ensure_numpy_format(self)
        if isinstance(other, list) or isinstance(other, tuple):
            o = []
            for i in range(len(other)):
                o.append(self._ensure_numpy_format(other[i], cast=True))
            o = np.concatenate(o, axis=1)
        else:
            o = self._ensure_numpy_format(other, cast=True)

        # compute the kernel value
        if   kernel == 'gaussian':
            gamma   = kwargs.get('gamma', 1)
            val     = np.exp(-gamma * np.sum((v - o)**2, axis=0))
        elif kernel == 'rbf':
            # vectors are expected to be normalized
            bandwidth = kwargs.get('bandwidth', None)
            gamma     = kwargs.get('gamma', 1)
            d         = np.sum((v - o)**2, axis=0)
            if bandwidth is not None:
                val   = 0
                for a in bandwidth:
                    gamma = 1.0 / (2 * a)
                    val  += np.exp(-gamma * d)
            else:
                # if no bandwidth is given, default to the gaussian kernel
                val = np.exp(-gamma * d)
        elif kernel == 'laplacian':
            gamma   = kwargs.get('gamma', 1)
            val     = np.exp(-gamma * np.sum(np.abs(v - o), axis=0))
        elif kernel == 'polynomial':
            gamma   = kwargs.get('gamma', 1)
            degree  = kwargs.get('degree', 3)
            coef    = kwargs.get('coef', 1)
            val     = (gamma * np.sum((v * o), axis=0) + coef)**degree
        elif kernel == 'sigmoid':
            gamma   = kwargs.get('gamma', 1)
            coef    = kwargs.get('coef', 1)
            val     = np.tanh(gamma * np.sum((v * o), axis=0) + coef)
        elif kernel == 'linear':
            val     = np.sum((v * o), axis=0)
        elif kernel == 'cauchy':
            gamma   = kwargs.get('gamma', 1)
            val     = 1 / (1 + np.sum((v - o)**2, axis=0) / gamma)
        elif kernel == 't-distribution':
            gamma   = kwargs.get('gamma', 1)
            degree  = kwargs.get('degree', 1)
            val     = 1 / (1 + (np.sum((v - o)**2, axis=0) / (gamma * degree))**(degree + 1) / 2)
        elif kernel == 'inverse-multiquadric':
            gamma   = kwargs.get('gamma', 1)
            val     = 1 / np.sqrt(np.sum((v - o)**2, axis=0) / gamma**2 + 1)
        elif kernel == 'cosine':
            val     = 1 - (np.sum(v * o, axis=0) / (np.sqrt(np.sum(v**2, axis=0)) * np.sqrt(np.sum(o**2, axis=0)) + eps))
        elif kernel == 'angular-cosine':
            c       = kwargs.get('c', 1)
            val     = c * np.arccos((np.sum(v * o, axis=0) / (np.sqrt(np.sum(v**2, axis=0)) * np.sqrt(np.sum(o**2, axis=0)) + eps))) / np.pi
        elif kernel == 'frechet':
            sigma1  = kwargs.get('sigma1', None)
            sigma2  = kwargs.get('sigma2', None)
            assert sigma1 is not None and sigma2 is not None, 'Frechet distance requires covariance matrices for both inputs'
            v       = v.T
            o       = o.T
            val     = calculate_frechet_distance(v, sigma1, o, sigma2, eps)
        elif kernel == 'mmd':
            v       = v.T
            o       = o.T
            val     = calculate_mmd(v, o, eps=eps)
        else:
            raise NotImplementedError(f"Kernel function {kernel} not implemented. Available functions: 'gaussian'")
        # get the kernel value(s)
        if len(val.shape) >= 1 and val.shape[0] > 1:
            val = val
        else:
            val = val.item()

        if normalize is not None:
            val = normalize(val)
        return val


    def zip(self, **kwargs) -> List[Tuple[str, List, Dict]]:
        '''
        Zips the Symbol's value with its embeddings and a query containing the value.
        This method zips the Symbol's value along with its embeddings and a query containing the value.

        Args:
            **kwargs: Additional keyword arguments for the `embed` method.

        Returns:
            List[Tuple[str, List, Dict]]: A list of tuples containing a unique ID, the value's embeddings, and a query containing the value.

        Raises:
            ValueError: If the Symbol's value is not a string or list of strings.
        '''
        if isinstance(self.value, str):
            self._value = [self.value]
        elif isinstance(self.value, list):
            pass
        else:
            raise ValueError(f'Expected id to be a string, got {type(self.value)}')

        embeds = self.embed(**kwargs).value
        idx    = [str(uuid.uuid4()) for _ in range(len(self.value))]
        query  = [{'text': str(self.value[i])} for i in range(len(self.value))]

        # convert embeds to list if it is a tensor or numpy array
        if type(embeds) == np.ndarray:
            embeds = embeds.tolist()
        elif type(embeds) == torch.Tensor:
            embeds = embeds.cpu().numpy().tolist()

        return list(zip(idx, embeds, query))


class IOHandlingPrimitives(Primitive):
    '''
    This mixin contains functionalities related to input/output operations.
    '''
    def input(self, message: str = 'Please add more information', **kwargs) -> 'Symbol':
        '''
        Request user input and return a Symbol containing the user input.

        Args:
            message (str, optional): The message displayed to request the user input. Defaults to 'Please add more information'.
            **kwargs: Additional keyword arguments to be passed to the `@core.userinput` decorator.

        Returns:
            Symbol: The resulting Symbol after receiving the user input.

        Examples:
        --------
        >>> from symai import Symbol
        >>> s = Symbol().input('Please enter your name')
        >>> [output: 'John']

        >>> s = Symbol('I was born in')
        >>> s = s.input('Please enter the year of your birth')
        >>> [output: 'I was born in 1990'] # if Symbol has a <str> value inputs will be concatenated

        # Works identically for the `Expression` class
        '''
        @core.userinput(**kwargs)
        def _func(_, message) -> str:
            pass

        res = _func(self, message)
        condition = self.value is not None and isinstance(self.value, str)

        if hasattr(self, 'sym_return_type'):
            return self.sym_return_type(self.value if condition else '') | res
        return self._to_symbol(self.value if condition else '') | self._to_symbol(res)

    def open(self, path: str = None, **kwargs) -> 'Symbol':
        '''
        Open a file and store its content in an Expression object as a string.

        Args:
            path (str): The path to the file that needs to be opened.
            **kwargs: Arbitrary keyword arguments to be used by the core.opening decorator.

        Returns:
            Symbol: An Expression object containing the content of the file as a string value.

        Examples:
        --------
        >>> from symai import Symbol
        >>> s = Symbol().open('file.txt')

        >>> s = Symbol('file.txt')
        >>> s = s.open()

        # Works identically for the `Expression` class
        '''

        path = path if path is not None else self.value
        if path is None:
            raise ValueError('Path is not provided; either provide a path or set the value of the Symbol to the path')

        @core.opening(path=path, **kwargs)
        def _func(_):
            pass

        if hasattr(self, 'sym_return_type'):
            return self.sym_return_type(_func(self))
        return self._to_symbol(_func(self))


class IndexingPrimitives(Primitive):
    '''
    This mixin contains functionalities related to indexing symbols.
    '''
    def config(self, path: str, index_name: str, **kwargs) -> 'Symbol':
        '''
        Execute a configuration operation on the index.

        Args:
            path (str): Index configuration path.
            **kwargs: Arbitrary keyword arguments to be used by the core.index decorator.

        Returns:
            Symbol: An Expression object containing the configuration result.
        '''
        @core.index(prompt=path, index_name=index_name, operation='config', **kwargs)
        def _func(_):
            pass
        return _func(self)

    def add(self, doc: List[Tuple[str, List, Dict]], index_name: str, **kwargs) -> 'Symbol':
        '''
        Add an entry to the existing index.

        Args:
            doc (List[Tuple[str, List, Dict]]): The document used to add an entry to the index. Use zip(...) to generate the document.
            **kwargs: Arbitrary keyword arguments to be used by the core.index decorator.

        Returns:
            Symbol: An Expression object containing the addition result.
        '''
        @core.index(prompt=doc, index_name=index_name, operation='add', **kwargs)
        def _func(_):
            pass
        return _func(self)

    def get(self, query: List[int], index_name: str, **kwargs) -> 'Symbol':
        '''
        Search the index based on the provided query.

        Args:
            query (List[int]): The query vector used to search entries in the index.
            **kwargs: Arbitrary keyword arguments to be used by the core.index decorator.

        Returns:
            Symbol: An Expression object containing the search result.
        '''
        # convert query to list if it is a tensor or numpy array
        if isinstance(query, np.ndarray):
            query = query.tolist()
        elif isinstance(query, torch.Tensor):
            query = query.cpu().numpy().tolist()

        @core.index(prompt=query, index_name=index_name, operation='search', **kwargs)
        def _func(_):
            pass
        return _func(self)


class PersistencePrimitives(Primitive):
    '''
    This mixin contains functionalities related to expanding symbols and saving/loading symbols to/from disk.
    Future functionalities in this mixin might include different ways of serialization and deserialization, or more complex expansion techniques etc.
    '''
    def expand(self, *args, **kwargs) -> str:
        '''
        Expand the current Symbol and create a new sub-component.
        The function writes a self-contained function (with all imports) to solve a specific user problem task.
        This method uses the `@core.expand` decorator with a maximum token limit of 2048, and allows additional keyword
        arguments to be passed to the decorator.

        Args:
            *args: Additional arguments for the `@core.expand` decorator.
            **kwargs: Additional keyword arguments for the `@core.expand` decorator.

        Returns:
            Symbol: The name of the newly created sub-component.
        '''
        @core.expand(**kwargs)
        def _func(_, *args): pass

        _tmp_llm_func = self._to_symbol(_func(self, *args))
        func_name = str(_tmp_llm_func.extract('function name'))

        def _llm_func(*args, **kwargs):
            res = _tmp_llm_func.fexecute(*args, **kwargs)

            return res['locals'][func_name]()

        setattr(self, func_name, _llm_func)

        return func_name

    def save(self, path: str, replace: Optional[bool] = False, serialize: Optional[bool] = True) -> None:
        '''
        Save the current Symbol to a file.

        Args:
            path (str): The filepath of the saved file.
            replace (Optional[bool]): Whether to replace the file if it already exists. Defaults to False.
            serialize (Optional[bool]): Whether to serialize the object via pickle instead of writing the string. Defaults to True.

        Returns:
            Symbol: The current Symbol.
        '''
        file_path = path

        if not replace:
            cnt = 0
            while os.path.exists(file_path):
                filename, file_extension = os.path.splitext(path)
                file_path = f'{filename}_{cnt}{file_extension}'
                cnt += 1

        if serialize:
            # serialize the object via pickle instead of writing the string
            path_ = str(file_path) + '.pkl' if not str(file_path).endswith('.pkl') else str(file_path)
            with open(path_, 'wb') as f:
                pickle.dump(self, file=f)
        else:
            with open(str(file_path), 'w') as f:
                f.write(str(self))

    @staticmethod
    def load(path: str) -> Any:
        '''
        Load a Symbol from a file.

        Args:
            path (str): The filepath of the saved file.

        Returns:
            Symbol: The loaded Symbol.
        '''
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj


class OutputHandlingPrimitives(Primitive):
    '''
    This mixin include functionalities related to outputting symbols. It can be expanded in the future to include different types of output methods or complex output formatting, etc.
    '''
    def output(self, *args, **kwargs) -> 'Symbol':
        '''
        Output the current Symbol to an output handler.
        This method uses the `@core.output` decorator and allows additional keyword arguments to be passed to the decorator.

        Args:
            *args: Additional arguments for the `@core.output` decorator.
            **kwargs: Additional keyword arguments for the `@core.output` decorator.

        Returns:
            Symbol: The resulting Symbol after the output operation.
        '''
        @core.output(**kwargs)
        def _func(_, *args):
            pass

        return self._to_symbol(_func(self, *args))


class FineTuningPrimitives(Primitive):
    '''
    This mixin contains functionalities related to fine tuning models.
    '''
    def tune(self, operation: str = 'create', **kwargs) -> 'Symbol':
        '''
        Fine tune a base model.

        Args:
            operation (str, optional): The specific operation to be performed. Defaults to 'create'.
            **kwargs: Additional keyword arguments to be passed to the `@core.tune` decorator dependent on the used operation.

        Returns:
            Symbol: The resulting Symbol containing the fine tuned model ID.
        '''
        @core.tune(operation=operation, **kwargs)
        def _func(_, *args, **kwargs) -> str:
            pass
        return self.sym_return_type(_func(self))

    @property
    def data(self) -> torch.Tensor:
        '''
        Get the data as a Pytorch tensor.

        Returns:
            Any: The data of the symbol.
        '''
        # if the data is not yet computed, compute it
        if self._metadata.data is None:
            # compute the data and store as numpy array
            self._metadata.data = self.embedding
        # if the data is a tensor, return it
        if isinstance(self._metadata.data, torch.Tensor):
            # return tensor
            return self._metadata.data
        # if the data is a numpy array, convert it to tensor
        elif isinstance(self._metadata.data, np.ndarray):
            # convert to tensor
            self._metadata.data = torch.from_numpy(self._metadata.data)
            return self._metadata.data
        else:
            raise TypeError(f'Expected data to be a tensor or numpy array, got {type(self._metadata.data)}')

