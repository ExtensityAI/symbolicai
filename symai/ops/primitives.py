import ast
import os
import pickle
import uuid
import numpy as np

from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Type, Union)

from .. import core
from .. import core_ext
from ..prompts import Prompt

if TYPE_CHECKING:
    from ..symbol import Expression, Symbol


class ArithmeticPrimitives:
    '''
    This mixin contains functions that perform arithmetic operations on symbols or symbol values.
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
            if (isinstance(key, int) or isinstance(key, slice)) and (isinstance(self.value, list) or isinstance(self.value, tuple) or isinstance(self.value, np.ndarray)):
                return self.value[key]
            elif (isinstance(key, str) or isinstance(key, int)) and isinstance(self.value, dict):
                return self.value[key]
        except KeyError:
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

        @core.delitem()
        def _func(_, index: str):
            pass

        self._value = self._to_symbol(_func(self, key)).value

    def __contains__(self, other: Any) -> bool:
        '''
        Check if a Symbol object is present in another Symbol object.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to check for containment.

        Returns:
            bool: True if the current Symbol contains the 'other' Symbol, otherwise False.
        '''
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
        print('debug')
        @core.equals()
        def _func(_, other) -> bool:
            pass

        return _func(self, other)

    def __matmul__(self, other: Any) -> 'Symbol':
        '''
        This method concatenates the string representation of two Symbol objects and returns a new Symbol with the concatenated result.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to concatenate.

        Returns:
            Symbol: A new Symbol object with the concatenated value.
        '''
        return self._to_symbol(str(self) + str(other))

    def __rmatmul__(self, other: Any) -> 'Symbol':
        '''
        This method concatenates the string representation of two Symbol objects in a reversed order and returns a new Symbol with the concatenated result.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to concatenate.

        Returns:
            Symbol: A new Symbol object with the concatenated value.
        '''
        other = self._to_symbol(other)
        return self._to_symbol(str(other) + str(self))

    def __imatmul__(self, other: Any) -> 'Symbol':
        '''
        This method concatenates the string representation of two Symbol objects and assigns the concatenated result to the value of the current Symbol object.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to concatenate.

        Returns:
            Symbol: The current Symbol object with the concatenated value.
        '''
        self._value =  self._to_symbol(str(self) + str(other))
        return self

    def __ne__(self, other: Any) -> bool:
        '''
        This method checks if a Symbol object is not equal to another Symbol by using the __eq__ method.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to check for inequality.

        Returns:
            bool: True if the current Symbol is not equal to the 'other' Symbol, otherwise False.
        '''
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
        @core.invert()
        def _func(_):
            pass

        return self._to_symbol(_func(self))

    def __lshift__(self, information: Any) -> 'Symbol':
        '''
        Add new information to the Symbol.
        The method uses the @core.include decorator to incorporate information into the Symbol.

        Args:
            information (Any): The information to include in the Symbol.

        Returns:
            Symbol: The Symbol with the new information included.
        '''
        @core.include()
        def _func(_, information: str):
            pass

        return self._to_symbol(_func(self, information))

    def __rshift__(self, information: Any) -> 'Symbol':
        '''
        Add new information to the Symbol.
        The method uses the @core.include decorator to incorporate information into the Symbol.

        Args:
            information (Any): The information to include in the Symbol.

        Returns:
            Symbol: The Symbol with the new information included.
        '''
        @core.include()
        def _func(_, information: str):
            pass

        return self._to_symbol(_func(self, information))

    def __rrshift__(self, information: Any) -> 'Symbol':
        '''
        Add new information to the Symbol.
        The method uses the @core.include decorator to incorporate information into the Symbol.

        Args:
            information (Any): The information to include in the Symbol.

        Returns:
            Symbol: The Symbol with the new information included.
        '''
        @core.include()
        def _func(_, information: str):
            pass

        return self._to_symbol(_func(self, information))

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
        val = self.__sub__(other)
        self._value = val.value

        return self

    def __and__(self, other: Any) -> 'Symbol':
        '''
        Performs a logical AND operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.
        Uses the core.logic decorator with operator='and' to create a _func method for the AND operation.

        Args:
            other (Any): The string to perform the AND operation with the symbol value.

        Returns:
            Symbol: A new symbol with the result of the AND operation.
        '''
        @core.logic(operator='and')
        def _func(_, a: str, b: str):
            pass

        return self._to_symbol(_func(self, other))

    def __or__(self, other: Any) -> 'Symbol':
        '''
        Performs a logical OR operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.
        Uses the core.logic decorator with operator='or' to create a _func method for the OR operation.

        Args:
            other (Any): The string to perform the OR operation with the symbol value.

        Returns:
            Symbol: A new symbol with the result of the OR operation.
        '''
        @core.logic(operator='or')
        def _func(_, a: str, b: str):
            pass

        return self._to_symbol(_func(self, other))

    def __xor__(self, other: Any) -> 'Symbol':
        '''
        Performs a logical XOR operation between the symbol value and another.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.
        Uses the core.logic decorator with operator='xor' to create a _func method for the XOR operation.

        Args:
            other (Any): The string to perform the XOR operation with the symbol value.

        Returns:
            Symbol: A new symbol with the result of the XOR operation.
        '''
        @core.logic(operator='xor')
        def _func(_, a: str, b: str):
            pass

        return self._to_symbol(_func(self, other))

    def __truediv__(self, other: Any) -> 'Symbol':
        '''
        Divides the symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        '''
        return self._to_symbol(str(self).split(str(other)))


class ContextualPrimitives:
    '''
    This mixin contains functions that deal with the context of the symbol. The functions in this mixin manage dynamic context of symbols (like adding, clearing), or deal with type checking and related functionalities.
    New functionalities might include operations that further interact with or manipulate the context associated with symbols.
    '''
    def update(self, feedback: str) -> None:
        '''
        Update the dynamic context with a given runtime feedback.

        Args:
            feedback (str): The feedback to be added to the dynamic context.

        '''
        type_ = str(type(self))
        if type_ not in self._dynamic_context:
            self._dynamic_context[type_] = []

        self._dynamic_context[type_].append(feedback)

    def clear(self) -> None:
        '''
        Clear the dynamic context associated with this symbol type.
        '''
        type_ = str(type(self))
        if type_ not in self._dynamic_context:
            self._dynamic_context[type_] = []
            return self

        self._dynamic_context[type_].clear()


class ValueHandlingPrimitives:
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

    def cast(self, as_type: Type) -> Any:
        '''
        Cast the Symbol's value to a specific type.

        Args:
            as_type (Type): The type to cast the Symbol's value to.

        Returns:
            The Symbol's value casted to the specified type.
        '''
        return as_type(self.value)

    def ast(self) -> Any:
        '''
        Converts the string representation of the Symbol's value to an abstract syntax tree using 'ast.literal_eval'.

        Returns:
            The abstract syntax tree representation of the Symbol's value.
        '''
        return ast.literal_eval(str(self.value))

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


class ComparisonPrimitives:
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


class ExpressionHandlingPrimitives:
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


class DataHandlingPrimitives:
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


class UniquenessPrimitives:
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


class PatternMatchingPrimitives:
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


class QueryHandlingPrimitives:
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


class ExecutionControlPrimitives:
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


class DictHandlingPrimitives:
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


class TemplateStylingPrimitives:
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


class DataClusteringPrimitives:
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
        if not isinstance(self.value, list):
            self._value = [self.value]

        @core.embed(entries=self.value, **kwargs)
        def _func(_) -> list:
            pass

        return self._to_symbol(_func(self))

    def similarity(self, other: Union['Symbol', np.ndarray], metric: Union['cosine', 'product', 'manhattan', 'euclidean', 'minkowski', 'jaccard'] = 'cosine', eps: float = 1e-8, **kwargs) -> float:
        '''
        Calculates the similarity between two Symbol objects using a specified metric.
        This method compares the values of two Symbol objects and calculates their similarity according to the specified metric.
        It supports the 'cosine' metric, and raises a NotImplementedError for other metrics.

        Args:
            other (Symbol): The other Symbol object to calculate the similarity with.
            metric (Optional[str]): The metric to use for calculating the similarity. Defaults to 'cosine'.
            eps (float): A small value to avoid division by zero.
            **kwargs: Additional keyword arguments for the @core.similarity decorator.

        Returns:
            float: The similarity value between the two Symbol objects.

        Raises:
            TypeError: If any of the Symbol objects is not of type np.ndarray or Symbol.
            NotImplementedError: If the given metric is not supported.
        '''
        def _ensure_format(x):
            if not isinstance(x, np.ndarray):
                if not isinstance(x, self._symbol_type): #@NOTE: enforce Symbol to avoid circular import
                    raise TypeError(f'Cannot compute similarity with type {type(x)}')
                x = np.array(x.value)
            return x.squeeze()[:, None]

        v = _ensure_format(self)
        o = _ensure_format(other)

        if metric == 'cosine':
            val = (v.T@o / (np.sqrt(v.T@v) * np.sqrt(o.T@o) + eps)).item()
        elif metric == 'product':
            val = (v.T@o / (v.shape[0] + eps)).item()
        elif metric == 'manhattan':
            val = (np.abs(v - o).sum(axis=0) / (v.shape[0] + eps)).item()
        elif metric == 'euclidean':
            val = (np.sqrt(np.sum((v - o)**2, axis=0)) / (v.shape[0] + eps)).item()
        elif metric == 'minkowski':
            p = kwargs.get('p', 3)
            val = (np.sum(np.abs(v - o)**p, axis=0)**(1/p) / (v.shape[0] + eps)).item()
        elif metric == 'jaccard':
            val = (np.sum(np.minimum(v, o)) / np.sum(np.maximum(v, o) + eps)).item()
        else:
            raise NotImplementedError(f"Similarity metric {metric} not implemented. Available metrics: 'cosine'")

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

        return list(zip(idx, embeds, query))


class PersistencePrimitives:
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


class OutputHandlingPrimitives:
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



class FineTuningPrimitives:
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