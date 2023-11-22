import json
from abc import ABC
from json import JSONEncoder
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import numpy as np

from . import core
from .ops import SYMBOL_PRIMITIVES
from .utils import deprecated


class SymbolEncoder(JSONEncoder):
    def default(self, sym):
        '''
        Encode a Symbol instance into its dictionary representation.

        Args:
            sym (Symbol): The Symbol instance to encode.

        Returns:
            dict: The dictionary representation of the Symbol instance.
        '''
        return sym.__dict__


class Symbol(ABC, *SYMBOL_PRIMITIVES):
    _dynamic_context: Dict[str, List[str]] = {}

    def __init__(self, *value, static_context: Optional[str] = '') -> None:
        '''
        Initialize a Symbol instance with a specified value. Unwraps nested symbols.

        Args:
            value (Optional[Any]): The value of the symbol. Can be a single value or multiple values.
            static_context (Optional[str]): The static context of the symbol. Defaults to an empty string.

        Attributes:
            value (Any): The value of the symbol.
            metadata (Optional[Dict[str, Any]]): The metadata associated with the symbol.
        '''
        super().__init__()
        self.value    = None
        self.metadata = None
        self.parent   = None #@TODO: to enable graph construction
        self.children = None #@TODO: to enable graph construction
        self._static_context = static_context

        if len(value) == 1:

            value = value[0]

            if isinstance(value, Symbol):
                self.value    = value.value
                self.parent   = value.parent
                self.children = value.children
                self.metadata = value.metadata

            elif isinstance(value, list) or isinstance(value, dict) or \
                    isinstance(value, set) or isinstance(value, tuple) or \
                        isinstance(value, str) or isinstance(value, int) or \
                            isinstance(value, float) or isinstance(value, bool):

                if isinstance(value, list):
                    value = [v.value if isinstance(v, Symbol) else v for v in value]

                elif isinstance(value, dict):
                    value = {k: v.value if isinstance(v, Symbol) else v for k, v in value.items()}

                elif isinstance(value, set):
                    value = {v.value if isinstance(v, Symbol) else v for v in value}

                elif isinstance(value, tuple):
                    value = tuple([v.value if isinstance(v, Symbol) else v for v in value])

                self.value = value

            else:
                self.value = value

        elif len(value) > 1:
            self.value = [v.value if isinstance(v, Symbol) else v for v in value]

    @property
    def global_context(self) -> str:
        '''
        Get the global context of the symbol, which consists of the static and dynamic context.

        Returns:
            str: The global context of the symbol.
        '''
        return (self.static_context, self.dynamic_context)

    @property
    def static_context(self) -> str:
        '''
        Get the static context of the symbol which is defined by the user when creating a symbol subclass.

        Returns:
            str: The static context of the symbol.
        '''
        return f'\n[STATIC CONTEXT]\n{self._static_context}' if self._static_context else ''

    @static_context.setter
    def static_context(self, value: str):
        '''
        Set the static context of the symbol which is defined by the user when creating a symbol subclass.
        '''
        if '\n[STATIC CONTEXT]\n' in value:
            value = value.replace('\n[STATIC CONTEXT]\n', '')
        self._static_context = value

    @property
    def dynamic_context(self) -> str:
        '''
        Get the dynamic context which is defined by the user at runtime.
        It helps to alter the behavior of the symbol at runtime.

        Returns:
            str: The dynamic context associated with this symbol type.
        '''
        type_ = str(type(self))
        if type_ not in self._dynamic_context:
            self._dynamic_context[type_] = []
            return ''

        val = '\n'.join(self._dynamic_context[type_])

        return f'\n[DYNAMIC CONTEXT]\n{val}' if val else ''

    def json(self) -> Dict[str, Any]:
        '''
        Get the json-serializable dictionary representation of the Symbol instance.

        Returns:
            dict: The json-serializable dictionary representation of the Symbol instance.
        '''
        return self.__getstate__()

    def serialize(self):
        '''
        Encode an Symbol instance into its dictionary representation.

        Args:
            obj (Symbol): The Expression instance to encode.

        Returns:
            dict: The dictionary representation of the Symbol instance.
        '''
        return json.dumps(self, cls=SymbolEncoder)

    def _to_symbol(self, value: Any) -> "Symbol":
        '''
        Convert a value to a Symbol instance.

        Args:
            value (Any): The value to convert to a Symbol instance.

        Returns:
            Symbol: The Symbol instance.
        '''
        if isinstance(value, Symbol):
            return value

        return Symbol(value)

    def __call__(self):
        '''
        Evaluate the symbol and return its value.
        '''
        return self.value

    def __hash__(self) -> int:
        '''
        Get the hash value of the symbol.

        Returns:
            int: The hash value of the symbol.
        '''
        return str(self.value).__hash__()

    def __getstate__(self) -> Dict[str, Any]:
        '''
        Get the state of the symbol for serialization.

        Returns:
            dict: The state of the symbol.
        '''
        return vars(self)

    def __setstate__(self, state) -> None:
        '''
        Set the state of the symbol for deserialization.

        Args:
            state (dict): The state to set the symbol to.
        '''
        vars(self).update(state)

    def __getattr__(self, key) -> Any:
        '''
        Get an attribute from the symbol if it exists. Otherwise, attempt to get the attribute from the symbol's value.
        If the attribute does not exist in the symbol or its value, raise an AttributeError with a cascading error message.

        Args:
            key (str): The name of the attribute.

        Returns:
            Any: The attribute value if the attribute exists.

        Raises:
            AttributeError: If the attribute does not exist.
        '''
        if not self.__dict__.__contains__(key):
            try:
                att = getattr(self.value, key)
            except AttributeError as e:
                raise AttributeError(f"Cascading call failed, since object has no attribute '{key}'. Original error message: {e}")
            return att

        return self.__dict__[key]

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

    def __eq__(self, other: Any) -> bool:
        '''
        Check if the current Symbol is equal to another Symbol.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to check for equality.

        Returns:
            bool: True if the current Symbol is equal to the 'other' Symbol, otherwise False.
        '''
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
        return Symbol(str(self) + str(other))

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
        return Symbol(str(other) + str(self))

    def __imatmul__(self, other: Any) -> 'Symbol':
        '''
        This method concatenates the string representation of two Symbol objects and assigns the concatenated result to the value of the current Symbol object.
        By default, if 'other' is not a Symbol, it's casted to a Symbol object.

        Args:
            other (Any): The object to concatenate.

        Returns:
            Symbol: The current Symbol object with the concatenated value.
        '''
        self.value =  Symbol(str(self) + str(other))
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

    def __len__(self) -> int:
        '''
        Get the length of the tokenized value of the Symbol.

        Returns:
            int: The length of the tokenized value of the Symbol.
        '''
        return len(self.tokens)

    def __bool__(self) -> bool:
        '''
        Get the boolean value of the Symbol.
        If the Symbol's value is of type 'bool', the method returns the boolean value, otherwise it returns False.

        Returns:
            bool: The boolean value of the Symbol.
        '''
        val = False
        if isinstance(self.value, bool):
            val = self.value
        elif self.value is not None:
            val = True if self.value else False

        return val

    def __str__(self) -> str:
        '''
        Get the string representation of the Symbol's value.

        Returns:
            str: The string representation of the Symbol's value.
        '''
        if self.value is None:
            return ''
        elif isinstance(self.value, list) or isinstance(self.value, np.ndarray) or isinstance(self.value, tuple):
            return str([str(v) for v in self.value])
        elif isinstance(self.value, dict):
            return str({k: str(v) for k, v in self.value.items()})
        elif isinstance(self.value, set):
            return str({str(v) for v in self.value})
        else:
            return str(self.value)

    def __repr__(self) -> str:
        '''
        Get the representation of the Symbol object as a string.

        Returns:
            str: The representation of the Symbol object.
        '''
        return f'{type(self)}(value={str(self.value)})'

    def _repr_html_(self) -> str:
        '''
        Get the HTML representation of the Symbol's value.

        Returns:
            str: The HTML representation of the Symbol's value.
        '''
        return str(self.value)

    def __iter__(self) -> Iterator:
        '''
        Get an iterator for the Symbol's value.
        If the Symbol's value is a list, tuple, or numpy.ndarray, iterate over the elements. Otherwise, create a new list with a single item and iterate over the list.

        Returns:
            Iterator: An iterator for the Symbol's value.
        '''
        if isinstance(self.value, list) or isinstance(self.value, tuple) or isinstance(self.value, np.ndarray):
            return iter(self.value)

        return self.list('item').value.__iter__()

    def __reversed__(self) -> Iterator:
        '''
        Get a reversed iterator for the Symbol's value.

        Returns:
            Iterator: A reversed iterator for the Symbol's value.
        '''
        return reversed(list(self.__iter__()))

    def __next__(self) -> Any:
        '''
        Get the next item in the iterable value of the Symbol.
        If it is not a list, tuple, or numpy array, the method falls back to using the @core.next() decorator, which retrieves and returns the next item using core functions.

        Returns:
            Symbol: The next item in the iterable value of the Symbol.

        Raises:
            StopIteration: If the iterable value reaches its end.
        '''
        return next(self.__iter__())

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

        return Symbol(_func(self, key))

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

        self.value = Symbol(_func(self, key, value)).value

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

        self.value = Symbol(_func(self, key)).value

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

        return Symbol(_func(self))

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

        return Symbol(_func(self))

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

        return Symbol(_func(self))

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

        return Symbol(_func(self, information))

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

        return Symbol(_func(self, information))

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

        return Symbol(_func(self, information))

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

        return Symbol(_func(self, other))

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
        return Symbol(_func(other, self))

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
        self.value = self.__add__(other)
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

        return Symbol(_func(self, other, ''))

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
        return Symbol(_func(other, self, ''))

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
        self.value = val.value

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

        return Symbol(_func(self, other))

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

        return Symbol(_func(self, other))

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

        return Symbol(_func(self, other))

    def __truediv__(self, other: Any) -> 'Symbol':
        '''
        Divides the symbol value by another, splitting the symbol value by the other value.
        The string representation of the other value is used to split the symbol value.

        Args:
            other (Any): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        '''
        return Symbol(str(self).split(str(other)))


class ExpressionEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, Expression):
            return o.__getstate__()
        return JSONEncoder.default(self, o)


class Expression(Symbol):

    def __init__(self, value = None, *args, **kwargs):
        '''
        Create an Expression object that will be evaluated lazily using the forward method.

        Args:
            value (Any, optional): The value to be stored in the Expression object. Usually not provided as the value
                                   is computed using the forward method when called. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        '''
        super().__init__(value)
        self._sym_return_type = type(self)

    def __call__(self, *args, **kwargs) -> Any:
        '''
        Evaluate the expression using the forward method and assign the result to the value attribute.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the forward method.
        '''
        return self.forward(*args, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_sym_return_type', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._sym_return_type = type(self)

    def __json__(self):
        '''
        Get the json-serializable dictionary representation of the Expression instance.

        Returns:
            dict: The json-serializable dictionary representation of the Expression instance.
        '''
        return self.__getstate__()

    def serialize(self):
        '''
        Encode an Expression instance into its dictionary representation.

        Args:
            obj (Expression): The Expression instance to encode.

        Returns:
            dict: The dictionary representation of the Expression instance.
        '''
        return json.dumps(self, cls=ExpressionEncoder)

    @property
    def sym_return_type(self) -> Type:
        '''
        Returns the casting type of this expression.

        Returns:
            Type: The casting type of this expression. Defaults to the current Expression-type.
        '''
        return self._sym_return_type

    @sym_return_type.setter
    def sym_return_type(self, type: Type) -> None:
        '''
        Sets the casting type of this expression.

        Args:
            type (Type): The casting type of this expression.
        '''
        self._sym_return_type = type

    def _to_symbol(self, value: Any) -> 'Symbol':
        '''
        Create a Symbol instance from a given input value.
        Helper function used to ensure that all values are wrapped in a Symbol instance.

        Args:
            value (Any): The input value.

        Returns:
            Symbol: The Symbol instance with the given input value.
        '''
        if isinstance(value, Symbol):
            return value

        return Symbol(value)


    def forward(self, *args, **kwargs) -> Symbol: #TODO make reserved kwargs with underscore: __<cmd>__
        '''
        Needs to be implemented by subclasses to specify the behavior of the expression during evaluation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Symbol: The evaluated result of the implemented forward method.
        '''
        raise NotImplementedError()

    # TODO: consider if this should be deprecated and moved only as an Interface
    def input(self, message: str = 'Please add more information', **kwargs) -> 'Symbol':
        '''
        Request user input and return a Symbol containing the user input.

        Args:
            message (str, optional): The message displayed to request the user input. Defaults to 'Please add more information'.
            **kwargs: Additional keyword arguments to be passed to the `@core.userinput` decorator.

        Returns:
            Symbol: The resulting Symbol after receiving the user input.
        '''
        @core.userinput(**kwargs)
        def _func(_, message) -> str:
            pass
        return self.sym_return_type(_func(self, message))

    # TODO: consider if this should be deprecated and moved only as an Interface
    def open(self, path: str, **kwargs) -> 'Symbol':
        '''
        Open a file and store its content in an Expression object as a string.

        Args:
            path (str): The path to the file that needs to be opened.
            **kwargs: Arbitrary keyword arguments to be used by the core.opening decorator.

        Returns:
            Symbol: An Expression object containing the content of the file as a string value.
        '''
        @core.opening(path=path, **kwargs)
        def _func(_) -> str:
            pass
        return self.sym_return_type(_func(self))

    # TODO: consider if this should be deprecated and moved only as an Interface
    def index(self, path: str, **kwargs) -> 'Symbol':
        '''
        Execute a configuration operation on the index.

        Args:
            path (str): Index configuration path.
            **kwargs: Arbitrary keyword arguments to be used by the core.index decorator.

        Returns:
            Symbol: An Expression object containing the configuration result.
        '''
        @core.index(prompt=path, operation='config', **kwargs)
        def _func(_):
            pass
        return self.sym_return_type(_func(self))

    # TODO: consider if this should be deprecated and moved only as an Interface
    def add(self, query: List[str], **kwargs) -> 'Symbol':
        '''
        Add an entry to the existing index.

        Args:
            query (List[str]): The query string used to add an entry to the index.
            **kwargs: Arbitrary keyword arguments to be used by the core.index decorator.

        Returns:
            Symbol: An Expression object containing the addition result.
        '''
        @core.index(prompt=query, operation='add', **kwargs)
        def _func(_):
            pass
        return self.sym_return_type(_func(self))

    # TODO: consider if this should be deprecated and moved only as an Interface
    def get(self, query: List[int], **kwargs) -> 'Symbol':
        '''
        Search the index based on the provided query.

        Args:
            query (List[int]): The query vector used to search entries in the index.
            **kwargs: Arbitrary keyword arguments to be used by the core.index decorator.

        Returns:
            Symbol: An Expression object containing the search result.
        '''
        @core.index(prompt=query, operation='search', **kwargs)
        def _func(_):
            pass
        return self.sym_return_type(_func(self))

    @staticmethod
    def command(engines: List[str] = ['all'], **kwargs) -> 'Symbol':
        '''
        Execute command(s) on engines.

        Args:
            engines (List[str], optional): The list of engines on which to execute the command(s). Defaults to ['all'].
            **kwargs: Arbitrary keyword arguments to be used by the core.command decorator.

        Returns:
            Symbol: An Expression object representing the command execution result.
        '''
        @core.command(engines=engines, **kwargs)
        def _func(_):
            pass
        return Expression(_func(Expression()))

    @staticmethod
    def setup(engines: Dict[str, Any], **kwargs) -> 'Symbol':
        '''
        Configure multiple engines.

        Args:
            engines (Dict[str, Any]): A dictionary containing engine names as keys and their configurations as values.
            **kwargs: Arbitrary keyword arguments to be used by the core.setup decorator.

        Returns:
            Symbol: An Expression object representing the setup result.
        '''
        @core.setup(engines=engines, **kwargs)
        def _func(_):
            pass
        return Expression(_func(Expression()))
