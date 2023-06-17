import ast
import os
import uuid
from abc import ABC
from json import JSONEncoder
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import core


class SymbolEncoder(JSONEncoder):
    def default(self, o):
        """Encode a Symbol instance into its dictionary representation.

        Args:
            o (Symbol): The Symbol instance to encode.

        Returns:
            dict: The dictionary representation of the Symbol instance.
        """
        return o.__dict__


class Symbol(ABC):
    _dynamic_context: Dict[str, List[str]] = {}

    def __init__(self, *value) -> None:
        """Initialize a Symbol instance with a specified value. Unwraps nested symbols.

        Args:
            value (Optional[Any]): The value of the symbol. Can be a single value or multiple values.

        Attributes:
            value (Any): The value of the symbol.
            _static_context (str): Static context associated with the symbol.
            relations (List[Symbol]): List of related symbols.
        """
        super().__init__()
        if len(value) == 1:
            value = value[0]
            if isinstance(value, Symbol):
                self.value = value.value
            elif isinstance(value, list) or isinstance(value, dict) or \
                    isinstance(value, set) or isinstance(value, tuple) or \
                        isinstance(value, str) or isinstance(value, int) or \
                            isinstance(value, float) or isinstance(value, bool):
                # unwrap nested symbols
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
        else:
            self.value = None

        self._static_context: str = ''
        self.relations: List[Symbol] = []

    @property
    def _sym_return_type(self):
        """Get the return type of the symbol, which is Symbol itself.
        Can be overridden by subclasses to return a different type.

        Returns:
            type: The Symbol class.
        """
        return Symbol

    @property
    def global_context(self) -> str:
        """Get the global context of the symbol, which consists of the static and dynamic context.

        Returns:
            str: The global context of the symbol.
        """
        return (self.static_context, self.dynamic_context)

    @property
    def static_context(self) -> str:
        """Get the static context of the symbol which is defined by the user when creating a symbol subclass.

        Returns:
            str: The static context of the symbol.
        """
        return self._static_context

    @property
    def dynamic_context(self) -> str:
        """Get the dynamic context which is defined by the user at runtime.
        It helps to alter the behavior of the symbol at runtime.

        Returns:
            str: The dynamic context associated with this symbol type.
        """
        type_ = str(type(self))
        if type_ not in Symbol._dynamic_context:
            Symbol._dynamic_context[type_] = []
        _dyn_ctxt = Symbol._dynamic_context[type_]
        if len(_dyn_ctxt) == 0:
            return ''
        val_ = '\n'.join(_dyn_ctxt)
        val_ = f"\nSPECIAL RULES:\n{val_}"
        return val_

    def _to_symbol(self, value: Any) -> "Symbol":
        """Create a Symbol instance from a given input value.
        Used to ensure that all values are wrapped in a Symbol instance.

        Args:
            value (Any): The input value.

        Returns:
            Symbol: The Symbol instance with the given input value.
        """
        if isinstance(value, Symbol):
            return value
        return Symbol(value)

    def update(self, feedback: str) -> "Symbol":
        """Update the dynamic context with a given runtime feedback.

        Args:
            feedback (str): The feedback to be added to the dynamic context.

        Returns:
            Symbol: The updated symbol.
        """
        type_ = str(type(self))
        if type_ not in Symbol._dynamic_context:
            Symbol._dynamic_context[type_] = []
        self._dynamic_context[type_].append(feedback)
        return self

    def clear(self) -> "Symbol":
        """Clear the dynamic context associated with this symbol type.

        Returns:
            Symbol: The symbol with cleared dynamic context.
        """
        type_ = str(type(self))
        if type_ not in Symbol._dynamic_context:
            Symbol._dynamic_context[type_] = []
            return self
        self._dynamic_context.clear()
        return self

    def __call__(self, *args, **kwargs):
        """Evaluate the symbol and return its value."""
        return self.value

    def __hash__(self) -> int:
        """Get the hash value of the symbol.

        Returns:
            int: The hash value of the symbol.
        """
        return str(self.value).__hash__()

    def __getstate__(self):
        """Get the state of the symbol for serialization.

        Returns:
            dict: The state of the symbol.
        """
        return vars(self)

    def __setstate__(self, state):
        """Set the state of the symbol for deserialization.

        Args:
            state (dict): The state to set the symbol to.
        """
        vars(self).update(state)

    def __getattr__(self, key):
        """Get an attribute from the symbol if it exists. Otherwise, attempt to get the attribute from the symbol's value.
        If the attribute does not exist in the symbol or its value, raise an AttributeError with a cascading error message.

        Args:
            key (str): The name of the attribute.

        Returns:
            Any: The attribute value if the attribute exists.

        Raises:
            AttributeError: If the attribute does not exist.
        """
        if not self.__dict__.__contains__(key):
            try:
                att = getattr(self.value, key)
            except AttributeError as e:
                raise AttributeError(f"Cascading call failed, since object has no attribute '{key}'. Original error message: {e}")
            return att
        return self.__dict__[key]

    def __contains__(self, other) -> bool:
        """Check if the current Symbol contains another Symbol.

        This method checks if a Symbol object contains another Symbol using the @core.contains() decorator.

        Args:
            other (Symbol): The Symbol object to check for containment.

        Returns:
            bool: True if the current Symbol contains the 'other' Symbol, otherwise False.
        """
        @core.contains()
        def _func(_, other) -> bool:
            pass
        return _func(self, other)

    def isinstanceof(self, query: str, **kwargs) -> bool:
        """Check if the current Symbol is an instance of a specific type.

        This method checks if the current Symbol is an instance of the specified type using the @core.isinstanceof() decorator.

        Args:
            query (str): The type to check if the Symbol is an instance of.
            **kwargs: Any additional kwargs for @core.isinstanceof() decorator.

        Returns:
            bool: True if the current Symbol is an instance of the specified type, otherwise False.
        """
        @core.isinstanceof()
        def _func(_, query: str, **kwargs) -> bool:
            pass
        return _func(self, query, **kwargs)

    def __eq__(self, other) -> bool:
        """Check if the current Symbol is equal to another Symbol.

        This method checks if a Symbol object is equal to another Symbol using the @core.equals() decorator.

        Args:
            other (Symbol): The Symbol object to check for equality.

        Returns:
            bool: True if the current Symbol is equal to the 'other' Symbol, otherwise False.
        """
        @core.equals()
        def _func(_, other) -> bool:
            pass
        return _func(self, other)

    def __matmul__(self, other) -> "Symbol":
        """Concatenate the string representations of two Symbol objects.

        This method concatenates the string representation of two Symbol objects and returns a new Symbol with the concatenated result.

        Args:
            other (Symbol): The Symbol object to concatenate.

        Returns:
            Symbol: A new Symbol object with the concatenated value.
        """
        return self._sym_return_type(str(self) + str(other))

    def __rmatmul__(self, other) -> "Symbol":
        """Concatenate the string representations of two Symbol objects in a reversed order.

        This method concatenates the string representation of two Symbol objects in a reversed order and returns a new Symbol with the concatenated result.

        Args:
            other (Symbol): The Symbol object to concatenate.

        Returns:
            Symbol: A new Symbol object with the concatenated value.
        """
        return self._sym_return_type(str(other) + str(self))

    def __imatmul__(self, other) -> "Symbol":
        """Concatenate the string representations of two Symbol objects and assign the result to the current Symbol.

        This method concatenates the string representation of two Symbol objects and assigns the concatenated result to the value of the current Symbol object.

        Args:
            other (Symbol): The Symbol object to concatenate.

        Returns:
            Symbol: The current Symbol object with the concatenated value.
        """
        self.value = Symbol(str(self) + str(other))
        return self

    def __ne__(self, other) -> bool:
        """Check if the current Symbol is not equal to another Symbol.

        This method checks if a Symbol object is not equal to another Symbol by using the __eq__ method.

        Args:
            other (Symbol): The Symbol object to check for inequality.

        Returns:
            bool: True if the current Symbol is not equal to the 'other' Symbol, otherwise False.
        """
        return not self.__eq__(other)

    def __gt__(self, other) -> bool:
        """Check if the current Symbol is greater than another Symbol.

        This method checks if a Symbol object is greater than another Symbol using the @core.compare() decorator with the '>' operator.

        Args:
            other (Symbol): The Symbol object to compare.

        Returns:
            bool: True if the current Symbol is greater than the 'other' Symbol, otherwise False.
        """
        @core.compare(operator = '>')
        def _func(_, other) -> bool:
            pass
        return _func(self, other)

    def __lt__(self, other) -> bool:
        """Check if the current Symbol is less than another Symbol.

        This method checks if a Symbol object is less than another Symbol using the @core.compare() decorator with the '<' operator.

        Args:
            other (Symbol): The Symbol object to compare.

        Returns:
            bool: True if the current Symbol is less than the 'other' Symbol, otherwise False.
        """
        @core.compare(operator = '<')
        def _func(_, other) -> bool:
            pass
        return _func(self, other)

    def __le__(self, other) -> bool:
        """Check if the current Symbol is less than or equal to another Symbol.

        This method checks if a Symbol object is less than or equal to another Symbol using the @core.compare() decorator with the '<=' operator.

        Args:
            other (Symbol): The Symbol object to compare.

        Returns:
            bool: True if the current Symbol is less than or equal to the 'other' Symbol, otherwise False.
        """
        @core.compare(operator = '<=')
        def _func(_, other) -> bool:
            pass
        return _func(self, other)

    def __ge__(self, other) -> bool:
        """Check if the current Symbol is greater than or equal to another Symbol.

        This method checks if a Symbol object is greater than or equal to another Symbol using the @core.compare() decorator with the '>=' operator.

        Args:
            other (Symbol): The Symbol object to compare.

        Returns:
            bool: True if the current Symbol is greater than or equal to the 'other' Symbol, otherwise False.
        """
        @core.compare(operator = '>=')
        def _func(_, other) -> bool:
            pass
        return _func(self, other)

    def __len__(self):
        """Get the length of the string representation of the Symbol's value.

        Returns:
            int: The length of the string representation of the Symbol's value.
        """
        return len(str(self.value))

    def __bool__(self):
        """Get the boolean value of the Symbol.

        If the Symbol's value is of type bool, the method returns the boolean value, otherwise it returns False.

        Returns:
            bool: The boolean value of the Symbol.
        """
        return bool(self.value) if isinstance(self.value, bool) else False

    @property
    def length(self) -> int:
        """Get the length of the string representation of the Symbol's value.

        Returns:
            int: The length of the string representation of the Symbol's value.
        """
        return len(str(self.value))

    @property
    def size(self) -> int:
        """Get the number of tokens in the Symbol's value.

        Returns:
            int: The number of tokens in the Symbol's value.
        """
        return len(self.tokens)

    @property
    def tokens(self) -> int:
        """Tokenize the Symbol's value using the tokenizer method.

        The tokenizer method is bound to the 'neurosymbolic' engine using the @core.bind() decorator.

        Returns:
            int: The tokenized value of the Symbol.
        """
        return self.tokenizer().encode(str(self))

    @core.bind(engine='neurosymbolic', property='tokenizer')
    def tokenizer(self) -> object:
        """The tokenizer method.

        This method is bound to the 'neurosymbolic' engine using the @core.bind() decorator.

        Returns:
            object: The tokenizer object.
        """
        pass

    def type(self):
        """Get the type of the Symbol's value.

        Returns:
            type: The type of the Symbol's value.
        """
        return type(self.value)

    def cast(self, type_):
        """Cast the Symbol's value to a specific type.

        Args:
            type_ (type): The type to cast the Symbol's value to.

        Returns:
            The Symbol's value casted to the specified type.
        """
        return type_(self.value)

    def ast(self):
        """Converts the string representation of the Symbol's value to an abstract syntax tree using ast.literal_eval.

        Returns:
            The abstract syntax tree representation of the Symbol's value.
        """
        return ast.literal_eval(str(self.value))

    def __str__(self) -> str:
        """Get the string representation of the Symbol's value.

        Returns:
            str: The string representation of the Symbol's value.
        """
        if self.value is None:
            return str(None)
        elif isinstance(self.value, list) or isinstance(self.value, np.ndarray) or isinstance(self.value, tuple):
            return str([str(v) for v in self.value])
        elif isinstance(self.value, dict):
            return str({k: str(v) for k, v in self.value.items()})
        elif isinstance(self.value, set):
            return str({str(v) for v in self.value})
        else:
            return str(self.value)

    def __repr__(self):
        """Get the representation of the Symbol object as a string.

        Returns:
            str: The representation of the Symbol object.
        """
        return f"{type(self)}(value={str(self.value)})"

    def _repr_html_(self):
        """Get the HTML representation of the Symbol's value.

        Returns:
            str: The HTML representation of the Symbol's value.
        """
        return f"""<div class="alert alert-success" role="alert">
  {str(self.value)}
</div>"""

    def __iter__(self):
        """Get an iterator for the Symbol's value.

        If the Symbol's value is a list, tuple, or numpy.ndarray, iterate over the elements. Otherwise, create a new list with a single item and iterate over the list.

        Returns:
            iterator: An iterator for the Symbol's value.
        """
        if isinstance(self.value, list) or isinstance(self.value, tuple) or isinstance(self.value, np.ndarray):
            return iter(self.value)
        return self.list('item').value.__iter__()

    def __reversed__(self):
        """Get a reversed iterator for the Symbol's value.

        Returns:
            iterator: A reversed iterator for the Symbol's value.
        """
        return reversed(list(self.__iter__()))

    def __next__(self) -> "Symbol":
        """Get the next item in the iterable value of the Symbol.
        If it is not a list, tuple, or numpy array, the method falls back to using the @core.next() decorator, which retrieves and returns the next item using core functions.

        Returns:
            Symbol: The next item in the iterable value of the Symbol.

        Raises:
            StopIteration: If the iterable value reaches its end.
        """
        return next(self.__iter__())

    def __getitem__(self, key) -> "Symbol":
        """Get the item of the Symbol value with the specified key or index.

        If the Symbol value is a list, tuple, or numpy array, the key can be an integer or slice.
        If the Symbol value is a dictionary, the key can be a string or an integer.

        Args:
            key (Union[str, int, slice]): The key or index for the item in the Symbol value.

        Returns:
            Symbol: The item of the Symbol value with the specified key or index.

        Note:
            If the direct item retrieval fails, the method falls back to using the @core.getitem decorator, which retrieves and returns the item using core functions.
        """
        try:
            if (isinstance(key, int) or isinstance(key, slice)) and (isinstance(self.value, list) or isinstance(self.value, tuple) or isinstance(self.value, np.ndarray)):
                return self.value[key]
            elif (isinstance(key, str) or isinstance(key, int)) and isinstance(self.value, dict):
                return self.value[key]
        except:
            pass
        @core.getitem()
        def _func(_, index: str):
            pass
        return self._sym_return_type(_func(self, key))

    def __setitem__(self, key, value):
        """Set the item of the Symbol value with the specified key or index to the given value.

        If the Symbol value is a list, tuple, or numpy array, the key can be an integer or slice.
        If the Symbol value is a dictionary, the key can be a string or an integer.

        Args:
            key (Union[str, int, slice]): The key or index for the item in the Symbol value.
            value: The value to set the item to.

        Note:
            If the direct item setting fails, the method falls back to using the @core.setitem decorator, which sets the item using core functions.
        """
        try:
            if (isinstance(key, int) or isinstance(key, slice)) and (isinstance(self.value, list) or isinstance(self.value, tuple) or isinstance(self.value, np.ndarray)):
                self.value[key] = value
                return
            elif (isinstance(key, str) or isinstance(key, int)) and isinstance(self.value, dict):
                self.value[key] = value
                return
        except:
            pass
        @core.setitem()
        def _func(_, index: str, value: str):
            pass
        self.value = Symbol(_func(self, key, value)).value

    def __delitem__(self, key):
        """Delete the item of the Symbol value with the specified key or index.

        If the Symbol value is a dictionary, the key can be a string or an integer.

        Args:
            key (Union[str, int]): The key for the item in the Symbol value.

        Note:
            If the direct item deletion fails, the method falls back to using the @core.delitem decorator, which deletes the item using core functions.
        """
        try:
            if (isinstance(key, str) or isinstance(key, int)) and isinstance(self.value, dict):
                del self.value[key]
                return
        except:
            pass
        @core.delitem()
        def _func(_, index: str):
            pass
        self.value = Symbol(_func(self, key)).value

    def __neg__(self) -> "Symbol":
        """Return the negated value of the Symbol.

        The method uses the @core.negate decorator to compute the negation of the Symbol value.

        Returns:
            Symbol: The negated value of the Symbol.
        """
        @core.negate()
        def _func(_):
            pass
        return self._sym_return_type(_func(self))

    def __not__(self) -> "Symbol":
        """Return the negated value of the Symbol.

        The method uses the @core.negate decorator to compute the negation of the Symbol value.

        Returns:
            Symbol: The negated value of the Symbol.
        """
        @core.negate()
        def _func(_):
            pass
        return self._sym_return_type(_func(self))

    def __invert__(self) -> "Symbol":
        """Return the inverted value of the Symbol.

        The method uses the @core.invert decorator to compute the inversion of the Symbol value.

        Returns:
            Symbol: The inverted value of the Symbol.
        """
        @core.invert()
        def _func(_):
            pass
        return self._sym_return_type(_func(self))

    def __lshift__(self, information) -> "Symbol":
        """Add new information to the Symbol.

        The method uses the @core.include decorator to incorporate information into the Symbol.

        Args:
            information: The information to include in the Symbol.

        Returns:
            Symbol: The Symbol with the new information included.
        """
        @core.include()
        def _func(_, information: str):
            pass
        return self._sym_return_type(_func(self, information))

    def __rshift__(self, information) -> "Symbol":
        """Add new information to the Symbol.

        The method uses the @core.include decorator to incorporate information into the Symbol.

        Args:
            information: The information to include in the Symbol.

        Returns:
            Symbol: The Symbol with the new information included.
        """
        @core.include()
        def _func(_, information: str):
            pass
        return self._sym_return_type(_func(self, information))

    def __rrshift__(self, information) -> "Symbol":
        """Add new information to the Symbol.

        The method uses the @core.include decorator to incorporate information into the Symbol.

        Args:
            information: The information to include in the Symbol.

        Returns:
            Symbol: The Symbol with the new information included.
        """
        @core.include()
        def _func(_, information: str):
            pass
        return self._sym_return_type(_func(self, information))

    def __add__(self, other) -> "Symbol":
        """Combine the Symbol with another value.

        The method uses the @core.combine decorator to merge the Symbol and the other value.

        Args:
            other: The value to combine with the Symbol.

        Returns:
            Symbol: The Symbol combined with the other value.
        """
        @core.combine()
        def _func(_, a: str, b: str):
            pass
        return self._sym_return_type(_func(self, other))

    def __radd__(self, other) -> "Symbol":
        """Combine another value with the Symbol.

        The method uses the @core.combine decorator to merge the other value and the Symbol.

        Args:
            other: The value to combine with the Symbol.

        Returns:
            Symbol: The other value combined with the Symbol.
        """
        @core.combine()
        def _func(_, a: str, b: str):
            pass
        return self._sym_return_type(_func(other, self))

    def __iadd__(self, other) -> "Symbol":
        """Update the Symbol with the result of adding another value.

        This method adds another value to the Symbol and updates its value with the result.

        Args:
            other: The value to add to the Symbol.

        Returns:
            Symbol: The updated Symbol with the added value.
        """
        self.value = self.__add__(other)
        return self

    def __sub__(self, other) -> "Symbol":
        """Replace occurrences of a value with another value in the Symbol.

        The method uses the @core.replace decorator to replace occurrences of the other value with an empty string in the Symbol.

        Args:
            other: The value to replace in the Symbol.

        Returns:
            Symbol: The Symbol with occurrences of the other value replaced with an empty string.
        """
        @core.replace()
        def _func(_, text: str, replace: str, value: str):
            pass
        return self._sym_return_type(_func(self, other, ''))

    def __rsub__(self, other) -> "Symbol":
        """Subtracts the symbol value from another string and removes the substrings that match the symbol value.

        Using the core.replace decorator, this function creates a _func method to remove matching substrings.

        Args:
            other (str): The string to subtract the symbol value from.

        Returns:
            Symbol: A new symbol with the result of the subtraction.
        """
        @core.replace()
        def _func(_, text: str, replace: str, value: str):
            pass
        return self._sym_return_type(_func(other, self, ''))

    def __isub__(self, other) -> "Symbol":
        """In-place subtraction of the symbol value by the other symbol value.

        First, it calls the __sub__ method with the other symbol and then updates the current symbol's value with the result.

        Args:
            other (Symbol): The symbol to subtract from the current symbol.

        Returns:
            Symbol: The current symbol with the updated value.
        """
        val = self.__sub__(other)
        self.value = val.value
        return self

    def __and__(self, other) -> "Symbol":
        """Performs a logical AND operation between the symbol value and another.

        Uses the core.logic decorator with operator='and' to create a _func method for the AND operation.

        Args:
            other (str): The string to perform the AND operation with the symbol value.

        Returns:
            Symbol: A new symbol with the result of the AND operation.
        """
        @core.logic(operator='and')
        def _func(_, a: str, b: str):
            pass
        return self._sym_return_type(_func(self, other))

    def __or__(self, other) -> "Symbol":
        """Performs a logical OR operation between the symbol value and another.

        Uses the core.logic decorator with operator='or' to create a _func method for the OR operation.

        Args:
            other (str): The string to perform the OR operation with the symbol value.

        Returns:
            Symbol: A new symbol with the result of the OR operation.
        """
        @core.logic(operator='or')
        def _func(_, a: str, b: str):
            pass
        return self._sym_return_type(_func(self, other))

    def __xor__(self, other) -> "Symbol":
        """Performs a logical XOR operation between the symbol value and another.

        Uses the core.logic decorator with operator='xor' to create a _func method for the XOR operation.

        Args:
            other (str): The string to perform the XOR operation with the symbol value.

        Returns:
            Symbol: A new symbol with the result of the XOR operation.
        """
        @core.logic(operator='xor')
        def _func(_, a: str, b: str):
            pass
        return self._sym_return_type(_func(self, other))

    def __truediv__(self, other) -> "Symbol":
        """Divides the symbol value by another, splitting the symbol value by the other value.

        Args:
            other (str): The string to split the symbol value by.

        Returns:
            Symbol: A new symbol with the result of the division.
        """
        return self._sym_return_type(str(self).split(str(other)))

    def index(self, item: str, **kwargs) -> "Symbol":
        """Returns the index of a specified item in the symbol value.

        Uses the core.getitem decorator to create a _func method that finds the index of the item.

        Args:
            item (str): The item to find the index of within the symbol value.

        Returns:
            Symbol: A new symbol with the index of the specified item.
        """
        @core.getitem(**kwargs)
        def _func(_, item: str) -> int:
            pass
        return self._sym_return_type(_func(self, item))

    def equals(self, other: str, context: str = 'contextually', **kwargs) -> "Symbol":
        """Checks if the symbol value is equal to another string.

        Uses the core.equals decorator to create a _func method that checks for equality in a specific context.

        Args:
            other (str): The string to compare with the symbol value.
            context (str, optional): The context in which to compare the strings. Defaults to 'contextually'.

        Returns:
            Symbol: A new symbol indicating whether the two strings are equal or not.
        """
        @core.equals(context=context, **kwargs)
        def _func(_, other: str) -> bool:
            pass
        return self._sym_return_type(_func(self, other))

    def expression(self, expr: Optional[str] = None, expression_engine: str = None, **kwargs) -> "Symbol":
        """Evaluates an expression using the provided expression engine.

        Uses the core.expression decorator to create a _func method that evaluates the given expression.

        Args:
            expr (str, optional): The expression to evaluate. Defaults to the symbol value.
            expression_engine (str, optional): The expression engine to use for evaluation. Defaults to None.

        Returns:
            Symbol: A new symbol with the result of the expression evaluation.
        """
        if expr is None:
            expr = self.value
        @core.expression(expression_engine=expression_engine, **kwargs)
        def _func(_, expr: str):
            pass
        return self._sym_return_type(_func(self, expr))

    def clean(self, **kwargs) -> "Symbol":
        """Cleans the symbol value.

        Uses the core.clean decorator to create a _func method that cleans the symbol value.

        Returns:
            Symbol: A new symbol with the cleaned value.
        """
        @core.clean(**kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def summarize(self, context: Optional[str] = None, **kwargs) -> "Symbol":
        """Summarizes the symbol value.

        Uses the core.summarize decorator with an optional context to create a _func method that summarizes the symbol value.

        Args:
            context (str, optional): The context to be used for summarization. Defaults to None.

        Returns:
            Symbol: A new symbol with the summarized value.
        """
        @core.summarize(context=context, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def outline(self, **kwargs) -> "Symbol":
        """Creates an outline of the symbol value.

        Uses the core.outline decorator to create a _func method that generates an outline of the symbol value.

        Returns:
            Symbol: A new symbol with the outline of the value.
        """
        @core.outline(**kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def unique(self, keys: List[str] = [], **kwargs) -> "Symbol":
        """Extracts unique information from the symbol value, using provided keys.

        Uses the core.unique decorator with a list of keys to create a _func method that extracts unique data from the symbol value.

        Args:
            keys (List[str], optional): The list of keys to extract unique data. Defaults to [].

        Returns:
            Symbol: A new symbol with the unique information.
        """
        @core.unique(keys=keys, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def compose(self, **kwargs) -> "Symbol":
        """Composes a text based on the symbol value.

        Uses the core.compose decorator to create a _func method that composes a text using the symbol value.

        Returns:
            Symbol: A new symbol with the composed text.
        """
        @core.compose(**kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def filter(self, criteria: str, include: bool = False, **kwargs) -> "Symbol":
        """Filters the symbol value based on a specified criteria.

        Uses the core.filtering decorator with the provided criteria and include flag to create a _func method to filter the symbol value.

        Args:
            criteria (str): The criteria to filter the symbol value by.
            include (bool, optional): Whether to include or exclude items based on the criteria. Defaults to False.

        Returns:
            Symbol: A new symbol with the filtered value.
        """
        @core.filtering(criteria=criteria, include=include, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def modify(self, changes: str, **kwargs) -> "Symbol":
        """Modifies the symbol value based on the specified changes.

        Uses the core.modify decorator with the provided changes to create a _func method to modify the symbol value.

        Args:
            changes (str): The changes to apply to the symbol value.

        Returns:
            Symbol: A new symbol with the modified value.
        """
        @core.modify(changes=changes, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def replace(self, replace: str, value: str, **kwargs) -> "Symbol":
        """Replaces one value in the symbol value with another.

        Uses the core.replace decorator to create a _func method that replaces the values in the symbol value.

        Args:
            replace (str): The value to be replaced in the symbol value.
            value (str): The value to replace the existing value with.

        Returns:
            Symbol: A new symbol with the replaced value.
        """
        @core.replace(**kwargs)
        def _func(_, replace: str, value: str):
            pass
        return self._sym_return_type(_func(self, replace, value))

    def remove(self, information: str, **kwargs) -> "Symbol":
        """Removes a specified piece of information from the symbol value.

        Uses the core.replace decorator to create a _func method that removes the specified information.

        Args:
            information (str): The information to remove from the symbol value.

        Returns:
            Symbol: A new symbol with the removed information.
        """
        @core.replace(**kwargs)
        def _func(_, text: str, replace: str, value: str):
            pass
        return self._sym_return_type(_func(self, information, ''))

    def include(self, information: str, **kwargs) -> "Symbol":
        """Includes a specified piece of information in the symbol value.

        Uses the core.include decorator to create a _func method that includes the specified information.

        Args:
            information (str): The information to include in the symbol value.

        Returns:
            Symbol: A new symbol with the included information.
        """
        @core.include(**kwargs)
        def _func(_, information: str):
            pass
        return self._sym_return_type(_func(self, information))

    def combine(self, sym: str, **kwargs) -> "Symbol":
        """Combines the current symbol value with another string.

        Uses the core.combine decorator to create a _func method that combines the symbol value with another string.

        Args:
            sym (str): The string to combine with the symbol value.

        Returns:
            Symbol: A new symbol with the combined value.
        """
        @core.combine(**kwargs)
        def _func(_, a: str, b: str):
            pass
        return self._sym_return_type(_func(self, sym))

    def rank(self, measure: str = 'alphanumeric', order: str = 'desc', **kwargs) -> "Symbol":
        """Ranks the symbol value based on a measure and a sort order.

        Uses the core.rank decorator with the specified measure and order to create a _func method that ranks the symbol value.

        Args:
            measure (str, optional): The measure to rank the symbol value by. Defaults to 'alphanumeric'.
            order (str, optional): The sort order for ranking. Defaults to 'desc'.

        Returns:
            Symbol: A new symbol with the ranked value.
        """
        @core.rank(order=order, **kwargs)
        def _func(_, measure: str) -> str:
            pass
        return self._sym_return_type(_func(self, measure))

    def extract(self, pattern: str, **kwargs) -> "Symbol":
        """Extracts data from the symbol value based on a pattern.

        Uses the core.extract decorator with the specified pattern to create a _func method that extracts data from the symbol value.

        Args:
            pattern (str): The pattern to use for data extraction.

        Returns:
            Symbol: A new symbol with the extracted data.
        """
        @core.extract(**kwargs)
        def _func(_, pattern: str) -> str:
            pass
        return self._sym_return_type(_func(self, pattern))

    def analyze(self, exception: Exception, query: Optional[str] = '', **kwargs) -> "Symbol":
        """Uses the @core.analyze decorator, analyzes an exception and returns a symbol.

        Args:
            exception (Exception): The exception to be analyzed.
            query (str, optional): An additional query to provide context during analysis. Defaults to ''.

        Returns:
            Symbol: The analyzed result as a Symbol.
        """
        @core.analyze(exception=exception, query=query, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def correct(self, context: str, **kwargs) -> "Symbol":
        """Uses the @core.correct decorator, corrects the value of the symbol based on the given context.

        Args:
            context (str): The context used to correct the value of the symbol.

        Returns:
            Symbol: The corrected value as a Symbol.
        """
        @core.correct(context=context, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def translate(self, language: str = 'English', **kwargs) -> "Symbol":
        """Uses the @core.translate decorator to translate the symbol's value to the specified language.

        Args:
            language (str, optional): The language to translate the value to. Defaults to 'English'.

        Returns:
            Symbol: The translated value as a Symbol.
        """
        @core.translate(language=language, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def choice(self, cases: List[str], default: str, **kwargs) -> "Symbol":
        """Uses the @core.case decorator, selects one of the cases based on the symbol's value.

        Args:
            cases (List[str]): The list of possible cases.
            default (str): The default case if none of the cases match the symbol's value.

        Returns:
            Symbol: The chosen case as a Symbol.
        """
        @core.case(enum=cases, default=default, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def query(self, context: str, prompt: Optional[str] = None, examples = [], **kwargs) -> "Symbol":
        """Uses the @core.query decorator, queries based on the context, prompt, and examples.

        Args:
            context (str): The context used for the query.
            prompt (str, optional): The prompt for the query. Defaults to None.
            examples (List[str]): A list of examples to help guide the query. Defaults to [].

        Returns:
            Symbol: The result of the query as a Symbol.
        """
        @core.query(context=context, prompt=prompt, examples=examples, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def convert(self, format: str, **kwargs) -> "Symbol":
        """Uses the @core.convert decorator, converts the symbol's value to the specified format.

        Args:
            format (str): The format to convert the value to.

        Returns:
            Symbol: The converted value as a Symbol.
        """
        @core.convert(format=format, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def transcribe(self, modify: str, **kwargs) -> "Symbol":
        """Uses the @core.transcribe decorator, modifies the symbol's value based on the modify parameter.

        Args:
            modify (str): The modification to be applied to the value.

        Returns:
            Symbol: The modified value as a Symbol.
        """
        @core.transcribe(modify=modify, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def execute(self, **kwargs) -> "Symbol":
        """Executes the symbol's expression using the @core.execute decorator.

        Returns:
            Symbol: The result of the executed expression as a Symbol.
        """
        @core.execute(**kwargs)
        def _func(_):
            pass
        return _func(self)

    def fexecute(self, **kwargs) -> "Symbol":
        """Executes the symbol's expression using the fallback execute method (ftry).

        Returns:
            Symbol: The result of the executed expression as a Symbol.
        """
        def _func(sym: Symbol, **kargs):
            return sym.execute(**kargs)
        return self.ftry(_func, **kwargs)

    def simulate(self, **kwargs) -> "Symbol":
        """Uses the @core.simulate decorator, simulates the value of the symbol. Used for hypothesis testing or code simulation.

        Returns:
            Symbol: The simulated value as a Symbol.
        """
        @core.simulate(**kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))

    def sufficient(self, query: str, **kwargs) -> "Symbol":
        """Uses the @core.sufficient decorator, checks if the symbol's value is sufficient based on the query.

        Args:
            query (str): The query to verify if the symbol's value is sufficient.

        Returns:
            Symbol: The sufficiency result as a Symbol.
        """
        @core.sufficient(query=query, **kwargs)
        def _func(_) -> bool:
            pass
        return self._sym_return_type(_func(self))

    def list(self, condition: str, **kwargs) -> "Symbol":
        """Uses the @core.listing decorator, lists elements based on the condition.

        Args:
            condition (str): The condition to filter the elements in the list.

        Returns:
            Symbol: The filtered list as a Symbol.
        """
        @core.listing(condition=condition, **kwargs)
        def _func(_) -> list:
            pass
        return self._sym_return_type(_func(self))

    def contains(self, other, **kwargs) -> bool:
        """Uses the @core.contains decorator, checks whether the symbol's value contains the other element.

        Args:
            other: The element to be checked for containment.

        Returns:
            bool: True if the symbol's value contains the other element, False otherwise.
        """
        @core.contains(**kwargs)
        def _func(_, other) -> bool:
            pass
        return _func(self, other)

    def foreach(self, condition, apply, **kwargs) -> "Symbol":
        """Uses the @core.foreach decorator, iterates through the symbol's value and applies the provided function.

        Args:
            condition: The condition to filter the elements in the list.
            apply: The function to be applied to each element in the list.

        Returns:
            Symbol: The result of the iterative application of the function as a Symbol.
        """
        @core.foreach(condition=condition, apply=apply, **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))

    def map(self, **kwargs) -> "Symbol":
        """Transforms the keys of the dictionary value of a Symbol object to be unique.

        This function asserts that the Symbol's value is a dictionary and creates a new dictionary with the same values but unique keys. It is useful for ensuring that there are no duplicate keys in a dictionary.

        Args:
            **kwargs: Additional keyword arguments for the `unique` method.

        Returns:
            Symbol: A Symbol object with its value being the transformed dictionary with unique keys.

        Raises:
            AssertionError: If the Symbol's value is not a dictionary.
        """
        assert isinstance(self.value, dict), "Map can only be applied to a dictionary"
        map_ = {}
        keys = []
        for v in self.value.values():
            k = Symbol(v).unique(keys, **kwargs)
            keys.append(k.value)
            map_[k.value] = v
        return self._sym_return_type(map_)

    def dict(self, context: str, **kwargs) -> "Symbol":
        """Maps related content together under a common abstract topic as a dictionary of the Symbol value.

        This method uses the @core.dictionary decorator to apply the given context to the Symbol. It is useful for applying additional context to the symbol.

        Args:
            context (str): The context to apply to the Symbol.
            **kwargs: Additional keyword arguments for the @core.dictionary decorator.

        Returns:
            Symbol: A Symbol object with a dictionary applied.
        """
        @core.dictionary(context=context, **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))

    def relate(self, sym: "Symbol") -> "Symbol":
        """Relates the given Symbol to the current Symbol.

        This method appends the input Symbol to the current Symbol's relations list, creating a relationship between them. It is useful for tracking dependencies and relations between Symbol objects.

        Args:
            sym (Symbol): The Symbol to relate to the current Symbol.

        Returns:
            Symbol: The current Symbol object (self) with the relation added.
        """
        self.relations.append(sym)
        return self

    def template(self, template: str, placeholder = '{{placeholder}}', **kwargs) -> "Symbol":
        """Applies a template to the Symbol.

        This method uses the @core.template decorator to apply the given template and placeholder to the Symbol. It is useful for providing structure to the Symbol's value.

        Args:
            template (str): The template to apply to the Symbol.
            placeholder (str, optional): The placeholder in the template to be replaced with the Symbol's value. Defaults to '{{placeholder}}'.
            **kwargs: Additional keyword arguments for the @core.template decorator.

        Returns:
            Symbol: A Symbol object with a template applied.
        """
        @core.template(template=template, placeholder=placeholder, **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))

    def style(self, description: str, libraries = [], template: str = None, placeholder: str = '{{placeholder}}', **kwargs) -> "Symbol":
        """Applies a style to the Symbol.

        This method uses the @core.style decorator to apply the given style description, libraries, template, and placeholder to the Symbol. It is useful for providing structure and style to the Symbol's value.

        Args:
            description (str): The description of the style to apply.
            libraries (List, optional): A list of libraries that may be included in the style. Defaults to an empty list.
            template (str, optional): The template to apply, if any. Defaults to the Symbol's value.
            placeholder (str, optional): The placeholder in the template to be replaced with the Symbol's value. Defaults to '{{placeholder}}'.
            **kwargs: Additional keyword arguments for the @core.style decorator.

        Returns:
            Symbol: A Symbol object with the style applied.
        """
        if template is None:
            template = self.value
        @core.style(description=description, libraries=libraries, template=template, placeholder=placeholder, **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))

    def cluster(self, **kwargs) -> "Symbol":
        """Creates a cluster from the Symbol's value.

        This method uses the @core.cluster decorator to create a cluster from the Symbol's value. It is useful for grouping values in the Symbol.

        Args:
            **kwargs: Additional keyword arguments for the @core.cluster decorator.

        Returns:
            Symbol: A Symbol object with its value clustered.
        """
        @core.cluster(entries=self.value, **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))

    def embed(self, **kwargs) -> "Symbol":
        """Generates embeddings for the Symbol's value.

        This method uses the @core.embed decorator to generate embeddings for the Symbol's value. If the value is not a list, it is converted to a list.

        Args:
            **kwargs: Additional keyword arguments for the @core.embed decorator.

        Returns:
            Symbol: A Symbol object with its value embedded.
        """
        if not isinstance(self.value, list): self.value = [self.value]

        @core.embed(entries=self.value, **kwargs)
        def _func(_) -> list:
            pass
        return self._sym_return_type(_func(self))

    def similarity(self, other: 'Symbol', metric: str = 'cosine') -> float:
        """Calculates the similarity between two Symbol objects using a specified metric.

        This method compares the values of two Symbol objects and calculates their similarity according to the specified metric. It supports the 'cosine' metric, and raises a NotImplementedError for other metrics.

        Args:
            other (Symbol): The other Symbol object to calculate the similarity with.
            metric (str, optional): The metric to use for calculating the similarity. Defaults to 'cosine'.

        Returns:
            float: The similarity value between the two Symbol objects.

        Raises:
            TypeError: If any of the Symbol objects is not of type np.ndarray or Symbol.
            NotImplementedError: If the given metric is not supported.
        """
        def _ensure_format(x):
            if not isinstance(x, np.ndarray):
                if not isinstance(x, Symbol):
                    raise TypeError(f"Cannot compute similarity with type {type(x)}")
                x = np.array(x.value)
            return x.squeeze()[:, None]

        v = _ensure_format(self)
        o = _ensure_format(other)

        if metric == 'cosine':
            return (v.T@o / (v.T@v)**.5 * (o.T@o)**.5).item()
        else:
            raise NotImplementedError(f"Similarity metric {metric} not implemented. Available metrics: 'cosine'")

    def zip(self, **kwargs) -> List[Tuple[str, List, Dict]]:
        """Zips the Symbol's value with its embeddings and a query containing the value.

        This method zips the Symbol's value along with its embeddings and a query containing the value. It is useful for processing the Symbol's value further.

        Args:
            **kwargs: Additional keyword arguments for the `embed` method.

        Returns:
            List[Tuple[str, List, Dict]]: A list of tuples containing a unique ID, the value's embeddings, and a query containing the value.

        Raises:
            ValueError: If the Symbol's value is not a string.
        """
        if not isinstance(self.value, str):
            raise ValueError(f'Expected id to be a string, got {type(self.value)}')

        embeds = self.embed(**kwargs).value
        idx    = str(uuid.uuid4())
        query  = {'text': self.value}

        return list(zip([idx], embeds, [query]))

    def stream(self, expr: "Expression",
               max_tokens: int = 4000,
               char_token_ratio: float = 0.6,
               **kwargs) -> "Symbol":
        """Streams the Symbol's value through an Expression object.

        This method divides the Symbol's value into chunks and processes each chunk through the given Expression object. It is useful for processing large strings in smaller parts.

        Args:
            expr (Expression): The Expression object to evaluate the Symbol's chunks.
            max_tokens (int, optional): The maximum number of tokens allowed in a chunk. Defaults to 4000.
            char_token_ratio (float, optional): The ratio between characters and tokens for calculating max_chars. Defaults to 0.6.
            **kwargs: Additional keyword arguments for the given Expression.

        Returns:
            Symbol: A Symbol object containing the evaluated chunks.

        Raises:
            ValueError: If the Expression object exceeds the maximum allowed tokens.
        """
        max_chars = int(max_tokens * char_token_ratio)
        steps = (len(self)// max_chars) + 1
        for chunks in range(steps):
            # iterate over string in chunks of max_chars
            r = Symbol(str(self)[chunks * max_chars: (chunks + 1) * max_chars])
            size = max_tokens - len(r)

            # simulate the expression
            prev = expr(r, max_tokens=size, preview=True, **kwargs)
            prev = self._to_symbol(prev)
            # if the expression is too big, split it
            if len(prev) > max_tokens:
                # split
                r1_split = r.value[:len(r)//2]
                r = expr(r1_split, max_tokens=size, **kwargs)
                yield r
                r2_split = r.value[len(r)//2:]
                r = expr(r2_split, max_tokens=size, **kwargs)
            else:
                # run the expression
                r = expr(r, max_tokens=size, **kwargs)

            yield r

    def fstream(self, expr: "Expression",
                max_tokens: int = 4000,
                char_token_ratio: float = 0.6,
                **kwargs) -> "Symbol":
        """Streams the Symbol's value through an Expression object and returns a Symbol containing a list of processed chunks.

        This method is a wrapper around the `stream` method that returns a Symbol object containing a list of processed chunks instead of a generator.

        Args:
            expr (Expression): The Expression object to evaluate the Symbol's chunks.
            max_tokens (int, optional): The maximum number of tokens allowed in a chunk. Defaults to 4000.
            char_token_ratio (float, optional): The ratio between characters and tokens for calculating max_chars. Defaults to 0.6.
            **kwargs: Additional keyword arguments for the given Expression.

        Returns:
            Symbol: A Symbol object containing a list of processed chunks.
        """
        return self._sym_return_type(list(self.stream(expr, max_tokens, char_token_ratio, **kwargs)))

    def ftry(self, expr: "Expression", retries: int = 1, **kwargs) -> "Symbol":
        """Tries to evaluate a Symbol using a given Expression.

        This method evaluates a Symbol using a given Expression. If it fails, it retries the evaluation a specified number of times.

        Args:
            expr (Expression): The Expression object to evaluate the Symbol.
            retries (int, optional): The number of retries if the evaluation fails. Defaults to 1.
            **kwargs: Additional keyword arguments for the given Expression.

        Returns:
            Symbol: A Symbol object with the evaluated result.

        Raises:
            Exception: If the evaluation fails after all retries.
        """
        prompt = {'message': ''}
        def output_handler(input_):
            prompt['message'] = input_
        kwargs['output_handler'] = output_handler
        retry_cnt: int = 0
        sym = self
        while True:
            try:
                sym = expr(sym, **kwargs)
                retry_cnt = 0
                return sym
            except Exception as e:
                retry_cnt += 1
                if retry_cnt > retries:
                    raise e
                else:
                    err =  Symbol(prompt['message'])
                    res = err.analyze(query="What is the issue in this expression?", payload=sym, exception=e, max_tokens=2000)
                    sym = sym.correct(context=prompt['message'], exception=e, payload=res, max_tokens=2000)

    def expand(self, *args, **kwargs) -> "Symbol":
        """Expand the current Symbol and create a new sub-component.
        The function writes a self-contained function (with all imports) to solve a specific user problem task.

        This method uses the `@core.expand` decorator with a maximum token limit of 2048, and allows additional keyword
        arguments to be passed to the decorator.

        Returns:
            Symbol: The name of the newly created sub-component.
        """
        @core.expand(max_tokens=2048, **kwargs)
        def _func(_, *args):
            pass
        _tmp_llm_func = self._sym_return_type(_func(self, *args))
        func_name = str(_tmp_llm_func.extract('function name'))
        def _llm_func(*args, **kwargs):
            res = _tmp_llm_func.fexecute(*args, **kwargs)
            return res['locals'][func_name]()
        setattr(self, func_name, _llm_func)
        return func_name

    def save(self, path: str, replace: bool = False) -> "Symbol":
        """Save the current Symbol to a file.

        Args:
            path (str): The filepath of the saved file.
            replace (bool, optional): Whether to replace the file if it already exists. Defaults to False.

        Returns:
            Symbol: The current Symbol.
        """
        file_path = path
        if not replace:
            cnt = 0
            while os.path.exists(file_path):
                filename, file_extension = os.path.splitext(path)
                file_path = f'{filename}_{cnt}{file_extension}'
                cnt += 1
        with open(file_path, 'w') as f:
            f.write(str(self))
        return self

    def output(self, *args, **kwargs) -> "Symbol":
        """Output the current Symbol to a output handler.

        This method uses the `@core.output` decorator and allows additional keyword arguments to be passed to the decorator.

        Returns:
            Symbol: The resulting Symbol after the output operation.
        """
        @core.output(**kwargs)
        def _func(_, *args):
            pass
        return self._sym_return_type(_func(self, *args))


class Expression(Symbol):
    def __init__(self, value = None):
        """Create an Expression object that will be evaluated lazily using the forward method.

        Args:
            value (Any, optional): The value to be stored in the Expression object. Usually not provided as the value
                                    is computed using the forward method when called. Defaults to None.
        """
        super().__init__(value)

    @property
    def _sym_return_type(self):
        return Expression

    def __call__(self, *args, **kwargs) -> Symbol:
        """Evaluate the expression using the forward method and assign the result to the value attribute.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Symbol: The evaluated result of the forward method, stored as the value attribute.
        """
        self.value = self.forward(*args, **kwargs)
        return self.value

    def forward(self, *args, **kwargs) -> Symbol:
        """Needs to be implemented by subclasses to specify the behavior of the expression during evaluation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Symbol: The evaluated result of the implemented forward method.
        """
        raise NotImplementedError()

    def draw(self, operation: str = 'create', **kwargs) -> "Symbol":
        """Draw an image using the current Symbol as the base.

        Args:
            operation (str, optional): The operation to perform on the Symbol. Defaults to 'create'.
            **kwargs: Additional keyword arguments to be passed to the `@core.draw` decorator.

        Returns:
            Symbol: The resulting Symbol after the drawing operation.
        """
        @core.draw(operation=operation, **kwargs)
        def _func(_):
            pass
        return self._sym_return_type(_func(self))

    def input(self, message: str = "Please add more information", **kwargs) -> "Symbol":
        """Request user input and return a Symbol containing the user input.

        Args:
            message (str, optional): The message displayed to request the user input. Defaults to "Please add more information".
            **kwargs: Additional keyword arguments to be passed to the `@core.userinput` decorator.

        Returns:
            Symbol: The resulting Symbol after receiving the user input.
        """
        @core.userinput(**kwargs)
        def _func(_, message) -> str:
            pass
        return self._sym_return_type(_func(self, message))

    def fetch(self, url: str, pattern: str = '', **kwargs) -> "Symbol":
        """Fetch data from a URL and return a Symbol containing the fetched data.

        Args:
            url (str): The URL to fetch data from.
            pattern (str, optional): The pattern to extract specific data from the fetched content. Defaults to ''.
            **kwargs: Additional keyword arguments to be passed to the `@core.fetch` decorator.

        Returns:
            Symbol: The resulting Symbol after fetching data from the URL.
        """
        @core.fetch(url=url, pattern=pattern, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def ocr(self, image_url: str, **kwargs) -> "Symbol":
        """Perform OCR on an image using the image URL or image path.

        Args:
            image_url (str): The URL of the image to perform OCR on.
            **kwargs: Additional keyword arguments to be passed to the `@core.ocr` decorator.

        Returns:
            Symbol: The resulting Symbol after performing OCR on the image.
        """
        if not image_url.startswith('http'):
            image_url = f'file://{image_url}'
        @core.ocr(image=image_url, **kwargs)
        def _func(_) -> dict:
            pass
        return self._sym_return_type(_func(self))

    def vision(self, image: Optional[str] = None, text: Optional[List[str]] = None, **kwargs) -> "Symbol":
        """Perform a vision operation on an image using the image URL or image path.

        Args:
            image (str, optional): The image to use for the vision operation. Defaults to None.
            text (List[str], optional): A list of text prompts to guide the vision operation. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the `@core.vision` decorator.

        Returns:
            Symbol: The resulting Symbol after performing the vision operation.
        """
        @core.vision(image=image, prompt=text, **kwargs)
        def _func(_) -> np.ndarray:
            pass
        return self._sym_return_type(_func(self))

    def speech(self, audio_path: str, operation: str = 'decode', **kwargs) -> "Symbol":
        """Perform a speech operation on an audio file using the audio file path.

        Args:
            audio_path (str): The path of the audio file to perform the speech operation on.
            operation (str, optional): The operation to perform on the audio file (e.g., 'decode'). Defaults to 'decode'.
            **kwargs: Additional keyword arguments to be passed to the `@core.speech` decorator.

        Returns:
            Symbol: The resulting Symbol after performing the speech operation.
        """
        @core.speech(audio=audio_path, prompt=operation, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def search(self, query: str, **kwargs) -> "Symbol":
        """Search for information on the internet based on the query.

        Args:
            query (str): The query for the search operation.
            **kwargs: Additional keyword arguments to be passed to the `@core.search` decorator.

        Returns:
            Symbol: The resulting Symbol after performing the search operation.
        """
        @core.search(query=query, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def open(self, path: str, **kwargs) -> "Symbol":
        """Open a file and store its content in an Expression object as a string.

        Args:
            path (str): The path to the file that needs to be opened.
            **kwargs: Arbitrary keyword arguments to be used by the core.opening decorator.

        Returns:
            Symbol: An Expression object containing the content of the file as a string value.
        """
        @core.opening(path=path, **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def index(self, path: str, **kwargs) -> "Symbol":
        """Execute a configuration operation on the index.

        Args:
            path (str): Index configuration path.
            **kwargs: Arbitrary keyword arguments to be used by the core.index decorator.

        Returns:
            Symbol: An Expression object containing the configuration result.
        """
        @core.index(prompt=path, operation='config', **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def add(self, query: str, **kwargs) -> "Symbol":
        """Add an entry to the existing index.

        Args:
            query (str): The query string used to add an entry to the index.
            **kwargs: Arbitrary keyword arguments to be used by the core.index decorator.

        Returns:
            Symbol: An Expression object containing the addition result.
        """
        @core.index(prompt=query, operation='add', **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    def get(self, query: str, **kwargs) -> "Symbol":
        """Search the index based on the provided query.

        Args:
            query (str): The query string used to search entries in the index.
            **kwargs: Arbitrary keyword arguments to be used by the core.index decorator.

        Returns:
            Symbol: An Expression object containing the search result.
        """
        @core.index(prompt=query, operation='search', **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(self))

    @staticmethod
    def command(engines: List[str] = ['all'], **kwargs) -> "Symbol":
        """Execute command(s) on engines.

        Args:
            engines (List[str], optional): The list of engines on which to execute the command(s). Defaults to ['all'].
            **kwargs: Arbitrary keyword arguments to be used by the core.command decorator.

        Returns:
            Symbol: An Expression object representing the command execution result.
        """
        @core.command(engines=engines, **kwargs)
        def _func(_):
            pass
        return Expression(_func(Expression()))

    @staticmethod
    def setup(engines: Dict[str, Any], **kwargs) -> "Symbol":
        """Configure multiple engines.

        Args:
            engines (Dict[str, Any]): A dictionary containing engine names as keys and their configurations as values.
            **kwargs: Arbitrary keyword arguments to be used by the core.setup decorator.

        Returns:
            Symbol: An Expression object representing the setup result.
        """
        @core.setup(engines=engines, **kwargs)
        def _func(_):
            pass
        return Expression(_func(Expression()))
