import json
import copy
import numpy as np

from box import Box
from json import JSONEncoder
from typing import Any, Dict, Iterator, List, Optional, Type

from . import core
from .ops import SYMBOL_PRIMITIVES


class SymbolEncoder(JSONEncoder):
    def default(self, o):
        '''
        Encode a Symbol instance into its dictionary representation.

        Args:
            sym (Symbol): The Symbol instance to encode.

        Returns:
            dict: The dictionary representation of the Symbol instance.
        '''
        if isinstance(o, Symbol):
            return o.__getstate__()
        return JSONEncoder.default(self, o)


class Metadata(object):
    # create a method that allow to dynamically assign a attribute if not in __dict__
    # example: metadata = Metadata()
    # metadata.some_new_attribute = 'some_value'
    # metadata.some_new_attribute
    def __getattr__(self, name):
        '''
        Get a metadata attribute by name.

        Args:
            name (str): The name of the metadata attribute to get.

        Returns:
            Any: The value of the metadata attribute.
        '''
        return self.__dict__.get(name)

    def __setattr__(self, name, value):
        '''
        Set a metadata attribute by name.

        Args:
            name (str): The name of the metadata attribute to set.
            value (Any): The value of the metadata attribute.
        '''
        self.__dict__[name] = value


class SymbolMeta(type):
    """
    Metaclass to unify metaclasses of mixed-in primitives.
    """
    def __instancecheck__(cls, obj):
        if str(obj.__class__) == str(cls):
            return True
        return super().__instancecheck__(obj)

    def __new__(mcls, name, bases, attrs):
        """
        Create a new class with a unified metaclass.
        """
        # create a new cls type that inherits from Symbol and the mixin primitive types
        cls            = type.__new__(mcls, name, bases, attrs)
        return cls


class Symbol(metaclass=SymbolMeta):
    _mixin      = True
    _primitives = SYMBOL_PRIMITIVES
    _metadata   = Metadata()
    _dynamic_context: Dict[str, List[str]] = {}

    def __init__(self, *value, static_context: Optional[str] = '', **kwargs) -> None:
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
        self._value     = None
        self._metadata  = self._metadata # use global metadata by default
        self._parent    = None #@TODO: to enable graph construction
        self._children  = None #@TODO: to enable graph construction
        self._static_context = static_context

        if len(value) == 1:

            value = value[0]

            if isinstance(value, Symbol):
                self._value     = value._value
                self._metadata  = value._metadata
                self._parent    = value._parent
                self._children  = value._children

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

                self._value = value

            else:
                self._value = value

        elif len(value) > 1:
            self._value = [v.value if isinstance(v, Symbol) else v for v in value]

    def __new__(cls, *args, mixin: Optional[bool] = None, primitives: Optional[List[Type]] = None, **kwargs) -> "Symbol":
        '''
        Create a new Symbol instance.

        Args:
            *args: Variable length argument list.
            mixin (Optional[bool]): Whether to mix in the SymbolArithmeticPrimitives class. Defaults to None.
            primitives (Optional[List[Type]]): A list of primitive classes to mix in. Defaults to None.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Symbol: The new Symbol instance.
        '''
        use_mixin  = mixin if mixin is not None else cls._mixin
        primitives = primitives if primitives is not None else cls._primitives
        ori_cls    = cls
        # Initialize instance as a combination of Symbol and the mixin primitive types
        if use_mixin:
            # create a new cls type that inherits from Symbol and the mixin primitive types
            cls       = SymbolMeta(cls.__name__, (cls,) + tuple(primitives), {})
        obj = super().__new__(cls)
        return obj

    def __reduce__(self):
        '''
        This method is called by pickle to serialize the object.
        It returns a tuple that contains:
        - A callable object that when called produces a new object (e.g., the class of the object)
        - A tuple of arguments for the callable object
        - Optionally, the state which will be passed to the object’s `__setstate__` method

        Returns:
            tuple: A tuple containing the callable object, the arguments for the callable object, and the state of the object.
        '''
        # Get the state of the object
        state = self.__getstate__()

        # We create a simple tuple of primitives and their names to be able to pickle them.
        # Note: This assumes that the primitives are pickleable (it can be a limitation).
        primitives = [(primitive, primitive.__name__) for primitive in self._primitives]

        # Get the base class for reconstruction
        base_cls = self.__class__.__bases__[0]

        # The __reduce__ method returns:
        # - A callable object that when called produces a new object (e.g., the class of the object)
        # - A tuple of arguments for the callable object
        # - Optionally, the state which will be passed to the object’s `__setstate__` method
        return (self._reconstruct_class, (base_cls, self._mixin, primitives), state)

    def __reduce_ex__(self, protocol):
        return self.__reduce__()

    # This will be called by pickle with the info from __reduce__ to recreate the dynamic class
    @staticmethod
    def _reconstruct_class(base_cls, use_mixin, primitives_info):
        '''
        Reconstruct the class from the serialized state.

        Args:
            base_cls (Type): The base class of the Symbol.
            use_mixin (bool): Whether to mix in the SymbolArithmeticPrimitives class.
            primitives_info (List[Tuple[Type, str]]): A list of primitive classes and their names.

        Returns:
            Type: The reconstructed class.
        '''
        ori_cls = base_cls
        if use_mixin:
            # Convert primitive info tuples back to types
            primitives     = [primitive for primitive, name in primitives_info]
            # Create new cls with UnifiedMeta metaclass
            cls            = SymbolMeta(base_cls.__name__, (base_cls,) + tuple(primitives), {})
            obj            = cls()
            return obj
        return base_cls()

    def __getstate__(self) -> Dict[str, Any]:
        '''
        Get the state of the symbol for serialization.

        Returns:
            dict: The state of the symbol.
        '''
        state = vars(self).copy()
        state.pop('_metadata', None)
        state.pop('_parent', None)
        state.pop('_children', None)
        return state

    def __setstate__(self, state) -> None:
        '''
        Set the state of the symbol for deserialization.

        Args:
            state (dict): The state to set the symbol to.
        '''
        vars(self).update(state)
        self._metadata = Symbol._metadata
        self._parent   = None
        self._children = None

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

    @property
    def _symbol_type(self) -> "Symbol":
        '''
        Get the type of the Symbol instance.

        Returns:
            Symbol: The type of the Symbol instance.
        '''
        return type(self._to_symbol(None))

    def __hash__(self) -> int:
        '''
        Get the hash value of the symbol.

        Returns:
            int: The hash value of the symbol.
        '''
        return str(self.value).__hash__()

    @property
    def metadata(self) -> Dict[str, Any]:
        '''
        Get the metadata associated with the symbol.

        Returns:
            Dict[str, Any]: The metadata associated with the symbol.
        '''
        return self._metadata

    @property
    def value(self) -> Any:
        '''
        Get the value of the symbol.

        Returns:
            Any: The value of the symbol.
        '''
        return self._value

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

    def __len__(self) -> int:
        '''
        Get the length of the value of the Symbol.

        Returns:
            int: The length of the value of the Symbol.
        '''
        return len(self.value)

    @property
    def shape(self) -> tuple:
        '''
        Get the shape of the value of the Symbol.

        Returns:
            tuple: The shape of the value of the Symbol.
        '''
        return self.value.shape

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
        # class with full path
        class_ = self.__class__.__module__ + '.' + self.__class__.__name__
        hex_ = hex(id(self))
        return f'<class {class_} at {hex_}>(value={str(self.value)})'

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
        state = super().__getstate__().copy()
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

    def index(self, path: str, index_name: str, **kwargs) -> 'Symbol':
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

    def add(self, query: List[str], index_name: str, **kwargs) -> 'Symbol':
        '''
        Add an entry to the existing index.

        Args:
            query (List[str]): The query string used to add an entry to the index.
            **kwargs: Arbitrary keyword arguments to be used by the core.index decorator.

        Returns:
            Symbol: An Expression object containing the addition result.
        '''
        @core.index(prompt=query, index_name=index_name, operation='add', **kwargs)
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
        @core.index(prompt=query, index_name=index_name, operation='search', **kwargs)
        def _func(_):
            pass
        return _func(self)

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
    def register(engines: Dict[str, Any], **kwargs) -> 'Symbol':
        '''
        Configure multiple engines.

        Args:
            engines (Dict[str, Any]): A dictionary containing engine names as keys and their configurations as values.
            **kwargs: Arbitrary keyword arguments to be used by the core.register decorator.

        Returns:
            Symbol: An Expression object representing the register result.
        '''
        @core.register(engines=engines, **kwargs)
        def _func(_):
            pass
        return Expression(_func(Expression()))

    def copy(self) -> Any:
        '''
        Returns a deep copy of the own object.

        Returns:
            Any: A deep copy of the own object.
        '''
        return copy.deepcopy(self)


class Result(Expression):
    def __init__(self, value = None, *args, **kwargs):
        '''
        Create a Result object that stores the results operations, including the raw result, value and metadata, if any.

        Args:
            value (Any, optional): The value to be stored in the Expression object. Usually not provided as the value
                                   is computed using the forward method when called. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        '''
        super().__init__(value) # value is the same as raw when initialized, however, it can be changed later
        self._sym_return_type = type(self)
        try:
            # try to make the values easily accessible
            self.raw              = Box(value)
        except:
            # otherwise, store the unprocessed view
            self.raw              = value

    @property
    def value(self) -> Any:
        '''
        Get the value of the symbol.

        Returns:
            Any: The value of the symbol.
        '''
        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        '''
        Set the value of the Result object.

        Args:
            value (Any): The value to set the Result object to.
        '''
        self._value = value