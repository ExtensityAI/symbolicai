import json
import copy
import html
import numpy as np

from box import Box
from json import JSONEncoder
from typing import Any, Dict, Iterator, List, Optional, Type, Callable, Tuple

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

    def __delattr__(self, name):
        '''
        Delete a metadata attribute by name.

        Args:
            name (str): The name of the metadata attribute to delete.
        '''
        del self.__dict__[name]

    def __getitem__(self, name):
        '''
        Get a metadata attribute by name.

        Args:
            name (str): The name of the metadata attribute to get.

        Returns:
            Any: The value of the metadata attribute.
        '''
        return self.__getattr__(name)

    def __setitem__(self, name, value):
        '''
        Set a metadata attribute by name.

        Args:
            name (str): The name of the metadata attribute to set.
            value (Any): The value of the metadata attribute.
        '''
        self.__setattr__(name, value)

    def __delitem__(self, name):
        '''
        Delete a metadata attribute by name.

        Args:
            name (str): The name of the metadata attribute to delete.
        '''
        self.__delattr__(name)

    def __str__(self) -> str:
        '''
        Get the string representation of the Symbol's value.

        Returns:
            str: The string representation of the Symbol's value.
        '''
        _val = ''
        if self.value is not None:
            _val += str(self.value)
        return _val + f"Properties({str({k: str(v) for k, v in self.__dict__.items() if not k.startswith('_')})})"

    def __repr__(self) -> str:
        '''
        Get the representation of the Symbol object as a string.

        Returns:
            str: The representation of the Symbol object.
        '''
        # class with full path
        class_ = self.__class__.__module__ + '.' + self.__class__.__name__
        hex_   = hex(id(self))
        from_symbol =  f' from {self.symbol_type.__module__}.{self.symbol_type.__name__}' if self.symbol_type else ''
        return f'<class {class_} at {hex_}{from_symbol}>'


class Linker(Metadata):
    def keys(self) -> List[str]:
        '''
        Get all keys of the linker.

        Returns:
            List[str]: All keys of the linker.
        '''
        return list(self.results.keys())

    def values(self) -> List[Any]:
        '''
        Get all values of the linker.

        Returns:
            List[Any]: All values of the linker.
        '''
        return list(self.results.values())

    def find(self, name: str, single: bool = True, strict: bool = False) -> Any:
        '''
        Find a result in the linker.

        Args:
            name (str): The name of the result to find.
            single (bool): Whether to return a single result or a list of results. Defaults to True.
            strict (bool): Whether to match the name exactly or not. Defaults to False.

        Returns:
            Any: The result.
        '''
        # search all results and return the first one that matches the name
        res = []
        for k in list(self.results.keys()):
            match_ = lambda k, name: str(name).lower() in str(k).lower() if not strict else str(name) == str(k)
            if match_(k, name):
                res.append(self.results[k])
        if single:
            assert len(res) == 1, f'Found {len(res)} results for name {name}. Expected 1.'
        if len(res) == 0:
            return None
        elif len(res) == 1:
            return res[0]
        return res


class SymbolMeta(type):
    """
    Metaclass to unify metaclasses of mixed-in primitives.
    """
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__(*args, **kwargs)
        return obj

    def __instancecheck__(cls, obj):
        if str(obj.__class__) == str(cls):
            return True
        return super().__instancecheck__(obj)

    def __new__(mcls, name, bases, attrs):
        """
        Create a new class with a unified metaclass.
        """
        # create a new cls type that inherits from Symbol and the mixin primitive types
        cls = type.__new__(mcls, name, bases, attrs)
        # inherit the base class module for dynamic type creation
        if '__module__' in attrs:
            cls.__module__ = attrs['__module__']
        elif len(bases) > 0:
            cls.__module__ = bases[0].__module__
        return cls


class Symbol(metaclass=SymbolMeta):
    _mixin                = True
    _primitives           = SYMBOL_PRIMITIVES
    _metadata             = Metadata()
    _metadata._primitives = {}
    _dynamic_context: Dict[str, List[str]] = {}

    def __init__(self, *value, static_context: Optional[str] = '', dynamic_context: Optional[str] = None, **kwargs) -> None:
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
        # store kwargs for new symbol instance type passing
        self._kwargs    = {
            'static_context': static_context,
            **kwargs
        }
        self._metadata  = Metadata() # use global metadata by default
        self._metadata.symbol_type = type(self)
        self._parent    = None
        self._children  = []
        self._static_context  = static_context
        self._dynamic_context = dynamic_context or Symbol._dynamic_context
        # if value is a single value, unwrap it
        _value          = self._unwrap_symbols_args(*value)
        self._value     = _value
        # construct dependency graph for symbol
        self._construct_dependency_graph(*value)

    def __post_init__(self, *args, **kwargs): # this is called at the end of __init__
        '''
        Post-initialization method that is called at the end of the __init__ method.
        '''
        def _func(k, v):
            # check if property is of type Symbol and not private and a class variable (not a function)
            if isinstance(v, Symbol) and not k.startswith('_') and not v is self:
                v._parent = self
                self._children.append(v)
            # else if iterable, check if it contains symbols
            elif (isinstance(v, list) or isinstance(v, tuple)) and not k.startswith('_'):
                for i in v:
                    _func(k, i)

        # analyze all self. properties if they are of type Symbol and add their parent and root
        for k, v in self.__dict__.items():
            _func(k, v)

    def _unwrap_symbols_args(self, *args, nested: bool = False) -> Any:
        if len(args) == 0:
            return None
        # check if args is a single value
        elif len(args) == 1:
            # get the first args value
            value = args[0]

            # if the value is a primitive, store it as is
            if isinstance(value, str) or isinstance(value, int) or isinstance(value, float) or isinstance(value, bool):
                pass

            # if the value is a symbol, unwrap it
            elif isinstance(value, Symbol):
                # if not nested, copy the symbol's value, metadata, parent, and children
                if not nested:
                    self._metadata       = value.metadata
                    self._static_context = value.static_context
                    self._kwargs         = value._kwargs
                # unwrap the symbol's value
                value                    = value.value

            # if the value is a list, tuple, dict, or set, unwrap all nested symbols
            elif isinstance(value, list) or isinstance(value, dict) or \
                    isinstance(value, set) or isinstance(value, tuple):

                if isinstance(value, list):
                    value = [self._unwrap_symbols_args(v, nested=True) for v in value]

                elif isinstance(value, dict):
                    value = {self._unwrap_symbols_args(k, nested=True): self._unwrap_symbols_args(v, nested=True) for k, v in value.items()}

                elif isinstance(value, set):
                    value = {self._unwrap_symbols_args(v, nested=True) for v in value}

                elif isinstance(value, tuple):
                    value = tuple([self._unwrap_symbols_args(v, nested=True) for v in value])

            return value

        elif len(args) > 1:
            return [self._unwrap_symbols_args(a, nested=True) if isinstance(a, Symbol) else a for a in args]

    def _construct_dependency_graph(self, *value):
        '''
        Construct a dependency graph for the symbol.

        Args:
            value (Any): The value of the symbol.
        '''
        # for each value
        for v in value:
            if isinstance(v, Symbol) and not v is self:
                # new instance becomes child of previous instance
                v._parent = self
                # add new instance to children of previous instance
                self._children.append(v)

    def __new__(cls, *args,
                mixin: Optional[bool] = None,
                primitives: Optional[List[Type]] = None,
                callables: Optional[List[Tuple[str, Callable]]] = None,
                only_nesy: bool = False,
                iterate_nesy: bool = False,
                **kwargs) -> "Symbol":
        '''
        Create a new Symbol instance.

        Args:
            *args: Variable length argument list.
            mixin (Optional[bool]): Whether to mix in the SymbolArithmeticPrimitives class. Defaults to None.
            primitives (Optional[List[Type]]): A list of primitive classes to mix in. Defaults to None.
            callables (Optional[List[Callable]]): A list of dynamic primitive functions to mix in. Defaults to None.
            only_nesy (bool): Whether to only use neuro-symbolic function or first check for type specific shortcut and the neuro-symbolic function. Defaults to False.
            iterate_nesy (bool): Whether to allow to iterate over iterables for neuro-symbolic values. Defaults to False.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Symbol: The new Symbol instance.
        '''
        use_mixin           = mixin if mixin is not None else cls._mixin
        standard_primitives = primitives is None
        primitives          = primitives if not standard_primitives else cls._primitives
        if not isinstance(primitives, list):
            primitives = [primitives]
        # Initialize instance as a combination of Symbol and the mixin primitive types
        if use_mixin:
            # create a new cls type that inherits from Symbol and the mixin primitive types
            cls       = SymbolMeta(cls.__name__, (cls,) + tuple(primitives), {})
        obj = super().__new__(cls)
        # store to inherit when creating new instances
        obj._kwargs = {
            'mixin': use_mixin,
            'primitives': primitives,
            'callables': callables,
            'only_nesy': only_nesy,
            'iterate_nesy': iterate_nesy,
            **kwargs
        }
        # configure standard primitives
        if use_mixin and standard_primitives:
            # disable shortcut matches for all primitives
            if only_nesy:
                obj.__disable_shortcut_matches__ = True
            # allow to iterate over iterables for neuro-symbolic values
            if iterate_nesy:
                obj.__nesy_iteration_primitives__ = True
        # If metatype has additional runtime primitives, add them to the instance
        if Symbol._metadata._primitives is not None:
            for prim_name in list(Symbol._metadata._primitives.keys()):
                # create a new function that binds the instance to the callable
                setattr(obj, prim_name, Symbol._metadata._primitives[prim_name](obj))
        # If has additional runtime callables, add them to the instance
        if callables is not None:
            if not isinstance(callables, list):
                callables = [callables]
            for call_name, call_func in callables:
                # create a new function that binds the instance to the callable
                setattr(obj, call_name, call_func(obj))
        return obj

    def __getattr__(self, name: str) -> Any:
        '''
        Get the attribute of the Symbol's value with the specified name or the attribute of the Symbol value with the specified name.

        Args:
            name (str): The name of the attribute to get from the Symbol's value.

        Returns:
            Any: The attribute of the Symbol's value with the specified name.
        '''
        try:
            # try to get attribute from current instance
            if name in self.__dict__:
                return self.__dict__[name]
            value = self.value if self.value is not None else None
            if isinstance(value, Exception):
                raise value
            raise AttributeError(f'<class \'{self.__class__.__name__}\'> or nested value of {str(type(value))} have no attribute \'{name}\'')
        except AttributeError as ex:
            # if has attribute and is public function
            if hasattr(self.value, name) and not name.startswith('_'):
                return getattr(self.value, name)
            raise ex

    def __array__(self, dtype=None):
        '''
        Get the numpy array representation of the Symbol's value.

        Returns:
            np.ndarray: The numpy array representation of the Symbol's value.
        '''
        return self.embedding.astype(dtype, copy=False)

    def __buffer__(self, flags=0):
        '''
        Get the buffer of the Symbol's value.

        Args:
            flags (int, optional): The flags for the buffer. Defaults to 0.

        Returns:
            memoryview: The buffer of the Symbol's value.
        '''
        return memoryview(self.embedding)

    @staticmethod
    def symbols(*values) -> List["Symbol"]:
        '''
        Create a list of Symbol instances from a list of values.

        Args:
            values (List[Any]): The list of values to create Symbol instances from.

        Returns:
            List[Symbol]: The list of Symbol instances.
        '''
        return [Symbol(value) for value in values]

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
        self._metadata   = Metadata()
        self._metadata.symbol_type = type(self)
        self._kwargs     = self._kwargs
        self._parent     = None
        self._children   = []

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

    def _to_symbol(self, value: Any, **kwargs) -> "Symbol":
        '''
        Convert a value to a Symbol instance.

        Args:
            value (Any): The value to convert to a Symbol instance.

        Returns:
            Symbol: The Symbol instance.
        '''
        if isinstance(value, Symbol):
            return value
        # inherit kwargs for new symbol instance
        kwargs = {**self._kwargs, **kwargs}
        sym    = Symbol(value, **kwargs)
        return sym

    @property
    def _symbol_type(self) -> "Symbol":
        '''
        Get the type of the Symbol instance.

        Returns:
            Symbol: The type of the Symbol instance.
        '''
        return Symbol

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
        return f'{self._static_context}' if self._static_context else ''

    @static_context.setter
    def static_context(self, value: str):
        '''
        Set the static context of the symbol which is defined by the user when creating a symbol subclass.
        '''
        self._static_context = value

    @property
    def dynamic_context(self) -> str:
        '''
        Get the dynamic context which is defined by the user at runtime.
        It helps to alter the behavior of the symbol at runtime.

        Returns:
            str: The dynamic context associated with this symbol type.
        '''
        # if dynamic context is manually set to a string, return it
        if isinstance(self._dynamic_context, str):
            return self._dynamic_context
        type_ = str(type(self))
        if type_ not in Symbol._dynamic_context:
            Symbol._dynamic_context[type_] = []
            return ''
        dyn_ctxt = Symbol._dynamic_context[type_]
        if len(dyn_ctxt) == 0:
            return ''
        sym_val = [str(v.value) if isinstance(v, Symbol) else str(v) for v in dyn_ctxt]
        val = '\n'.join(sym_val)
        return f'\n{val}' if val else ''

    @property
    def root(self) -> "Symbol":
        '''
        Get the root of the symbol.

        Returns:
            Symbol: The root of the symbol.
        '''
        root = self
        while root.parent is not None:
            root = root.parent
        return root

    @property
    def nodes(self) -> List["Symbol"]:
        '''
        Get all nodes descending recursively from the symbol.

        Returns:
            List[Symbol]: All nodes of the symbol.
        '''
        def _func(node, nodes):
            nodes.append(node)
            for child in node.children:
                _func(child, nodes)

        nodes = []
        _func(self, nodes)
        return nodes

    @property
    def edges(self) -> List[tuple]:
        '''
        Get all edges descending recursively from the symbol.

        Returns:
            List[tuple]: All edges of the symbol.
        '''
        def _func(node, edges):
            for child in node.children:
                edges.append((node, child))
                _func(child, edges)

        edges = []
        _func(self, edges)
        return edges

    @property
    def graph(self) -> (List["Symbol"], List[tuple]):
        '''
        Get the graph representation of the symbol.

        Returns:
            List[Symbol], List[tuple]: The nodes and edges of the symbol.
        '''
        return self.nodes, self.edges

    @property
    def linker(self) -> List["Symbol"]:
        '''
        Returns the link object metadata by descending recursively from the root of the symbol to the root_link object.

        Returns:
            List[Symbol]: All results of the symbol.
        '''
        return self.root.metadata.root_link

    @property
    def parent(self) -> "Symbol":
        '''
        Get the parent of the symbol.

        Returns:
            Symbol: The parent of the symbol.
        '''
        return self._parent

    @property
    def children(self) -> List["Symbol"]:
        '''
        Get the children of the symbol.

        Returns:
            List[Symbol]: The children of the symbol.
        '''
        return self._children

    def _root_link(self, sym: Any, **kwargs) -> Any:
        '''
        Call the forward method and assign the result to the graph value attribute.

        Args:
            res (Any): The result of the forward method.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the forward method.
        '''
        # transport results to the root node for global access
        if not self is self.root and not self.metadata.detach:
            ref = self.root.metadata
            if ref.root_link is None:
                ref.root_link = Linker()
            if ref.root_link.results is None:
                ref.root_link.results = {}
            prev = None
            if len(ref.root_link.results) > 0:
                prev = list(ref.root_link.results.values())[-1] # get previous result
            # create new symbol to avoid circular references
            res_ = Symbol(sym)
            if prev is not None and not prev is res_.root:
                prev.children.append(res_.root)
                res_.root._parent = prev
            ref.root_link.results[self.__repr__()] = res_

    def adapt(self, context: str, types: List[Type] = []) -> None:
        '''
        Update the dynamic context with a given runtime context.

        Args:
            context (str): The context to be added to the dynamic context.
            type (Type): The type used to update the dynamic context

        '''
        if not isinstance(types, list):
            types = [types]
        if len(types) == 0:
            types = [type(self)]

        for type_ in types:
            type_ = str(type_)
            if type_ not in Symbol._dynamic_context:
                Symbol._dynamic_context[type_] = []

            Symbol._dynamic_context[type_].append(str(context))

    def clear(self, types: List[Type] = []) -> None:
        '''
        Clear the dynamic context associated with this symbol type.
        '''
        if not isinstance(types, list):
            types = [types]
        if len(types) == 0:
            types = [type(self)]

        for type_ in types:
            type_ = str(type_)
            if type_ not in Symbol._dynamic_context:
                Symbol._dynamic_context[type_] = []
                return

            Symbol._dynamic_context[type_].clear()

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

    def __repr__(self, simplified: bool = False) -> str:
        '''
        Get the representation of the Symbol object as a string.

        Returns:
            str: The representation of the Symbol object.
        '''
        # class with full path
        class_ = self.__class__.__module__ + '.' + self.__class__.__name__
        hex_   = hex(id(self))
        val    = str(self.value)
        # only show first n characters of value and then add '...' and the last x characters
        if len(val) > 50:
            val = val[:25] + ' ... ' + val[-20:]
        return f'<class {class_} at {hex_}>(value={val})' if not simplified else f'{class_} at {hex_}'

    def _repr_html_(self) -> str:
        '''
        Get the HTML representation of the Symbol's value.

        Returns:
            str: The HTML representation of the Symbol's value.
        '''
        return html.escape(self.__repr__())

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

    def primitive(self, name: str, callable: callable) -> None:
        '''
        Set a primitive function to the Symbol instance.

        Args:

            callable (callable): The primitive function to set.
            scope (Union['instance', 'type', 'class'], optional): The scope of the primitive function. Defaults to 'instance'.

        Args:
            callable (callable): The primitive function to set.
        '''
        def _func(*args, **kwargs):
            return callable(self, *args, **kwargs)
        setattr(self, name, _func)

    @staticmethod
    def _global_primitive(name: str, callable: callable) -> None:
        '''
        Set a primitive function to the Symbol class.

        Args:
            callable (callable): The primitive function to set.
        '''
        def _func(obj):
            return lambda *args, **kwargs: callable(obj, *args, **kwargs)
        Symbol._metadata._primitives[name] = _func


# TODO: Workaround for Python bug to enable runtime assignment of lambda function to new Symbol objects.
# Currently creating multiple lambda functions within class __new__ definition only links last lambda function to all new Symbol attribute assignments.
# Need to contact Python developers to fix this bug.
class Call(object):
    def __new__(self, name, callable: Callable) -> Any:
        '''
        Prepare a callable for use in a Symbol instance.

        Args:
            callable (Callable): The callable to prepare.

        Returns:
            Callable: The prepared callable.
        '''
        def _func(obj):
            return lambda *args, **kwargs: callable(obj, *args, **kwargs)
        return (name, _func)


class GlobalSymbolPrimitive(object):
    def __new__(self, name, callable: Callable) -> Any:
        '''
        Prepare a callable for use in a Symbol instance.

        Args:
            callable (Callable): The callable to prepare.

        Returns:
            Callable: The prepared callable.
        '''
        Symbol._global_primitive(name, callable)


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
        super().__init__(value, *args, **kwargs)
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
        # evaluate the expression
        res = self.forward(*args, **kwargs)
        # store the result in the root node and link it to the previous result
        self._root_link(res, **kwargs)
        return res

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

    @staticmethod
    def prompt(message: str, **kwargs) -> 'Symbol':
        '''
        General raw input prompt method.

        Args:
            message (str): The prompt message for describing the task.
            **kwargs: Arbitrary keyword arguments to be used by the core.prompt decorator.

        Returns:
            Symbol: An Expression object representing the prompt result.
        '''
        @core.prompt(message=message, **kwargs)
        def _func(_):
            pass
        return Expression(_func(None))


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
        super().__init__(value, **kwargs) # value is the same as raw when initialized, however, it can be changed later
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