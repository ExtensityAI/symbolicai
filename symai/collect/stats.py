import json
import re
from collections.abc import Callable
from json import JSONEncoder
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch

from ..ops.primitives import OperatorPrimitives
from ..symbol import Symbol
from ..utils import UserMessage

SPECIAL_CONSTANT = '__aggregate_'
EXCLUDE_LIST     = ['_ipython_canary_method_should_not_exist_', '__custom_documentations__']


def _normalize_name(name: str) -> str:
    # Replace any character that is not a letter or a number with an underscore
    normalized_name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    return normalized_name.lower()


class AggregatorJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # drop active from state
        if isinstance(obj, Aggregator):
            state = obj.__dict__.copy()
            state.pop('_raise_error', None)
            state.pop('_active', None)
            state.pop('_finalized', None)
            state.pop('_map', None)
            # drop everything that starts with SPECIAL_CONSTANT
            for key in list(state.keys()):
                if  (not key.startswith(SPECIAL_CONSTANT) and key != '_value') or \
                    (key == '_value' and obj._value == []) or \
                    key.replace(SPECIAL_CONSTANT, '') in EXCLUDE_LIST:
                    state.pop(key, None)
            return state
        return obj.__dict__


class Aggregator(Symbol):
    def __init__(self,
                 value: Union["Aggregator", Symbol] | None = None,
                 path: str | None = None,
                 active: bool = True,
                 raise_error: bool = False,
                 *args, **kwargs):
        super().__init__(*args,
                         **kwargs)
        # disable nesy engine to avoid side effects
        self.__disable_nesy_engine__ = True
        if value is not None and isinstance(value, Symbol):
            # use this to avoid recursion on map setter
            self._value = value._value
            if isinstance(self._value, np.ndarray):
                self._value = self._value.tolist()
            elif isinstance(self._value, torch.Tensor):
                self._value = self._value.detach().cpu().numpy().tolist()
            elif not isinstance(self._value, (list, tuple)):
                self._value = [self._value]
        elif value is not None:
            UserMessage(f'Aggregator object must be of type Aggregator or Symbol! Got: {type(value)}', raise_with=Exception)
        else:
            self._value = []
        self._raise_error   = raise_error
        self._active    = active
        self._finalized = False
        self._map       = None
        self._path      = path

    def __new__(cls, *args,
            mixin: bool | None = None,
            primitives: list[type] | None = None, # only inherit arithmetic primitives
            callables: list[tuple[str, Callable]] | None = None,
            semantic: bool = False,
            **kwargs) -> "Symbol":
        if primitives is None:
            primitives = [OperatorPrimitives]
        return super().__new__(cls, *args,
                         mixin=mixin,
                         primitives=primitives,
                         callables=callables,
                         semantic=semantic,
                         **kwargs)

    def __getattr__(self, name):
        # replace name special characters and spaces with underscores
        name = _normalize_name(name)
        # Dynamically create new aggregator instance if it does not exist
        if self._active and name not in self.__dict__:
            aggregator = Aggregator(path=name)
            aggregator._parent = self
            self._children.append(aggregator)
            # create a new aggregate aggregator
            # named {SPECIAL_CONSTANT}{name} for automatic aggregation
            self.__dict__[f'{SPECIAL_CONSTANT}{name}'] = aggregator
            # add also a property with the same name but without the SPECIAL_CONSTANT prefix as a shortcut
            self.__dict__[name] = self.__dict__[f'{SPECIAL_CONSTANT}{name}']
            return self.__dict__.get(name)
        if not self._active and name not in self.__dict__:
            UserMessage(f'Aggregator object is frozen! No attribute {name} found!', raise_with=Exception)
        return self.__dict__.get(name)

    def __setattr__(self, name, value):
        # replace name special characters and spaces with underscores
        name = _normalize_name(name)
        return super().__setattr__(name, value)

    def __delattr__(self, name):
        # replace name special characters and spaces with underscores
        name = _normalize_name(name)
        return super().__delattr__(name)

    def __getitem__(self, name):
        # replace name special characters and spaces with underscores
        name = _normalize_name(name)
        return self.__getattr__(name)

    def __setitem__(self, name, value):
        # replace name special characters and spaces with underscores
        name = _normalize_name(name)
        return self.__setattr__(name, value)

    def __delitem__(self, name):
        # replace name special characters and spaces with underscores
        name = _normalize_name(name)
        return self.__delattr__(name)

    def __setstate__(self, state):
        # replace name special characters and spaces with underscores
        # drop active from state
        state.pop('_raise_error', None)
        state.pop('_active', None)
        state.pop('_finalized', None)
        state.pop('_map', None)
        return super().__setstate__(state)

    @staticmethod
    def _set_values(obj, dictionary, parent, strict: bool = True):
        # recursively reconstruct the object
        for key, value in dictionary.items():
            attr_key = key
            attr_value = value
            if isinstance(attr_value, dict):
                if parent is not None:
                    obj._path = attr_key
                attr_value = Aggregator._reconstruct(attr_value, parent=parent, strict=strict)
            if attr_key.startswith(SPECIAL_CONSTANT):
                attr_key = attr_key.replace(SPECIAL_CONSTANT, '')
            if attr_key == '_value':
                try:
                    attr_value = np.asarray(attr_value, dtype=np.float32)
                except Exception as e:
                    if strict:
                        msg = f'Could not set value of Aggregator object: {obj.path}! ERROR: {e}'
                        UserMessage(msg)
                        raise Exception(msg) from e
            obj.__setattr__(attr_key, attr_value)

    @staticmethod
    def _reconstruct(dictionary, parent = None, strict: bool = True):
        obj = Aggregator()
        obj._parent = parent
        if parent is not None:
            parent._children.append(obj)
        Aggregator._set_values(obj, dictionary, parent=obj, strict=strict)
        return obj

    def __str__(self) -> str:
        '''
        Get the string representation of the Symbol object.

        Returns:
            str: The string representation of the Symbol object.
        '''
        return str(self.entries)

    def _to_symbol(self, other) -> Symbol:
        sym = super()._to_symbol(other)
        res = Aggregator(sym)
        res._parent = self
        self._children.append(res)
        return

    @property
    def path(self) -> str:
        path = ''
        obj  = self
        while obj is not None:
            if obj._path is not None:
                path = obj._path.replace(SPECIAL_CONSTANT, '') + '.' + path
            obj = obj._parent
        return path[:-1] # remove last dot

    def __or__(self, other: Any) -> Any:
        self.add(other)
        return other

    def __ror__(self, other: Any) -> Any:
        self.add(other)
        return other

    def __ior__(self, other: Any) -> Any:
        self.add(other)
        return other

    def __len__(self) -> int:
        return len(self._value)

    @property
    def entries(self):
        return self._value

    @property
    def value(self):
        if self.map is not None:
            return np.asarray(self.map(np.asarray(self._value, dtype=np.float32)))
        return np.asarray(self._value, dtype=np.float32)

    @property
    def map(self):
        return self._map if not self.empty() else None

    @map.setter
    def map(self, value):
        self._set_map_recursively(value)

    def _set_map_recursively(self, map):
        self._map = map
        for key, value in self.__dict__.items():
            if isinstance(value, Aggregator) and (not key.startswith('_') or key.startswith(SPECIAL_CONSTANT)):
                value.map = map

    def shape(self):
        if len(self.entries) > 0:
            return np.asarray(self.entries).shape
        return ()

    def serialize(self):
        return json.dumps(self, cls=AggregatorJSONEncoder)

    def save(self, path: str):
        with Path(path).open('w') as f:
            json.dump(self, f, cls=AggregatorJSONEncoder)

    @staticmethod
    def load(path: str, strict: bool = True):
        with Path(path).open() as f:
            json_ = json.load(f)
        return Aggregator._reconstruct(json_, strict=strict)

    def empty(self) -> bool:
        return len(self) == 0

    def add(self, entries):
        # Add entries to the aggregator
        if not self.active and self._finalized:
            UserMessage('Aggregator object is frozen!', raise_with=Exception)
            return
        try:
            processed_entries = self._prepare_entries(entries)
            if processed_entries is None:
                return
            processed_entries = self._squeeze_entries(processed_entries)
            self.entries.append(processed_entries)
        except Exception as e:
            msg = f'Could not add entries to Aggregator object! Please verify type or original error: {e}'
            if self._raise_error:
                UserMessage(msg)
                raise Exception(msg) from e
            UserMessage(msg)

    def _prepare_entries(self, entries):
        valid_types = (tuple, list, np.float32, np.float64, np.ndarray, torch.Tensor, int, float, bool, str, Symbol)
        assert isinstance(entries, valid_types), f'Entries must be a tuple, list, numpy array, torch tensor, integer, float, boolean, string, or Symbol! Got: {type(entries)}'
        if isinstance(entries, torch.Tensor):
            return entries.detach().cpu().numpy().astype(np.float32)
        if isinstance(entries, (tuple, list)):
            return np.asarray(entries, dtype=np.float32)
        if isinstance(entries, bool):
            return int(entries)
        if isinstance(entries, str):
            return Symbol(entries).embedding.astype(np.float32)
        if isinstance(entries, Symbol):
            # Use this to avoid recursion on map setter
            self.add(entries._value)
            return None
        if isinstance(entries, Aggregator):
            self.add(entries.get())
            return None
        return entries

    def _squeeze_entries(self, entries):
        if isinstance(entries, (np.ndarray, np.float32)):
            return entries.squeeze()
        return entries

    def keys(self):
        # Get all key names of items that have the SPECIAL_CONSTANT prefix
        return [key.replace(SPECIAL_CONSTANT, '') for key in self.__dict__ if not key.startswith('_') and \
                key.replace(SPECIAL_CONSTANT, '') not in EXCLUDE_LIST]

    @property
    def active(self):
        # Get the active status of the aggregator
        return self._active

    @active.setter
    def active(self, value):
        # Set the active status of the aggregator
        assert isinstance(value, bool), f'Active status must be a boolean! Got: {type(value)}'
        self._active = value

    @property
    def finalized(self):
        # Get the finalized status of the aggregator
        return self._finalized

    @finalized.setter
    def finalized(self, value):
        # Set the finalized status of the aggregator
        assert isinstance(value, bool), f'Finalized status must be a boolean! Got: {type(value)}'
        self._finalized = value

    def finalize(self):
        # Finalizes the dynamic creation of the aggregators and freezes the object to prevent further changes
        self._active     = False
        self._finalized  = True
        def raise_exception(name, value):
            if name == 'map':
                self.__setattr__(name, value)
            else:
                UserMessage('Aggregator object is frozen!', raise_with=Exception)
        self.__setattr__ = raise_exception
        def get_attribute(*args, **kwargs):
            return self.__dict__.get(*args, **kwargs)
        self.__getattr__ = get_attribute
        # Do the same recursively for all properties of type Aggregator
        for key, value in self.__dict__.items():
            if isinstance(value, Aggregator) and (not key.startswith('_') or key.startswith(SPECIAL_CONSTANT)):
                value.finalize()

    def get(self, *args, **kwargs):
        if self._map is not None:
            return self._map(self.entries, *args, **kwargs)
        # Get the entries of the aggregator
        return self.entries

    def clear(self):
        # Clear the entries of the aggregator
        if self._finalized:
            UserMessage('Aggregator object is frozen!', raise_with=Exception)
        self._value = []

    def sum(self, axis=0):
        # Get the sum of the entries of the aggregator
        return np.sum(self.entries, axis=axis)

    def mean(self, axis=0):
        # Get the mean of the entries of the aggregator
        return np.mean(self.entries, axis=axis)

    def median(self, axis=0):
        # Get the median of the entries of the aggregator
        return np.median(self.entries, axis=axis)

    def var(self, axis=0):
        # Get the variance of the entries of the aggregator
        return np.var(self.entries, axis=axis)

    def cov(self, rowvar=False):
        # Get the covariance of the entries of the aggregator
        return np.cov(self.entries, rowvar=rowvar)

    def moment(self, moment=2, axis=0):
        # Get the moment of the entries of the aggregator
        return np.mean(np.power(self.entries, moment), axis=axis)

    def std(self, axis=0):
        # Get the standard deviation of the entries of the aggregator
        return np.std(self.entries, axis=axis)

    def min(self, axis=0):
        # Get the minimum of the entries of the aggregator
        return np.min(self.entries, axis=axis)

    def max(self, axis=0):
        # Get the maximum of the entries of the aggregator
        return np.max(self.entries, axis=axis)
