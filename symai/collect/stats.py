import torch
import re
import json
import numpy as np

from typing import Any, Optional
from json import JSONEncoder

from ..symbol import Symbol


def _normalize_name(name: str) -> str:
    # Replace any character that is not a letter or a number with an underscore
    normalized_name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    return normalized_name.lower()


class AggregatorJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # drop active from state
        elif isinstance(obj, Aggregator):
            obj = obj.__dict__.copy()
            obj.pop('active', None)
            # drop everything that starts with __aggregate_
            for key in list(obj.keys()):
                if not key.startswith('__aggregate_') and not key.startswith('entries'):
                    obj.pop(key, None)
            return obj
        return obj.__dict__


class Aggregator(Symbol):
    def __init__(self,
                 aggregator: Optional["Aggregator"] = None,
                 active: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if aggregator is not None:
            assert isinstance(aggregator, Aggregator), 'Aggregator must be an instance of Aggregator! Got: {}'.format(type(aggregator))
            Aggregator._set_values(self, aggregator.__dict__)
        self._active    = active
        self._finalized = False
        self.entries    = []

    def __getattr__(self, name):
        # replace name special characters and spaces with underscores
        name = _normalize_name(name)
        # Dynamically create new aggregator instance if it does not exist
        if self._active and name not in self.__dict__:
            aggregator = Aggregator()
            # create a new aggregate aggregator
            # named __aggregate_{name} for automatic aggregation
            self.__dict__[f'__aggregate_{name}'] = aggregator
            # add also a property with the same name but without the __aggregate_ prefix as a shortcut
            self.__dict__[name] = self.__dict__[f'__aggregate_{name}']
            return self.__dict__.get(name)
        elif not self._active and name not in self.__dict__:
            raise Exception(f'Aggregator object is frozen! No attribute {name} found!')
        return self.__dict__.get(name)

    def shape(self):
        if len(self.entries) > 0:
            return self.entries[0].shape
        else:
            return ()

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
        state.pop('active', None)
        return super().__setstate__(state)

    def serialize(self):
        return json.dumps(self, cls=AggregatorJSONEncoder)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self, f, cls=AggregatorJSONEncoder)

    @staticmethod
    def _set_values(obj, dictionary):
        # recursively reconstruct the object
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Aggregator._reconstruct(value)
            if key.startswith('__aggregate_'):
                key = key.replace('__aggregate_', '')
            if key.startswith('entries'):
                value = np.array(value, dtype=np.float32)
            setattr(obj, key, value)

    @staticmethod
    def _reconstruct(json_):
        obj = Aggregator()
        return Aggregator._set_values(obj, json_)

    @staticmethod
    def load(path: str):
        with open(path, 'r') as f:
            json_ = json.load(f)
        obj = Aggregator._reconstruct(json_)
        return obj

    def __repr__(self) -> str:
        '''
        Get the representation of the Symbol object as a string.

        Returns:
            str: The representation of the Symbol object.
        '''
        # class with full path
        class_ = self.__class__.__module__ + '.' + self.__class__.__name__
        hex_   = hex(id(self))
        return f'<class {class_} at {hex_}>(entries={self.entries})'

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
        return len(self.entries)

    def empty(self) -> bool:
        return len(self) == 0

    def add(self, entries):
        # Add entries to the aggregator
        if not self.active:
            if self._finalized:
                raise Exception('Aggregator object is frozen!')
            return

        try:
            # Append a new entry to the aggregator
            assert type(entries) in [tuple, list, np.float32, np.float64, np.ndarray, torch.Tensor, int, float, bool, str] or isinstance(entries, Symbol), 'Entries must be a tuple, list, numpy array, torch tensor, integer, float, boolean, string, or Symbol! Got: {}'.format(type(entries))
            if type(entries) == torch.Tensor:
                entries = entries.detach().cpu().numpy().astype(np.float32)
            elif type(entries) in [tuple, list]:
                entries = np.array(entries, dtype=np.float32)
            elif type(entries) in [int, float]:
                entries = np.array([entries], dtype=np.float32)
            elif type(entries) == bool:
                entries = np.array([int(entries)], dtype=np.float32)
            elif type(entries) == str:
                entries = Symbol(entries).embedding.astype(np.float32)
            elif isinstance(entries, Symbol):
                self.add(entries.value)
                return
            elif isinstance(entries, Aggregator):
                self.add(entries.get())
                return
            self.entries.append(entries)
        except Exception as e:
            raise Exception(f'Could not add entries to Aggregator object! Please verify type or original error: {e}') from e

    @property
    def active(self):
        # Get the active status of the aggregator
        return self._active

    @active.setter
    def active(self, value):
        # Set the active status of the aggregator
        assert isinstance(value, bool), 'Active status must be a boolean! Got: {}'.format(type(value))
        self._active = value

    @property
    def finalized(self):
        # Get the finalized status of the aggregator
        return self._finalized

    @finalized.setter
    def finalized(self, value):
        # Set the finalized status of the aggregator
        assert isinstance(value, bool), 'Finalized status must be a boolean! Got: {}'.format(type(value))
        self._finalized = value

    def finalize(self):
        # Finalizes the dynamic creation of the aggregators and freezes the object to prevent further changes
        self._active      = False
        self._finalized  = True
        def raise_exception(*args, **kwargs):
            raise Exception('Aggregator object is frozen!')
        self.__setattr__ = raise_exception
        def get_attribute(*args, **kwargs):
            return self.__dict__.get(*args, **kwargs)
        self.__getattr__ = get_attribute
        # Do the same recursively for all properties of type Aggregator
        for key, value in self.__dict__.items():
            if isinstance(value, Aggregator) and (not key.startswith('_') or key.startswith('__aggregate_')):
                value.finalize()

    def get(self):
        # Get the entries of the aggregator
        assert 'entries' in self.__dict__, 'No entries found!'
        return self.entries

    def clear(self):
        # Clear the entries of the aggregator
        assert 'entries' in self.__dict__, 'No entries found!'
        if self._finalized:
            raise Exception('Aggregator object is frozen!')
        self.entries = []

    def sum(self, axis=0):
        # Get the sum of the entries of the aggregator
        assert 'entries' in self.__dict__, 'No entries found!'
        return np.sum(self.entries, axis=axis)

    def mean(self, axis=0):
        # Get the mean of the entries of the aggregator
        assert 'entries' in self.__dict__, 'No entries found!'
        return np.mean(self.entries, axis=axis)

    def median(self, axis=0):
        # Get the median of the entries of the aggregator
        assert 'entries' in self.__dict__, 'No entries found!'
        return np.median(self.entries, axis=axis)

    def var(self, axis=0):
        # Get the variance of the entries of the aggregator
        assert 'entries' in self.__dict__, 'No entries found!'
        return np.var(self.entries, axis=axis)

    def cov(self, rowvar=False):
        # Get the covariance of the entries of the aggregator
        assert 'entries' in self.__dict__, 'No entries found!'
        return np.cov(self.entries, rowvar=rowvar)

    def moment(self, moment=2, axis=0):
        # Get the moment of the entries of the aggregator
        assert 'entries' in self.__dict__, 'No entries found!'
        return np.mean(np.power(self.entries, moment), axis=axis)

    def std(self, axis=0):
        # Get the standard deviation of the entries of the aggregator
        assert 'entries' in self.__dict__, 'No entries found!'
        return np.std(self.entries, axis=axis)

    def min(self, axis=0):
        # Get the minimum of the entries of the aggregator
        assert 'entries' in self.__dict__, 'No entries found!'
        return np.min(self.entries, axis=axis)

    def max(self, axis=0):
        # Get the maximum of the entries of the aggregator
        assert 'entries' in self.__dict__, 'No entries found!'
        return np.max(self.entries, axis=axis)

