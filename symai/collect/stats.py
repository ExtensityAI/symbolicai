import torch
import numpy as np

from typing import Any

from ..symbol import Symbol, Metadata


def _normalize_name(name: str) -> str:
    name.replace(' ', '_').replace('-', '_').replace('.', '_').replace(',', '_').replace('(', '_').replace(')', '_').replace('[', '_').replace(']', '_').replace('{', '_').replace('}', '_').replace(':', '_').replace(';', '_').replace('\'', '_').replace('"', '_').replace('?', '_').replace('!', '_').replace('@', '_').replace('#', '_').replace('$', '_').replace('%', '_').replace('^', '_').replace('&', '_').replace('*', '_').replace('+', '_').replace('=', '_').replace('/', '_').replace('\\', '_').replace('|', '_').replace('<', '_').replace('>', '_').replace('~', '_').replace('`', '_').replace('=', '_').replace(' ', '_')
    return name.lower()


class Aggregator(Metadata):
    def __getattr__(self, name):
        # replace name special characters and spaces with underscores
        name = _normalize_name(name)
        # Dynamically create new aggregator instance if it does not exist
        if name not in self.__dict__:
            aggregator = Aggregator()
            self.__dict__[name] = aggregator
            # create a new aggregate aggregator for the current aggregator
            # named __aggregate_{name} for automatic aggregation of the aggregator
            self.__dict__[f'__aggregate_{name}'] = aggregator
            return self.__dict__.get(name)
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

    def __repr__(self) -> str:
        '''
        Get the representation of the Symbol object as a string.

        Returns:
            str: The representation of the Symbol object.
        '''
        # class with full path
        class_ = self.__class__.__module__ + '.' + self.__class__.__name__
        hex_   = hex(id(self))
        return f'<class {class_} at {hex_}>'

    def __or__(self, other: Any) -> Any:
        self.add(other)
        return other

    def __ror__(self, other: Any) -> Any:
        self.add(other)
        return other

    def __ior__(self, other: Any) -> Any:
        self.add(other)
        return other

    def add(self, entries):
        try:
            # Append a new entry to the aggregator
            if 'entries' not in self.__dict__:
                self.entries = []
            assert type(entries) in [tuple, list, np.ndarray, torch.Tensor, int, float, bool, str] or isinstance(entries, Symbol), 'Entries must be a tuple, list, numpy array, torch tensor, integer, float, boolean, string, or Symbol!'
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

    def get(self):
        # Get the entries of the aggregator
        assert 'entries' in self.__dict__, 'No entries found!'
        return self.entries

    def clear(self):
        # Clear the entries of the aggregator
        assert 'entries' in self.__dict__, 'No entries found!'
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

    def variance(self, axis=0):
        # Get the variance of the entries of the aggregator
        assert 'entries' in self.__dict__, 'No entries found!'
        return np.var(self.entries, axis=axis)

    def covariance(self, rowvar=False):
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

