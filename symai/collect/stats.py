import torch
import numpy as np

from typing import Any

from ..symbol import Symbol, Metadata


class Aggregator(Metadata):
    def __getattr__(self, name):
        # Dynamically create new collector instance if it does not exist
        if name not in self.__dict__:
            # replace name special characters and spaces with underscores
            name = name.replace(' ', '_').replace('-', '_').replace('.', '_')
            name = name.lower()
            collector = Aggregator()
            self.__dict__[name] = collector
            # create a new aggregate collector for the current collector
            # named __aggregate_{name} for automatic aggregation of the collector
            self.__dict__[f'__aggregate_{name}'] = collector
            return self.__dict__.get(name)
        return self.__dict__.get(name)

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
        # Append a new entry to the collector
        if 'entries' not in self.__dict__:
            self.entries = []
        assert type(entries) in [tuple, list, np.ndarray, torch.Tensor, int, float, str] or isinstance(entries, Symbol), 'Entries must be a list or numpy array!'
        if type(entries) == torch.Tensor:
            entries = entries.detach().cpu().numpy()
        elif type(entries) in [tuple, list]:
            entries = np.array(entries)
        elif type(entries) in [int, float]:
            entries = np.array([entries])
        elif type(entries) == str:
            entries = Symbol(entries).embedding
        elif isinstance(entries, Symbol):
            self.add(entries.value)
            return
        elif isinstance(entries, Aggregator):
            self.add(entries.get())
            return
        self.entries.append(entries)

    def get(self):
        # Get the entries of the collector
        assert 'entries' in self.__dict__, 'No entries found!'
        return self.entries

    def clear(self):
        # Clear the entries of the collector
        assert 'entries' in self.__dict__, 'No entries found!'
        self.entries = []

    def sum(self, axis=0):
        # Get the sum of the entries of the collector
        assert 'entries' in self.__dict__, 'No entries found!'
        return np.sum(self.entries, axis=axis)

    def mean(self, axis=0):
        # Get the mean of the entries of the collector
        assert 'entries' in self.__dict__, 'No entries found!'
        return np.mean(self.entries, axis=axis)

    def median(self, axis=0):
        # Get the median of the entries of the collector
        assert 'entries' in self.__dict__, 'No entries found!'
        return np.median(self.entries, axis=axis)

    def variance(self, axis=0):
        # Get the variance of the entries of the collector
        assert 'entries' in self.__dict__, 'No entries found!'
        return np.var(self.entries, axis=axis)

    def covariance(self, rowvar=False):
        # Get the covariance of the entries of the collector
        assert 'entries' in self.__dict__, 'No entries found!'
        return np.cov(self.entries, rowvar=rowvar)

    def moment(self, moment=2, axis=0):
        # Get the moment of the entries of the collector
        assert 'entries' in self.__dict__, 'No entries found!'
        return np.mean(np.power(self.entries, moment), axis=axis)

    def std(self, axis=0):
        # Get the standard deviation of the entries of the collector
        assert 'entries' in self.__dict__, 'No entries found!'
        return np.std(self.entries, axis=axis)

    def min(self, axis=0):
        # Get the minimum of the entries of the collector
        assert 'entries' in self.__dict__, 'No entries found!'
        return np.min(self.entries, axis=axis)

    def max(self, axis=0):
        # Get the maximum of the entries of the collector
        assert 'entries' in self.__dict__, 'No entries found!'
        return np.max(self.entries, axis=axis)
