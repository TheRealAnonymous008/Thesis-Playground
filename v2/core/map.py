from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from .direction import *

class BaseMap(ABC): 
    """
    Base class for the terrain map. Specific implementations should override this
    """
    def __init__(self, map : np.ndarray,  padding: int = 0): 
        """
        :param map: Information about the map.
        :param padding: Padding added to all sides
        :param gradiated: If true, also computes a gradient map
        """
        self._padding = padding        
        self._map : np.ndarray[float] = np.pad(
            map, 
            pad_width=((self._padding, self._padding), (self._padding, self._padding)), 
            mode='constant', 
            constant_values=-1
        )
        self._dims = (map.shape[0], map.shape[1])

    def get(self, idx : tuple[int, int]):
        idx = self.translate_idx(idx)
        return self._map[idx[0], idx[1]]
    
    def set(self, idx : tuple[int, int], value):
        idx = self.translate_idx(idx)
        self._map[idx[0], idx[1]] = value
    
    def translate_idx(self, idx : tuple[int, int]) -> tuple[int, int]:
        """
        Translates the provided index based on the padding of the resource map. This way, we can treat the padding as nonexistent 
        """
        return idx[0] + self._padding, idx[1] + self._padding
    
    @property
    def copy(self):
        """
        Returns a copy of the  map.
        """
        return BaseMap(self._copy(), self._padding)
    
    def _copy(self):
        """
        Returns a copy of the map.
        Use as a util method 
        """
        x0, y0 = self.translate_idx((0, 0))
        x1, y1 = self.translate_idx((self._dims[0], self._dims[1]))
        h = self._map[x0 : x1, y0 : y1]
        return h

    @property
    def shape(self) -> np._Shape:
        """
        Returns the shape of the map
        """
        return self._dims
    
    @property
    def map(self) -> np.ndarray: 
        """
        Returns a copy of the map 
        """
        return self.copy()

class BaseMapCollection:
    def __init__(self):
        """
        Manages a collection of maps 
        """
        self._maps : dict[str, BaseMap] = {}

    def add_map(self, name : str,  map : BaseMap):
        self._maps[name] = map
    
    @property
    def copy(self) -> BaseMapCollection:
        collection = BaseMapCollection()
        for key, map in self._maps.items():
            collection.add_map(key, map.copy)

        return collection
    
    def get(self, name): 
        return self._maps.get(name, None)


class BaseMapGenerator(ABC):
    """
    Base class for generating the terrain.
    """
    def __init__(self,  padding = 0):
        """
        :param padding:    - the padding to place in the terrain map generator
        """
        self._padding = padding
    
    @abstractmethod
    def generate(self, dims : tuple[int, int]) -> BaseMap:
        """
        Generate a map. Derived classes should extend this method.
        """
        pass