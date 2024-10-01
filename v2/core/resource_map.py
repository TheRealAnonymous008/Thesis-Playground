from __future__ import annotations

from enum import Enum
import numpy as np
from abc import ABC 
from .env_params import RESOURCE_TYPES, MAX_VISIBILITY

from .resource import Resource, _QuantityType, _ResourceType

class ResourceMap:
    """
    Class for holding information on the resources of a map
    """
    def __init__(self, resource_type_map : np.ndarray[int], resource_quantity_map : np.ndarray[_QuantityType], padding : int):
        """    
        :param resource_type:  Holds resource type information.
        :param resource_quantity:  Holds the quantity of resource in a tile.
        :param padding: How much padding to place on each side. It is recommended to match the padding with the maximum visible range of the agents in the environment
        """
        if resource_quantity_map.shape != resource_quantity_map.shape:
            raise Exception(f"Error: Resource Type and Quantity Maps should have the same shape. Got {resource_type_map.shape} and {resource_quantity_map.shape} respectively")
        
        self._resource_type_map : np.ndarray[int]  = resource_type_map
        self._resource_quantity_map : np.ndarray[_QuantityType] = resource_quantity_map
        self._padding = padding
        self._dims = (resource_type_map.shape[0] - 2 * self._padding, resource_type_map.shape[1] - 2 * self._padding)

    def translate_idx(self, idx : tuple[int, int]) -> tuple[int, int]:
        """
        Translates the provided index based on the padding of the resource map. This way, we can treat the padding as nonexistent 
        """
        return idx[0] + self._padding, idx[1] + self._padding

    def add_resource(self, idx : tuple[int, int], type : int, quantity : _QuantityType):
        """
        Adds a resource to the map. If the quantity is less than or equal to 0, then addition is ignored.
        Assumes `idx` is correct.
        """
        if quantity <= 0:
            return 
        
        x, y = self.translate_idx(idx)
        self._resource_type_map[x, y]  = int(type) 
        self._resource_quantity_map[x, y] = quantity

    def subtract_resource(self, idx : tuple[int, int], quantity : _QuantityType):
        """
        Removes a resource from the map. If the final quantity is less than 0, the resource is removed entirely.
        If `quantity` <= 0, then subtraction is ignored. Assumes `idx` is correct.

        Returns the type at the specified `idx`
        """
        x, y = self.translate_idx(idx)
        q = self._resource_quantity_map[x, y]
        new_q = max(q - quantity, 0)
        
        self._resource_quantity_map[x, y] = new_q
        r = self._resource_type_map[x, y]

        if new_q == 0:
            self._resource_type_map[x, y] = 0

        return Resource(r, q - new_q)
    
    def get(self, idx : tuple[int, int]) -> tuple[int, _QuantityType]:
        """
        Returns a tuple of the resource type and quantity. Assumes `idx` is correct.
        """
        idx = self.translate_idx(idx)
        return self._resource_type_map[idx[0], idx[1]], self._resource_quantity_map[idx[0], idx[1]]
    
    @property
    def type_map(self) -> np.ndarray: 
        """
        Returns a copy of the whole type map 
        """
        return self._resource_type_map.copy()
    
    @property
    def quantity_map(self) -> np.ndarray:
        """
        Returns a copy of the whole quantity map
        """
        return self._resource_quantity_map.copy()

    def get_type(self, idx : tuple[int, int]) -> int:
        """
        Returns resource type at idx. Assumes `idx` is correct.
        """
        idx = self.translate_idx(idx)
        return self._resource_type_map[idx[0]][idx[1]]

    def get_quantity(self, idx : tuple[int, int]) -> _QuantityType:
        """
        Returns resource quantity at idx. Assumes `idx` is correct.
        """
        idx = self.translate_idx(idx)
        return self._resource_quantity_map[idx[0]][idx[1]]
    
    @property
    def copy(self) -> ResourceMap:
        """
        Returns a copy of the resource map.
        """
        return ResourceMap(self._resource_type_map.copy(), self._resource_quantity_map.copy(), self._padding)
    
    @property
    def shape(self) -> np._Shape:
        """
        Returns the shape of the resource map
        """
        return self._dims
    

class ResourceMapGenerator(ABC):
    """
    Base class for generating resource maps.
    """
    def __init__(self, resource_types = RESOURCE_TYPES, padding = MAX_VISIBILITY):
        """
        :param resource_types: How many resource types to place in the world.
        :param padding: How much padding to place on each side. It is recommended to match the padding with the maximum visible range of the agents in the environment
        """
        self.resource_types = resource_types
        self.padding = padding
    
    def generate(self, dims : tuple[int, int]) -> tuple[ResourceMap, tuple[int, int], tuple[int, int]]:
        """
        Generate a resource map. Derived classes should extend this method.
        """
        resource_type_map = np.zeros(dims, dtype=np.int32)
        resource_quantity_map = np.zeros(dims, dtype = _QuantityType)
        resource_type_map = np.pad(resource_type_map, ((self.padding, self.padding), (self.padding, self.padding)), mode = "constant", constant_values=-1)
        resource_quantity_map = np.pad(resource_quantity_map, ((self.padding, self.padding), (self.padding, self.padding)), mode = "constant", constant_values=-1)

        lower_extent = (self.padding ,self.padding)
        upper_extent = (dims[0] + self.padding, dims[1] + self.padding)
        return ResourceMap(resource_type_map=resource_type_map, resource_quantity_map=resource_quantity_map, padding=self.padding), lower_extent, upper_extent
    
class RandomMapGenerator(ResourceMapGenerator):
    def generate(self, dims: tuple[int, int]) -> tuple[ResourceMap, tuple[int, int], tuple[int, int]]:
        resource_map, lower_extent, upper_extent = super().generate(dims)
        num_clumps = 10 

        # Generate Clumps of Resources
        for resource_type in range(1, self.resource_types + 1):
            for _ in range(num_clumps):
                x, y = np.random.randint(lower_extent[0], upper_extent[0]), np.random.randint(lower_extent[1], upper_extent[1])
                clump_size = np.random.randint(1, 6)
                
                for i in range(-clump_size, clump_size + 1):
                    for j in range(-clump_size, clump_size + 1):
                        if 0 <= x + i < dims[0] and 0 <= y + j < dims[1]:
                            qty = _QuantityType(max(np.random.normal(5, 2), 1))
                            resource_map.add_resource((x + i, y + j), resource_type, qty)

        return resource_map, lower_extent, upper_extent
    

