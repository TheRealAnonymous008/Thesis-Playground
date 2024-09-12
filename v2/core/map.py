from __future__ import annotations
from dataclasses import dataclass

from enum import Enum
import numpy as np
from abc import ABC 

@dataclass
class Resource: 
    """
    Dataclass for resources. A resource holds a `type` andn `quantity`
    """
    type : int 
    quantity : float


class ResourceMap:
    """
    Class for holding information on the resources of a map
    
    `resource_type` - holds resource type information.

    `resource_quantity` - holds the quantity of resource in a tile.
    """
    def __init__(self, resource_type_map : np.ndarray[int], resource_quantity_map : np.ndarray[float]):
        if resource_quantity_map.shape != resource_quantity_map.shape:
            raise Exception(f"Error: Resource Type and Quantity Maps should have the same shape. Got {resource_type_map.shape} and {resource_quantity_map.shape} respectively")
        
        self._resource_type_map : np.ndarray[int]  = resource_type_map
        self._resource_quantity_map : np.ndarray[float] = resource_quantity_map

    def add_resource(self, idx : tuple[int, int], type : int, quantity : float):
        """
        Adds a resource to the map. If the quantity is less than or equal to 0, then addition is ignored.
        Assumes `idx` is correct.
        """
        if quantity <= 0:
            return 
        
        self._resource_type_map[idx[0]][idx[1]]  = int(type) 
        self._resource_quantity_map[idx[0]][idx[1]] = quantity

    def subtract_resource(self, idx : tuple[int, int], quantity : float):
        """
        Removes a resource from the map. If the final quantity is less than 0, the resource is removed entirely.
        If `quantity` <= 0, then subtraction is ignored. Assumes `idx` is correct.

        Returns the type at the specified `idx`
        """
        if quantity <= 0:
            return 
        
        q = self._resource_quantity_map[idx[0]][idx[1]]
        self._resource_quantity_map[idx[0]][idx[1]] = max(q - quantity, 0)
        
        removed = q - self._resource_quantity_map[idx[0]][idx[1]]
        r = self._resource_type_map[idx[0]][idx[1]]
        q = self._resource_quantity_map[idx[0]][idx[1]]

        if q <= 0:
            self._resource_type_map[idx[0]][idx[1]] = 0

        return Resource(r, removed)
    
    def get(self, idx : tuple[int, int]) -> tuple[int, float]:
        """
        Returns a tuple of the resource type and quantity. Assumes `idx` is correct.
        """
        return self._resource_type_map[idx[0]][idx[1]], self._resource_quantity_map[idx[0]][idx[1]]
    
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

    def get_type(self, idx : tuple[int, int]):
        """
        Returns resource type at idx. Assumes `idx` is correct.
        """
        return self._resource_type_map[idx[0]][idx[1]]

    def get_quantity(self, idx : tuple[int, int]):
        """
        Returns resource quantity at idx. Assumes `idx` is correct.
        """
        return self._resource_quantity_map[idx[0]][idx[1]]
    
    @property
    def copy(self) -> ResourceMap:
        """
        Returns a copy of the resource map.
        """
        return ResourceMap(self._resource_type_map.copy(), self._resource_quantity_map.copy())
    
    @property
    def shape(self) -> np._Shape:
        """
        Returns the shape of the resource map
        """
        return self._resource_type_map.shape
    
class MapGenerator(ABC):
    def __init__(self, resource_types = 5):
        self.resource_types = resource_types
    
    def generate(self, dims : tuple[int, int]) -> ResourceMap:
        resource_type_map = np.zeros(dims, dtype=np.int32)
        resource_quantity_map = np.zeros(dims)

        return ResourceMap(resource_type_map=resource_type_map, resource_quantity_map=resource_quantity_map)
    
class RandomMapGenerator(MapGenerator):
    def generate(self, dims: tuple[int, int]) -> ResourceMap:
        resource_map : ResourceMap = super().generate(dims)
        num_clumps = 10 

        # Generate Clumps of Resources
        for resource_type in range(1, self.resource_types + 1):
            for _ in range(num_clumps):
                x, y = np.random.randint(0, dims[0]), np.random.randint(0, dims[1])
                clump_size = np.random.randint(1, 6)
                
                for i in range(-clump_size, clump_size + 1):
                    for j in range(-clump_size, clump_size + 1):
                        if 0 <= x + i < dims[0] and 0 <= y + j < dims[1]:
                            qty = max(np.random.normal(5, 2), 1)
                            resource_map.add_resource((x + i, y + j), resource_type, qty)

        return resource_map