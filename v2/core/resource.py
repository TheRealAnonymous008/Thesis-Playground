from __future__ import annotations
from dataclasses import dataclass

from enum import Enum
import numpy as np
from abc import ABC 


class ResourceGenerator(ABC):
    def __init__(self, resource_types = 5):
        self.resource_types = resource_types
    
    def generate(self, dims : tuple[int, int]) -> np.ndarray:
        resource_map = np.zeros(dims)

        return resource_map
    
class RandomMapGenerator(ResourceGenerator):
    def generate(self, dims: tuple[int, int]) -> np.ndarray:
        resource_map = super().generate(dims)
        num_clumps = 10 

        # Generate Clumps of Resources
        for resource_type in range(1, self.resource_types + 1):
            for _ in range(num_clumps):
                x, y = np.random.randint(0, dims[0]), np.random.randint(0, dims[1])
                clump_size = np.random.randint(1, 6)
                
                for i in range(-clump_size, clump_size + 1):
                    for j in range(-clump_size, clump_size + 1):
                        if 0 <= x + i < dims[0] and 0 <= y + j < dims[1]:
                            resource_map[x + i, y + j] = resource_type

        return resource_map