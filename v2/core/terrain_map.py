from __future__ import annotations

from enum import Enum
import numpy as np
import numpy.random as random
from abc import ABC 
from .env_params import MAX_VISIBILITY
from .direction import *

class TerrainMap:
    """
    Class for holding information on the terrain of the map. 
    """
    def __init__(self, height_map : np.ndarray[float], padding : int, density_map : np.ndarray[float] = None) :
        """    
        :param height_map: Information about the terrain height at the sampled point
        :param padding: Padding added to all sides.
        :param density_map: Miscellaneous information
        """

        self._padding = padding        
        self._height_map : np.ndarray[float] = np.pad(
            height_map, 
            pad_width=((self._padding, self._padding), (self._padding, self._padding)), 
            mode='constant', 
            constant_values=0
        )
        if density_map is not None: 
            self._density_map : np.ndarray[float] = np.pad(
                density_map, 
                pad_width=((self._padding, self._padding), (self._padding, self._padding)), 
                mode='constant', 
                constant_values=-np.inf
            )
        else:  
            self._density_map = None

        self.compute_gradient_map()
        self._dims = (height_map.shape[0], height_map.shape[1])

    def translate_idx(self, idx : tuple[int, int]) -> tuple[int, int]:
        """
        Translates the provided index based on the padding of the resource map. This way, we can treat the padding as nonexistent 
        """
        return idx[0] + self._padding, idx[1] + self._padding
    
    def compute_gradient_map(self):
        """
        Compute the gradient at each point in the height map for easy access. In particular, it computes it for each direction of motion.
        The height_map_gradient has entries for each direction on each cell.
        At the boundary (i.e., if the agent moving would end up going out of bounds), the gradient should be infinity.
        """
        length, width = self._height_map.shape
        self._height_map_gradient = np.full((length, width, 5), np.inf) 

        for i in range(self._padding, length - self._padding):
            for j in range(self._padding, width - self._padding):
                current_height = self._height_map[i, j]

                self._height_map_gradient[i, j, 0] = 0

                # North direction
                if i > self._padding:
                    self._height_map_gradient[i, j, Direction.NORTH.value] = self._height_map[i - 1, j] - current_height
                else:
                    self._height_map_gradient[i, j, Direction.NORTH.value] = np.inf  # Boundary case

                # South direction
                if i < length - self._padding - 1:
                    self._height_map_gradient[i, j, Direction.SOUTH.value] = self._height_map[i + 1, j] - current_height
                else:
                    self._height_map_gradient[i, j, Direction.SOUTH.value] = np.inf  # Boundary case

                # East direction
                if j > self._padding:
                    self._height_map_gradient[i, j, Direction.WEST.value] = self._height_map[i, j - 1] - current_height
                else:
                    self._height_map_gradient[i, j, Direction.WEST.value] = np.inf  # Boundary case

                # West direction
                if j < width - self._padding - 1:
                    self._height_map_gradient[i, j, Direction.EAST.value] = self._height_map[i, j + 1] - current_height
                else:
                    self._height_map_gradient[i, j, Direction.EAST.value] = np.inf  # Boundary case

    def get_gradient(self, idx : tuple[int, int], direction : Direction):
        idx = self.translate_idx(idx)
        return self._height_map_gradient[idx[0], idx[1], direction.value]
    
    def get_height(self, idx : tuple[int, int]):
        idx = self.translate_idx(idx)
        return self._height_map[idx[0], idx[1]]

    def get_density(self, idx : tuple[int, int]):
        if self._density_map is None: 
            return 0
        
        idx = self.translate_idx(idx)
        return self._density_map[idx[0], idx[1]]


    @property
    def copy(self) -> TerrainMap:
        """
        Returns a copy of the terrain map.
        """
        x0, y0 = self.translate_idx((0, 0))
        x1, y1 = self.translate_idx((self._dims[0], self._dims[1]))
        h = self._height_map[x0 : x1, y0 : y1]
        d = None 
        if self._density_map is not None: 
            d = self._density_map[x0 : x1, y0 : y1]
        return TerrainMap(h, self._padding, d)
    
    @property
    def shape(self) -> np._Shape:
        """
        Returns the shape of the terrain map
        """
        return self._dims
    
    @property
    def height_map(self) -> np.ndarray: 
        """
        Returns a copy of the height map 
        """
        return self._height_map.copy()


class TerrainMapGenerator(ABC):
    """
    Base class for generating the terrain.
    """
    def __init__(self, min_height : float , max_height : float , padding = MAX_VISIBILITY):
        """
        :param min_height: - the minimum allowable height
        :param max_height: - the maximum allowable height
        :param padding:    - the padding to place in the terrain map generator
        """
        assert(min_height < max_height)

        self.min_height = min_height
        self.max_height = max_height 
        self.padding = padding
    
    def generate(self, dims : tuple[int, int]) -> TerrainMap:
        """
        Generate a resource map. Derived classes should extend this method.
        """
        height_map = np.ones((dims[0], dims[1]), dtype = np.float32) * (self.min_height + self.max_height) / 2 

        terrain_map = TerrainMap(height_map, self.padding)
        return terrain_map
