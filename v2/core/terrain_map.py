from __future__ import annotations

from enum import Enum
import numpy as np
from abc import ABC 
from .env_params import MAX_VISIBILITY
from .direction import *

class TerrainMap:
    """
    Class for holding information on the terrain of the map. 
    """
    def __init__(self, height_map : np.ndarray[float], padding : int):
        """    
        :param height_map: Information about the terrain height at the sampled point
        """
        
        self._height_map : np.ndarray[float]  = height_map
        self._padding = padding
        self.compute_gradient_map()
        self._dims = (height_map.shape[0] - 2 * self._padding, height_map.shape[1] - 2 * self._padding)

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

    @property
    def copy(self) -> TerrainMap:
        """
        Returns a copy of the terrain map.
        """
        return TerrainMap(self._height_map, self._padding)
    
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
    
    def generate(self, dims : tuple[int, int]) -> tuple[TerrainMap, tuple[int, int], tuple[int, int]]:
        """
        Generate a resource map. Derived classes should extend this method.
        """
        height_map = np.ones((dims[0], dims[1]), dtype = np.float32) * (self.min_height + self.max_height) / 2 

        padded_height_map = np.pad(
            height_map, 
            pad_width=((self.padding, self.padding), (self.padding, self.padding)), 
            mode='constant', 
            constant_values=0
        )

        terrain_map = TerrainMap(padded_height_map, self.padding)
        lower_extent = (self.padding ,self.padding)
        upper_extent = (dims[0] + self.padding, dims[1] + self.padding)
        return terrain_map, lower_extent, upper_extent
    
class CityTerrainMapGenerator(TerrainMapGenerator):
    """
    Derived class for generating a terrain map that resembles a city.
    """
    def generate(self, dims: tuple[int, int]) -> tuple[TerrainMap, tuple[int, int], tuple[int, int]]:
        """
        Generate a terrain map similar to a city, with flat areas and various building heights.
        """
        height_map = np.zeros(dims)  # Start with a flat terrain

        # Define central city zone (skyscraper area)
        central_zone_size = (dims[0] // 3, dims[1] // 3)
        central_start = (dims[0] // 3, dims[1] // 3)
        central_end = (central_start[0] + central_zone_size[0], central_start[1] + central_zone_size[1])

        # Skyscrapers in the central zone
        height_map[central_start[0]:central_end[0], central_start[1]:central_end[1]] = np.random.uniform(
            self.max_height * 0.7, self.max_height, central_zone_size)

        # Lower buildings in the surrounding zone (transition from central to outskirts)
        transition_zone_size = (dims[0] // 2, dims[1] // 2)
        transition_start = (dims[0] // 4, dims[1] // 4)
        transition_end = (transition_start[0] + transition_zone_size[0], transition_start[1] + transition_zone_size[1])

        height_map[transition_start[0]:transition_end[0], transition_start[1]:transition_end[1]] = np.random.uniform(
            self.max_height * 0.3, self.max_height * 0.6, transition_zone_size)

        # Flat areas for streets and parks (outer zones)
        height_map[:self.padding, :] = np.random.uniform(self.min_height, self.min_height + 0.1, (self.padding, dims[1]))
        height_map[-self.padding:, :] = np.random.uniform(self.min_height, self.min_height + 0.1, (self.padding, dims[1]))
        height_map[:, :self.padding] = np.random.uniform(self.min_height, self.min_height + 0.1, (dims[0], self.padding))
        height_map[:, -self.padding:] = np.random.uniform(self.min_height, self.min_height + 0.1, (dims[0], self.padding))

        # Pad the terrain map for boundary conditions
        padded_height_map = np.pad(
            height_map,
            pad_width=((self.padding, self.padding), (self.padding, self.padding)),
            mode='constant',
            constant_values=0
        )

        # Create the TerrainMap object
        terrain_map = TerrainMap(padded_height_map, self.padding)

        # Define the extents
        lower_extent = (self.padding, self.padding)
        upper_extent = (dims[0] + self.padding, dims[1] + self.padding)

        return terrain_map, lower_extent, upper_extent