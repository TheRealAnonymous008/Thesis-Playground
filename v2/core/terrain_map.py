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

from noise import snoise2
from utils.line import *
class UrbanTerrainMapGenerator(TerrainMapGenerator):
    """
    Derived class for generating a realistic urban area terrain map.
    """
    def __init__(self, 
                    base_height_range : tuple[int, int] = (-10, 10),
                    building_height_range : tuple[int, int] = (500, 550),
                    padding=MAX_VISIBILITY
                ):
        self.base_height_range =  base_height_range
        self.building_height_range = building_height_range

        min_height = base_height_range[0] 
        max_height = base_height_range[1] + building_height_range[1]
        super().__init__(min_height, max_height, padding)

    def generate(self, dims: tuple[int, int]) -> tuple[TerrainMap, tuple[int, int], tuple[int, int]]:
        """
        Generate a realistic urban terrain map with roads and buildings.
        """
        height_map = np.full((dims[0], dims[1]), (self.min_height + self.max_height) / 2, dtype=np.float32)

        self.create_underlying_terrain(height_map)
        road_mask = self.get_road_mask(height_map)
        height_map = self.generate_buildings(height_map, road_mask)

        # Step 4: Pad the height map with infinity
        padded_height_map = np.pad(
            height_map,
            pad_width=((self.padding, self.padding), (self.padding, self.padding)),
            mode='constant',
            constant_values=np.inf  # Pad with infinity
        )

        # Create the TerrainMap object
        terrain_map = TerrainMap(padded_height_map, self.padding)
        lower_extent = (self.padding, self.padding)
        upper_extent = (dims[0] + self.padding, dims[1] + self.padding)

        return terrain_map, lower_extent, upper_extent
    
    def create_underlying_terrain(self, height_map : np.ndarray):
        length, width = height_map.shape 

        # Set parameters for Perlin noise
        scale = 0.1  # Scale affects how "zoomed in" the noise is
        octaves = 6  # More octaves means more detail
        persistence = 0.5  # Controls the amplitude of each octave
        lacunarity = 2.0  # Controls the frequency of each octave
        
        # Generate height values using Perlin noise
        for x in range(length):
            for y in range(width):
                # Generate a height value using Perlin noise
                noise_value = snoise2(x * scale, 
                                    y * scale, 
                                    octaves=octaves, 
                                    persistence=persistence, 
                                    lacunarity=lacunarity)
                
                # Scale noise_value to the desired height range
                # Normalize noise_value from (-1, 1) to (min_height, max_height)
                height = self.base_height_range[0] + (noise_value + 1) * 0.5 * (self.base_height_range[1] - self.base_height_range[0])

                # Ensure the height is within the specified range
                height = np.clip(height, self.base_height_range[0], self.base_height_range[1])

                # Set the height in the height map
                height_map[x, y] = height

        return height_map
    
    def get_road_mask(self, height_map : np.ndarray) -> np.ndarray: 
        length, width = height_map.shape   
        road_mask = np.zeros_like(height_map, dtype=bool)
        
        # Parameters for junctions
        num_junctions = np.random.randint(5, 15)  # Randomly decide the number of junctions
        junctions = []

        # Sample random junction points in the world
        for _ in range(num_junctions):
            x = np.random.randint(0, length - 1)
            y = np.random.randint(0, width - 1)
            junctions.append((x, y))

        # Connect junctions with roads
        for i in range(len(junctions)):
            for j in range(i + 1, len(junctions)):
                start = junctions[i]
                end = junctions[j]

                # Use Bresenham's line algorithm to draw a line between two junctions
                bresenham_line(road_mask, start, end)

        return road_mask 
    
    def generate_buildings(self, height_map : np.ndarray, road_mask : np.ndarray):
        length, width = height_map.shape 

        for x in range(length):
            for y in range(width):
                if not road_mask[x, y]:
                    height_map[x, y] += np.random.uniform(self.building_height_range[0], self.building_height_range[1])
                

        return height_map