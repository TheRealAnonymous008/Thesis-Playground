from __future__ import annotations

from .map import *

class TerrainMap(BaseMap):
    """
    Class for holding information on the terrain of the map. 
    """
    def __init__(self, height_map : np.ndarray[float], padding : int) :
        """    
        :param height_map: Information about the terrain height at the sampled point
        :param padding: Padding added to all sides.
        """
        super().__init__(height_map, padding)
        self._dims = (height_map.shape[0], height_map.shape[1])
        self._height_map_gradient = self.compute_gradient_map()
    
    def compute_gradient_map(self) -> BaseMap:
        """
        Compute the gradient at each point in the height map for easy access. In particular, it computes it for each direction of motion.
        The height_map_gradient has entries for each direction on each cell.
        At the boundary (i.e., if the agent moving would end up going out of bounds), the gradient should be infinity.
        """
        length, width = self._map.shape
        height_map_gradient = np.full((length, width, 5), np.inf) 

        for i in range(self._padding, length - self._padding):
            for j in range(self._padding, width - self._padding):
                current_height = self._map[i, j]

                height_map_gradient[i, j, 0] = 0

                # North direction
                if i > self._padding:
                    height_map_gradient[i, j, Direction.NORTH.value] = self._map[i - 1, j] - current_height
                else:
                    height_map_gradient[i, j, Direction.NORTH.value] = np.inf  # Boundary case

                # South direction
                if i < length - self._padding - 1:
                    height_map_gradient[i, j, Direction.SOUTH.value] = self._map[i + 1, j] - current_height
                else:
                    height_map_gradient[i, j, Direction.SOUTH.value] = np.inf  # Boundary case

                # East direction
                if j > self._padding:
                    height_map_gradient[i, j, Direction.WEST.value] = self._map[i, j - 1] - current_height
                else:
                    height_map_gradient[i, j, Direction.WEST.value] = np.inf  # Boundary case

                # West direction
                if j < width - self._padding - 1:
                    height_map_gradient[i, j, Direction.EAST.value] = self._map[i, j + 1] - current_height
                else:
                    height_map_gradient[i, j, Direction.EAST.value] = np.inf  # Boundary case

        return height_map_gradient

    def get_gradient(self, idx : tuple[int, int], direction : Direction):
        idx = self.translate_idx(idx)
        return self._height_map_gradient[idx[0], idx[1], direction.value]
    
    @property
    def copy(self) -> TerrainMap:
        """
        Returns a copy of the terrain map.
        """
        h = self._copy()
        return TerrainMap(h, self._padding)


