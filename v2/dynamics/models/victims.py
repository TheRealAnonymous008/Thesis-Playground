from core.map import *

class VictimGenerator(BaseMapGenerator):
    def __init__(self, padding=MAX_VISIBILITY):
        """
        :param padding:    - the padding to place in the terrain map generator
        """
        super().__init__(padding)

        # The population density map from which we sample victims
        self._density_map: BaseMap | None = None

    def set_density_map(self, density_map: BaseMap):
        self._density_map = density_map

    def generate(self, dims: tuple[int, int]) -> BaseMap:
        """
        Generate a map of the victims in the area based on population density.
        The more densely populated areas will have a higher number of victims.
        """
        if self._density_map is None:
            raise ValueError("Density map has not been set.")

        # Create an empty map for storing the victim locations
        victim_map = np.zeros(dims)

        # Iterate over each location in the map
        for x in range(dims[0]):
            for y in range(dims[1]):
                # Get the population density at the current location
                density_value = self._density_map.get((x, y))
                
                # Use the density value to probabilistically place victims
                if np.random.rand() < density_value: 
                    victim_map[x, y] = 1  # Place a victim

        # Return the generated victim map as a BaseMap instance
        return BaseMap(victim_map, padding=self._padding)
