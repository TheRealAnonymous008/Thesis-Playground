from core.map import *
from core.models import *

from sar.sar_agent import * 
from sar.sar_env_params import * 
from core.world import BaseWorld

class VictimModel(BaseDynamicsModel):
    def __init__(self):
        pass 

    def forward(self, world : BaseWorld) -> float: 
        """
        Check if the agent is on a victim
        """
        victim_map = world.get_map("Victims")
        exploration_map = world.get_map("Exploration")
        for agent in world.agents:
            agent : SARAgent = agent 
            position = agent.current_position_const

            exploration_map.set(position, 1)

            if victim_map.get(position) != 0:
                agent.rescue()
                victim_map.set(position, 0)
            else: 
                agent._current_state.just_rescued_victim = 0
                


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

    def generate(self, dims: tuple[int, int], number: int = 100) -> BaseMap:
        """
        Generate a map of the victims in the area based on population density.
        The more densely populated areas will have a higher number of victims, 
        while ensuring the exact specified number is placed randomly.
        """
        if self._density_map is None:
            raise ValueError("Density map has not been set.")
        
        # Flatten the density map into a list of coordinates and their probabilities
        density_list = []
        for x in range(dims[0]):
            for y in range(dims[1]):
                density_value = self._density_map.get((x, y))
                density_list.append((x, y, density_value))
        
        # Normalize density values into probabilities
        total_density = sum(d[2] for d in density_list)
        if total_density == 0:
            raise ValueError("Density map contains no valid density values.")
        probabilities = [d[2] / total_density for d in density_list]
        
        # Randomly select `number` coordinates based on probabilities
        chosen_indices = np.random.choice(len(density_list), size=number, replace=False, p=probabilities)
        selected_coords = [density_list[i][:2] for i in chosen_indices]
        
        # Create victim map and populate selected coordinates
        victim_map = np.zeros(dims)
        for x, y in selected_coords:
            victim_map[x, y] = 1  # Place a victim at the chosen location
        
        # Return the generated victim map as a BaseMap instance
        return BaseMap(victim_map, padding=self._padding)
