from __future__ import annotations
from typing import Callable, TYPE_CHECKING

from core.world import *
from dynamics.agents.sar_agent import *

class SARWorld(BaseWorld): 
    """
    Class responsible for simulating the SAR environemnt 
    """
    def __init__(self, 
                 dims : tuple[int, int],
                 swarm_initializer : Callable, 
                 generation_pipeline : Callable,
        ):
        """
        :param dims: The dimensions of the world.
        :param swarm_initializer: A function to initialize a swarm of agents.
        """
        super().__init__(dims, swarm_initializer, generation_pipeline)
        

    def _reset(self):
        self._movement_mask = np.zeros(self._dims, dtype=bool)

    def _get_world_state(self):
        return self.get_presence_mask(self.agents)

    def _pre_update(self):
        """
        Perform any initialization work
        """
        # First, pre-shuffle the agents
        np.random.shuffle(self.agents)

        # Generate movement masks for the agents
        self.movement_mask = np.zeros(self._dims, dtype=bool)
        for agent in self.agents:
            agent : SARAgent = agent
            position_const = agent.current_position_const
            self.movement_mask[position_const[0]][position_const[1]] = True 

    def _update_agent_actuation(self): 
        """
        Resolve all agent actions here.
        """
        for agent in self.agents:
            agent : SARAgent = agent
            action : SARActionInformation  = agent.action 

            # Do stuff with agent actions
            if action.movement != None: 
                current_position = agent.current_position_const
                self.movement_mask[current_position[0], current_position[1]] = False 
                dir_movement = Direction.get_direction_of_movement(action.movement)
                current_position += dir_movement
                
                if not self.is_traversable(current_position) or \
                    self.movement_mask[current_position[0], current_position[1]] or \
                    self._maps.get("Terrain").get_gradient(current_position, action.movement) > agent._traits._max_slope:
                    current_position -= dir_movement

                self.movement_mask[current_position[0], current_position[1]] = True 
                agent.set_position(current_position)


    def _post_update(self):
        super()._post_update()


    def _update_agent_sensors(self, agent : SARAgent):
        """
        Gives the agent its observations
        """
        super()._update_agent_sensors(agent)

    def _get_nearby_agents(self, agent: SARAgent, visibility_range: int) -> np.ndarray:
        """
        Gets all agents nearby using the visibility range. 
        """
        x, y = agent.current_position_const
        x_min, x_max = max(0, x - visibility_range), min(self._dims[0], x + visibility_range + 1)
        y_min, y_max = max(0, y - visibility_range), min(self._dims[1], y + visibility_range + 1)

        # We assume the resource grid is final and will not change 
        observation = self._world_state[x_min:x_max, y_min:y_max]

        return observation

    def is_traversable(self, position: np.ndarray) -> bool:
        """
        Check whether a position is traversable to any agent
        """
        return self.is_in_bounds(position)
    
    def get_presence_mask(self, agents : list[SARAgent]) -> np.ndarray:
        """
        Returns a mask in the shape of the dims of the world that shows where agents are.
        """
        presence_mask = np.zeros(self._dims, dtype=np.int32)
        for agent in agents: 
            pos_const = agent.current_position_const
            x, y = pos_const[0], pos_const[1]
            presence_mask[x, y] = agent.id

        return presence_mask

    
def initialize_positions_randomly(world: SARWorld, swarm: list[SARAgent]):
    positions : list[tuple[int, int]] = []

    # Add all traversable cells to the positions set
    for x in range(world._dims[0]):
        for y in range(world._dims[1]):
            if world.is_traversable(np.array([x, y])):
                positions.append((x, y))

    # Randomly sample positions for the agents
    sampled_position_idx = np.random.choice(len(positions), size=len(swarm), replace=False)

    for agent, idx in zip(swarm, sampled_position_idx):
        pos = positions[idx]
        agent.set_position(np.array([pos[0], pos[1]], dtype=np.int32))

    return swarm