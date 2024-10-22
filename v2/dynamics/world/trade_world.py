from __future__ import annotations
from typing import Callable, TYPE_CHECKING

from core.world import *
from dynamics.agents.trade_agent import *

class TradeWorld(BaseWorld): 
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
            position_const = agent.current_position_const
            self.movement_mask[position_const[0]][position_const[1]] = True 

    def _update_agent_actuation(self): 
        """
        Resolve all agent actions here.
        """
        for agent in self.agents:
            action : TradeActionInformation = agent.action 

            # TODO: Implement this

    def _post_update(self):
        """
        Perform any clean up after updating but before passing all information to agents.
        """
        super()._post_update()

    def _update_agent_sensors(self, agent : Agent):
        """
        Gives the agent its observations
        """
        super()._update_agent_sensors(agent)

    def _get_nearby_agents(self, agent: Agent, visibility_range: int) -> np.ndarray:
        """
        Gets all agents nearby using the visibility range. 
        """
        x, y = agent.current_position_const
        x_min, x_max = max(0, x - visibility_range), min(self._dims[0], x + visibility_range + 1)
        y_min, y_max = max(0, y - visibility_range), min(self._dims[1], y + visibility_range + 1)

        # We assume the resource grid is final and will not change 
        observation = self._world_state[x_min:x_max, y_min:y_max]

        return observation

    def get_presence_mask(self, agents : list[TradeAgent]) -> np.ndarray:
        """
        Returns a mask in the shape of the dims of the world that shows where agents are.
        """
        presence_mask = np.zeros(self._dims, dtype=np.int32)
        for agent in agents: 
            pos_const = agent.current_position_const
            x, y = pos_const[0], pos_const[1]
            presence_mask[x, y] = agent.id

        return presence_mask

def initialize_positions_randomly(world: TradeWorld, swarm: list[TradeAgent]):
    positions : list[tuple[int, int]] = []

    # Add all traversable cells to the positions set
    for x in range(world._dims[0]):
        for y in range(world._dims[1]):
            if world.is_in_bounds(np.array([x, y])):
                positions.append((x, y))

    # Randomly sample positions for the agents
    sampled_position_idx = np.random.choice(len(positions), size=len(swarm), replace=False)

    for agent, idx in zip(swarm, sampled_position_idx):
        pos = positions[idx]
        agent.set_position(np.array([pos[0], pos[1]], dtype=np.int32))

    return swarm