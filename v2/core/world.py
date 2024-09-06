from __future__ import annotations

import numpy as np
from .agent import Agent
from .action import ActionInformation, Direction

class World: 
    def __init__(self, 
                 dims : tuple[int, int]
        ):
        """
        `dims`: Dimensions of the world (x, y) form.
        """
        if dims[0] <= 0 or dims[1] <= 0:
            raise Exception(f"Incorrect dimensions {dims}")
        
        self._dims : tuple[int, int] = dims
        self._agents : set[Agent] = set()
        self._time_step = 0

    def reset(self):
        """
        Reset the environment
        """
        self._agents.clear()
        self._time_step = 0 

    def update(self):
        """
        Update the environment 
        """
        # Get a working list of agents and shuffle them
        agent_turn_order = list(self._agents)
        self._update_movement(agent_turn_order)

        for agent in self._agents:
            agent.reset_for_next_action()

        self._time_step += 1

    def _update_movement(self, agents : list[Agent]):
        movement_mask = np.zeros(self._dims, dtype=bool)

        for agent in agents: 
            action : ActionInformation = agent._current_action
            if action.movement != None: 
                dir_movement = Direction.get_direction_of_movement(action.movement)
                new_position = agent.get_position()

                new_position[0] += dir_movement[0]
                new_position[1] += dir_movement[1]

                if not self.is_in_bounds(new_position) or not movement_mask[new_position[0]][new_position[1]] == False:
                    new_position = agent.get_position()
                
                movement_mask[new_position[0]][new_position[1]] = True 
                agent.set_position(new_position)

    # Functionalities  
    def add_agent(self, agent : Agent):
        """
        Adds an `agent` to the environment
        """
        self._agents.add(agent)

    def remove_agent(self, agent : Agent):
        """
        Removes `agent` from the environment. Assumes `agent` is in the environment
        """
        self._agents.remove(agent) 

    def is_in_bounds(self, position : np.ndarray) -> bool:
        """
        Check whether a position is in the bounds of the world or not.
        """
        return position[0] >= 0 and position[1] >= 0 and position[0] < self._dims[0] and position[1] < self._dims[1]


def initialize_positions_randomly(world : World, swarm : list[Agent]):
    positions = set()
    while len(positions) < len(swarm):
        x, y = np.random.randint(0, world._dims[0]), np.random.randint(0, world._dims[1])
        positions.add((x, y))

    for agent, pos in zip(swarm, positions):
        agent.set_position(np.array([pos[0], pos[1]], dtype = np.int32))

    return swarm