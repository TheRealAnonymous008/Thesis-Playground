from __future__ import annotations

import numpy as np
from .agent import Agent

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

        self._time_step += 1

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


def initialize_positions_randomly(world : World, swarm : list[Agent]):
    positions = set()
    while len(positions) < len(swarm):
        x, y = np.random.randint(0, world._dims[0]), np.random.randint(0, world._dims[1])
        positions.add((x, y))

    for agent, pos in zip(swarm, positions):
        agent.set_position(pos)

    return swarm