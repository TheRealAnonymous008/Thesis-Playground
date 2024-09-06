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
        self._world_grid = np.zeros(dims, dtype = np.int32)

        self._agents : dict[int, Agent] = {}
        self.reset()

    def reset(self):
        """
        Reset the environment
        """
        self._agents.clear()
        self._time_step = 0 
        self._nagents = 0

    def update(self):
        """
        Update the environment 
        """

        # Get a working list of agents and shuffle them
        agents = self.get_agents()
        self._update_movement(agents)
        
        self._world_state = self._get_world_state()

        for agent in agents:
            # Reset the agents
            agent.reset_for_next_action()

            # Give them the observations they can see in the environment
            visibility_range = agent._visibility_range  
            nearby_agents = self._get_nearby_agents(agent, visibility_range)
            agent.set_observation(nearby_agents)

        self._time_step += 1

    def _get_world_state(self):
        return self.get_presence_mask(self.get_agents())


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

    def _get_nearby_agents(self, agent: Agent, visibility_range: int) -> np.ndarray:
        """
        Gets all agents nearby using the visibility range. 
        """
        agent_pos = agent.get_position()
        x, y = agent_pos[0], agent_pos[1]

        # Calculate the boundaries of the observation grid (clipping to world bounds)
        x_min = max(0, x - visibility_range)
        x_max = min(self._dims[0], x + visibility_range + 1)
        y_min = max(0, y - visibility_range)
        y_max = min(self._dims[1], y + visibility_range + 1)

        # Return the sliced observation grid
        return np.array(self._world_state[x_min:x_max, y_min:y_max])

    # Functionalities  
    def add_agent(self, agent : Agent):
        """
        Adds an `agent` to the environment
        """
        agent.bind_to_world(self._nagents)
        self._agents[agent.get_id()] = agent
        self._nagents += 1

    def remove_agent(self, agent : Agent):
        """
        Removes `agent` from the environment. Assumes `agent` is in the environment
        """
        self._agents.pop(agent.get_id())

    def is_in_bounds(self, position : np.ndarray) -> bool:
        """
        Check whether a position is in the bounds of the world or not.
        """
        return position[0] >= 0 and position[1] >= 0 and position[0] < self._dims[0] and position[1] < self._dims[1]
    
    def get_presence_mask(self, agents : list[Agent]) -> np.ndarray:
        """
        Returns a mask in the shape of the dims of the world that shows where agents are.
        """
        presence_mask = np.zeros(self._dims, dtype=np.int32)
        for agent in agents: 
            pos = agent.get_position()
            x, y = pos[0], pos[1]
            presence_mask[x][y] = 1

        return presence_mask

    def get_agents(self) -> list[Agent]:
        """
        Returnns a list of all agents in the world
        """
        return list(self._agents.values())
    

def initialize_positions_randomly(world : World, swarm : list[Agent]):
    positions = set()
    while len(positions) < len(swarm):
        x, y = np.random.randint(0, world._dims[0]), np.random.randint(0, world._dims[1])
        positions.add((x, y))

    for agent, pos in zip(swarm, positions):
        agent.set_position(np.array([pos[0], pos[1]], dtype = np.int32))

    return swarm