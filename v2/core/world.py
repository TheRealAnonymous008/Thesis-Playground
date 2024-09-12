from __future__ import annotations

import numpy as np
from .agent import Agent
from .observation import LocalObservation
from .action import ActionInformation, Direction
from .map import MapGenerator, ResourceMap
from .models import *

class World: 
    def __init__(self, 
                 dims : tuple[int, int],
                 resource_generator : MapGenerator,
                 energy_model : EnergyModel = None
        ):
        """
        `dims`: Dimensions of the world (x, y) form.
        """
        if dims[0] <= 0 or dims[1] <= 0:
            raise Exception(f"Incorrect dimensions {dims}")
        
        self._dims : tuple[int, int] = dims
        self._world_grid = np.zeros(dims, dtype = np.int32)

        self._agents : dict[int, Agent] = {}
        self._resource_generator : MapGenerator = resource_generator

        self._energy_model : EnergyModel | None = energy_model
        self.reset()

    def reset(self):
        """
        Reset the environment
        """
        self._agents.clear()
        self._time_step = 0 
        self._nagents = 0

        # Generate a new resource map 
        self._resource_grid : ResourceMap  = self._resource_generator.generate(self._dims)

    def update(self):
        """
        Update the environment 
        """
        # Get a working list of agents and shuffle them
        agents = self.get_agents()
        self._update_movement(agents)
        self._update_agent_actuation(agents)

        # Get the current world state
        self._world_state = self._get_world_state()

        # Update any models that we have
        if self._energy_model != None: 
            for agent in agents:
                self._energy_model.forward(agent)


        for agent in agents:
            # Reset the agents
            agent.reset_for_next_action()

            # Give them the observations they can see in the environment
            visibility_range = agent._visibility_range  
            nearby_agents = self._get_nearby_agents(agent, visibility_range)
            observation = LocalObservation(nearby_agents)

            agent.set_observation(observation)

        self._time_step += 1

    def _get_world_state(self):
        return self.get_presence_mask(self.get_agents())


    def _update_movement(self, agents : list[Agent]):
        movement_mask = np.zeros(self._dims, dtype=bool)

        for agent in agents:
            position = agent.get_current_position()
            movement_mask[position[0]][position[1]] = True 

        for agent in agents: 
            action : ActionInformation = agent.get_action()
            old_position = agent.get_current_position()
            if action.movement != None: 
                dir_movement = Direction.get_direction_of_movement(action.movement)
                new_position = agent.get_current_position()

                new_position[0] += dir_movement[0]
                new_position[1] += dir_movement[1]

                if not self.is_traversable(new_position) or not movement_mask[new_position[0]][new_position[1]] == False:
                    new_position = agent.get_current_position()
            else: 
                new_position = agent.get_current_position()

            movement_mask[old_position[0]][old_position[1]] = False 
            movement_mask[new_position[0]][new_position[1]] = True 
            agent.set_position(new_position)

    def _update_agent_actuation(self, agents : list[Agent]): 
        for agent in agents:
            action : ActionInformation = agent.get_action()
            if action.pick_up != None: 
                pos = agent.get_current_position()
                dir_action = Direction.get_direction_of_movement(action.pick_up)

                pos[0] += dir_action[0]
                pos[1] += dir_action[1]

                if self.is_in_bounds(pos):
                    # TODO: Add it to the agent's inventory
                    self._resource_grid.subtract_resource(pos, 1)


    def _get_nearby_agents(self, agent: Agent, visibility_range: int) -> np.ndarray:
        """
        Gets all agents nearby using the visibility range. 
        """
        agent_pos = agent.get_current_position()
        x, y = agent_pos[0], agent_pos[1]

        # Calculate the boundaries of the observation grid (clipping to world bounds)
        x_min = max(0, x - visibility_range)
        x_max = min(self._dims[0], x + visibility_range + 1)
        y_min = max(0, y - visibility_range)
        y_max = min(self._dims[1], y + visibility_range + 1)

        # Get the sliced observation grid
        observation = np.array(self._world_state[x_min:x_max, y_min:y_max])

        # Mask the agent's own position with 0
        observation[x - x_min, y - y_min] = 0
        return observation

    # Functionalities  
    def add_agent(self, agent : Agent):
        """
        Adds an `agent` to the environment
        """
        self._nagents += 1
        agent.bind_to_world(self._nagents)
        self._agents[agent.get_id()] = agent

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
    
    def is_traversable(self, position: np.ndarray) -> bool:
        """
        Check whether a position is traversable to any agent
        """
        return self.is_in_bounds(position) and self._resource_grid.get_type((position[0], position[1])) == 0
    
    def get_presence_mask(self, agents : list[Agent]) -> np.ndarray:
        """
        Returns a mask in the shape of the dims of the world that shows where agents are.
        """
        presence_mask = np.zeros(self._dims, dtype=np.int32)
        for agent in agents: 
            pos = agent.get_current_position()
            x, y = pos[0], pos[1]
            presence_mask[x][y] = agent.get_id()

        return presence_mask

    def get_agents(self) -> list[Agent]:
        """
        Returnns a list of all agents in the world
        """
        return list(self._agents.values())
    
    def get_resource_map(self) -> ResourceMap:
        """
        Returns a copy of the resource map
        """
        return self._resource_grid.copy()
    
    def get_total_cell_count(self) -> int:
        """
        Returns the number of cells in the world
        """
        return self._dims[0] * self._dims[1]
    
def initialize_positions_randomly(world: World, swarm: list[Agent]):
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