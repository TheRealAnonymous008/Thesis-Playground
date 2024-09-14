from __future__ import annotations
from typing import Callable

import numpy as np
from .agent import Agent
from .observation import LocalObservation
from .action import ActionInformation, Direction
from .map import MapGenerator, ResourceMap, Resource
from .models import *

class World: 
    def __init__(self, 
                 dims : tuple[int, int],
                 swarm_initialzier : Callable, 
                 resource_generator : MapGenerator,
                 energy_model : EnergyModel = None,
                 chemistry_model : ChemistryModel = None,
        ):
        """
        `dims`: Dimensions of the world (x, y) form.
        """
        if dims[0] <= 0 or dims[1] <= 0:
            raise Exception(f"Incorrect dimensions {dims}")
        
        self._dims : tuple[int, int] = dims
        self._world_grid = np.zeros(dims, dtype = np.int32)

        self._agents : dict[int, Agent] = {}
        self._swarm_initializer : Callable = swarm_initialzier
        self._resource_generator : MapGenerator = resource_generator

        self._energy_model : EnergyModel | None = energy_model
        self._chemistry_model : ChemistryModel | None = chemistry_model
        self.reset()

    def reset(self):
        """
        Reset the environment
        """
        self._agents.clear()
        self._time_step = 0 
        self._nagents = 0

        # Generate a new resource map 
        self._resource_map : ResourceMap  = self._resource_generator.generate(self._dims)
        self._resource_grid = self._resource_map.type_map

        self._swarm_initializer(self)

        for agent in self.agents: 
            agent.reset()

    def update(self):
        """
        Update the environment 
        """
        # Get a working list of agents and shuffle them
        agents = self.agents
        self._update_movement(agents)
        self._update_agent_actuation(agents)

        # Get the current world state
        self._world_state = self._get_world_state()
        self._resource_grid = self._resource_map._resource_type_map

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
            nearby_resources = self._get_nearby_resources(agent, visibility_range)
            observation = LocalObservation(nearby_agents, nearby_resources)

            agent.set_observation(observation)

        self._time_step += 1

    def _get_world_state(self):
        return self.get_presence_mask(self.agents)


    def _update_movement(self, agents : list[Agent]):
        movement_mask = np.zeros(self._dims, dtype=bool)

        for agent in agents:
            position_const = agent.current_position_const
            movement_mask[position_const[0]][position_const[1]] = True 

        for agent in agents: 
            action : ActionInformation = agent.action
            # Note: We are being careful here to perfectly undo the current position. That way, we maintain invariants in the world.
            current_position = agent.current_position_const
            movement_mask[current_position[0], current_position[1]] = False 

            if action.movement != None: 
                dir_movement = Direction.get_direction_of_movement(action.movement)
                current_position += dir_movement

                if not self.is_traversable(current_position) or movement_mask[current_position[0], current_position[1]]:
                    current_position -= dir_movement

            movement_mask[current_position[0], current_position[1]] = True 
            agent.set_position(current_position)

    def _update_agent_actuation(self, agents : list[Agent]): 
        for agent in agents:
            action : ActionInformation = agent.action
            if action.pick_up != None: 
                # Note: We carefully undo the adding of dir_action to pos
                pos = agent.current_position_const 
                dir_action = Direction.get_direction_of_movement(action.pick_up)

                pos += dir_action

                if self.is_in_bounds(pos):
                    resource : Resource = self._resource_map.subtract_resource(pos, 1)
                    if resource.quantity != 0:
                        qty = agent.add_to_inventory(resource)
                        self._resource_map.add_resource(pos, resource.type, qty)

                pos -= dir_action

            elif action.production_job != None and self._chemistry_model != None: 
                self._chemistry_model.forward(agent, action.production_job)


    def _get_nearby_agents(self, agent: Agent, visibility_range: int) -> np.ndarray:
        """
        Gets all agents nearby using the visibility range. 
        """
        x, y = agent.current_position_const
        x_min, x_max, y_min, y_max = max(0, x - visibility_range), min(self._dims[0], x + visibility_range + 1), \
              max(0, y - visibility_range), min(self._dims[1], y + visibility_range + 1)

        # We assume the world state is final and will not change 
        observation = self._world_state[x_min:x_max, y_min:y_max]
        return observation

    def _get_nearby_resources(self, agent : Agent, visibility_range : int) -> np.ndarray:
        """
        Gets all resources nearby using the visibility range. 
        """
        x, y = agent.current_position_const
        x_min, x_max, y_min, y_max = max(0, x - visibility_range), min(self._dims[0], x + visibility_range + 1), \
              max(0, y - visibility_range), min(self._dims[1], y + visibility_range + 1)

        # We assume the resource grid is final and will not change 
        observation = self._resource_grid[x_min:x_max, y_min:y_max]
        observation[x - x_min, y - y_min] = 0
        return observation

    # Functionalities  
    def add_agent(self, agent : Agent):
        """
        Adds an `agent` to the environment
        """
        self._nagents += 1
        agent.bind_to_world(self._nagents)
        self._agents[agent.id] = agent

    def get_agent(self, id : int) -> Agent:
        """
        Returns the agent with the given `id`
        """
        return self._agents[id]

    def remove_agent(self, agent : Agent):
        """
        Removes `agent` from the environment. Assumes `agent` is in the environment
        """
        self._agents.pop(agent.id)

    def is_in_bounds(self, position : np.ndarray) -> bool:
        """
        Check whether a position is in the bounds of the world or not.
        """
        return position[0] >= 0 and position[1] >= 0 and position[0] < self._dims[0] and position[1] < self._dims[1]
    
    def is_traversable(self, position: np.ndarray) -> bool:
        """
        Check whether a position is traversable to any agent
        """
        return self.is_in_bounds(position) and self._resource_map.get_type((position[0], position[1])) == 0
    
    def get_presence_mask(self, agents : list[Agent]) -> np.ndarray:
        """
        Returns a mask in the shape of the dims of the world that shows where agents are.
        """
        presence_mask = np.zeros(self._dims, dtype=np.int32)
        for agent in agents: 
            pos_const = agent.current_position_const
            x, y = pos_const[0], pos_const[1]
            presence_mask[x, y] = agent.id

        return presence_mask

    @property
    def agents(self) -> list[Agent]:
        """
        Returnns a list of all agents in the world
        """
        return list(self._agents.values())
    
    @property
    def resource_map(self) -> ResourceMap:
        """
        Returns a copy of the resource map
        """
        return self._resource_map.copy
    
    @property
    def total_cell_count(self) -> int:
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