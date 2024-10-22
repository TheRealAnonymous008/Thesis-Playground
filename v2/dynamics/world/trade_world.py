from __future__ import annotations
from typing import Callable, TYPE_CHECKING

import numpy as np
from core.agent import Agent, _IdType
from core.observation import LocalObservation
from core.action import ActionInformation, Direction
from core.env_params import MAX_VISIBILITY

if TYPE_CHECKING: 
    from core.models import BaseDynamicsModel

from dynamics.space.terrain_map import *


class World: 
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
        if dims[0] <= 0 or dims[1] <= 0:
            raise Exception(f"Incorrect dimensions {dims}")
        
        self._dims : tuple[int, int] = dims
        self._world_grid = np.zeros(dims, dtype = np.int32)

        self._agents : dict[_IdType, Agent] = {}
        self._swarm_initializer : Callable = swarm_initializer
        self._generation_pipeline : Callable = generation_pipeline
        self._maps : BaseMapCollection = None

        self._models : dict[str, BaseDynamicsModel] = {}
        
        self.reset()

    def add_model(self, name : str, model : BaseDynamicsModel) -> World: 
        self._models[name] = model
        return self  

    def reset(self):
        """
        Reset the environment
        """
        self._agents.clear()
        self._time_step = 0 
        self._nagents = 0

        self._maps = self._generation_pipeline(self)
        self._movement_mask = np.zeros(self._dims, dtype=bool)

        self._swarm_initializer(self)

        for agent in self.agents: 
            agent.reset()

    def update(self):
        """
        Update the environment 
        """
        self._pre_update()
        self._update_agent_actuation()
        self._post_update()

        # Get the current world state
        self._world_state = self._get_world_state()        

        for agent in self.agents:
            # Reset the agents
            agent.reset_for_next_action()
            self._update_agent_sensors(agent)

        self._time_step += 1

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
            action : ActionInformation = agent.action 

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
        """
        Perform any clean up after updating but before passing all information to agents.
        """
        # Update any models that we have
        for model in self._models.values():
            model.forward(self)


    def _update_agent_sensors(self, agent : Agent):
        """
        Gives the agent its observations
        """
        visibility_range = agent._traits._visibility_range  
        nearby_agents = self._get_nearby_agents(agent, visibility_range)
        observation = LocalObservation(nearby_agents)

        agent.set_observation(observation)

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

    # Functionalities  
    def add_agent(self, agent : Agent):
        """
        Adds an `agent` to the environment
        """
        self._nagents += 1
        agent.bind_to_world(self._nagents)
        self._agents[agent.id] = agent

    def get_agent(self, id : _IdType) -> Agent:
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
        return self.is_in_bounds(position)
    
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
    def agent_aliases(self) -> list[_IdType]: 
        """
        Returns a list of all the agent ids
        """
        return list(self._agents.keys())
    
    def get_map(self, name : str) -> BaseMap | None: 
        return self._maps.get(name).copy
    
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