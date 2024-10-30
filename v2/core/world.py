from __future__ import annotations
from typing import Callable, TYPE_CHECKING

import numpy as np
from .agent import Agent, _IdType
from .map import * 

if TYPE_CHECKING: 
    from .models import BaseDynamicsModel



from abc import *

class BaseWorld(ABC):
    """
    Class responsible for simulating the environemnt .
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

    def add_model(self, name : str, model : BaseDynamicsModel) -> BaseWorld: 
        self._models[name] = model
        return self  

    def reset(self):
        """
        Reset the environment
        """
        self._agents.clear()
        self._time_step = 0 
        self._nagents = 0
        self._reset()

        self._maps = self._generation_pipeline(self)
        self._swarm_initializer(self)
        for agent in self.agents: 
            agent.reset()

    @abstractmethod
    def _reset(self):
        """
        Additional reset logic implemented by derived classes
        """
        pass

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

    @abstractmethod
    def _get_world_state(self):
        pass

    @abstractmethod
    def _pre_update(self):
        """
        Perform any initialization work
        """
        pass 

    @abstractmethod
    def _update_agent_actuation(self): 
        """
        Resolve all agent actions here.
        """
        pass 

    @abstractmethod
    def _post_update(self):
        """
        Perform any clean up after updating but before passing all information to agents.
        """
        # Update any models that we have
        for model in self._models.values():
            model.forward(self)

    @abstractmethod
    def _update_agent_sensors(self, agent : Agent):
        """
        Gives the agent its observations
        """
        pass

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
        return self._maps.get(name)
    
    @property
    def total_cell_count(self) -> int:
        """
        Returns the number of cells in the world
        """
        return self._dims[0] * self._dims[1]