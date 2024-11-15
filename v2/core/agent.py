from __future__ import annotations

import numpy as np
from .agent_state import AgentState, AgentTraits
from .observation import LocalObservation

from .action import *
from .message import * 

from .utility import UtilityFunction
from .agent_state import *
from .message import Message

from abc import ABC, abstractmethod

import torch

_IdType = int
class Agent:
    """
    Agent Class. Override or augment this class as needed.

    This is meant to be a wrapper that allows interfacing between actions and states.
    """

    def __init__(self):
        """
        Initializes a simple agent 
        """
        self._id : _IdType = "-1"
        self._current_belief : np.ndarray = 0
        self._device = "cpu"

        self._initializer()
        self.reset()

    @abstractmethod
    def _initializer(self, *args, **kwargs):
        """
        Code for initializing the agent. 
        """
        self._current_observation : LocalObservation = None
        self._current_action : ActionInformation = None 
        self._traits : AgentTraits = AgentTraits() 
        self._current_state :  AgentState = None
        self._utility_function : UtilityFunction = None 

    @abstractmethod
    def to(self, device : str):
        """
        Set the device of the agent for use in CUDA
        """
        self._device = device
        
    def reset(self):
        """
        Reset the agent
        """
        self._reset()
        self._current_state.reset(self._traits)

    @abstractmethod
    def _reset(self):
        """
        Code for reseting the agenn.
        """
        pass 

    @abstractmethod
    def update(self):
        """
        Update the agent's state
        """
        pass 

    def send_message(self, agent : Agent, message : Message):
        """
        Send a message to the target agent
        """
        agent._current_state.msgs.append(message)

    def get_messages(self) -> list[Message]:
        """
        Return all messages 
        """
        return self._current_state.msgs

    def clear_messages(self):
        """
        Clear all messages 
        """
        self._current_state.clear_messages() 
    
    def add_relation(self, agent_id : _IdType, weight : float):
        """
        Add a social relation between this agent and the one specified
        """
        self._current_state.set_relation(agent_id, weight  + self._current_state.get_relation(agent_id))

    def reset_for_next_action(self):
        """
        Resets the agent for a new action
        """
        self._current_action.reset()

    def set_observation(self, observation : LocalObservation):
        """
        Set local observation
        """
        self._current_observation = observation

    def set_utility(self, utility : UtilityFunction) :
        """
        Set utility functionn
        """
        self._utility_function = utility

    @property
    def utility(self):
        """
        Evaluate this agent's utility
        """
        return self._utility_function.forward(self._current_state)

    @property
    def agents_in_range(self) -> list[_IdType]:
        """
        Returns a list of id's of all visible agents.
        """
        return self._current_observation.neighbors(self.id)

    @property
    @abstractmethod
    def local_observation(self) -> LocalObservation:
        """
        Get local observation
        """
        return self._current_observation

    @property
    @abstractmethod
    def action(self) -> ActionInformation:
        """
        Return the current action of the agent 
        """
        return self._current_action
    
    @property 
    @abstractmethod 
    def trait_as_tensor(self) -> torch.Tensor: 
        return self._traits.to_tensor()
    
    @property 
    @abstractmethod
    def state_as_tensor(self) -> torch.Tensor: 
        pass 


    def bind_to_world(self, world_id : int):
        """
        Set the agent's ID to be that of the ID assigned to it by the world
        """
        self._id = _IdType(world_id)

    @property
    def id(self) -> _IdType:
        """
        Return the ID of this agent in the world
        """
        return self._id 

    def __hash__(self) -> int:
        return int(self._id)
