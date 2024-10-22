from __future__ import annotations

import numpy as np
from .observation import LocalObservation

from .action import *
from .message import * 

from .resource import Resource, _QuantityType, _ResourceType
from .utility import UtilityFunction
from .agent_state import *
from .direction import Direction
from .message import Message


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
        self._initializer()
        self.reset()

    def _initializer(self, *args, **kwargs):
        """
        Code for initializing the agent. Derived classes should override this. 
        """
        self._current_observation : LocalObservation = None
        self._current_action : ActionInformation = ActionInformation()
        self._traits : AgentTraits = AgentTraits()
        self._current_state :  AgentState = AgentState()
        self._utility_function : UtilityFunction = None 
        
    def reset(self):
        """
        Reset the agent
        """
        self._reset()
        self._current_state.reset(self._traits)

    def _reset(self):
        """
        Code for reseting the agennt. Derived classes should override this
        """
        pass

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

    def calculate_utility(self):
        """
        Evaluate this agent's utility
        """
        self._utility_function.forward(self._current_state)

    @property
    def agents_in_range(self) -> list[_IdType]:
        """
        Returns a list of id's of all visible agents.
        """
        return self._current_observation.neighbors(self.id)

    @property
    def local_observation(self) -> LocalObservation:
        """
        Get local observation
        """
        return self._current_observation

    @property
    def action(self) -> ActionInformation:
        """
        Return the current action of the agent 
        """
        return self._current_action
    
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

    @property
    def utility(self) : 
        return self._current_state.current_utility