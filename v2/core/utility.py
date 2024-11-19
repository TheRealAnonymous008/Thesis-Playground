from __future__ import annotations

from .agent_state import AgentState, _UtilityType
from abc import ABC, abstractmethod

import numpy as np 


class UtilityFunction(ABC):
    """
    Base class that models the utility function of a specific agent
    """
    @abstractmethod
    def dense_forward(self, state : AgentState) -> _UtilityType:
        """
        Calculates the utility given the current agent state.
        """
        pass 

    @abstractmethod
    def sparse_forward(self, state: AgentState) -> _UtilityType:
        """
        Calculates the utility given the current agent state
        Used for when the agent is terminated and a distinct reward at the end is needed 
        """
        return self.dense_forward(state)

    @abstractmethod
    def update(self):
        """
        Updates the utility function parameters 
        """
        pass 
