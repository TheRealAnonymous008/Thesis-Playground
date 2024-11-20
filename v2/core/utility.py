from __future__ import annotations

from .agent_state import AgentState, _UtilityType
from abc import ABC, abstractmethod

import numpy as np 

from typing import TYPE_CHECKING 

if TYPE_CHECKING: 
    from .agent import Agent

class UtilityFunction(ABC):
    """
    Base class that models the utility function of a specific agent
    """
    @abstractmethod
    def dense_forward(self, agent : Agent) -> _UtilityType:
        """
        Calculates the utility given the current agent state.
        """
        pass 

    @abstractmethod
    def sparse_forward(self, agent:  Agent ) -> _UtilityType:
        """
        Calculates the utility given the current agent state
        Used for when the agent is terminated and a distinct reward at the end is needed 
        """
        return self.dense_forward(agent)

    @abstractmethod
    def update(self):
        """
        Updates the utility function parameters 
        """
        pass 
