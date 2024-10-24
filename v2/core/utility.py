from __future__ import annotations

from .agent_state import AgentState, _UtilityType
from .env_params import *
from abc import ABC, abstractmethod

import numpy as np 


class UtilityFunction(ABC):
    """
    Base class that models the utility function of a specific agent
    """
    @abstractmethod
    def forward(self, state : AgentState) -> _UtilityType:
        """
        Calculates the utility given the current agent state. Mutates the agent's current utility 
        """
        pass 

    @abstractmethod
    def update(self):
        """
        Updates the utility function parameters 
        """
        pass 
