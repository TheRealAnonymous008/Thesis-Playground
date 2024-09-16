from __future__ import annotations

from .agent_state import AgentState, _UtilityType
from .env_params import *

import numpy as np 


class UtilityFunction:
    """
    Models the utility function of a specific agent
    """
    def __init__(self, params : np.ndarray): 
        self.params =  np.ndarray = params 

    def forward(self, state : AgentState) -> _UtilityType:
        """
        Calculates the utility given the current agent state. Mutates the agent's current utility 
        """
        utility = np.zeros_like(self.params)
        utility[list(state.inventory.keys())] = list(state.inventory.values())
        state.current_utility = self.params.dot(utility)

    def update(self):
        """
        Updates the utility function parameters 
        """
        pass 
