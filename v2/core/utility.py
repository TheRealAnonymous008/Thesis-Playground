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
        Calculates the utility given the current agent state
        """
        u : float = 0
        for i in range(self.params.shape[0]): 
            u += state.get_qty_in_inventory(i)
        state.current_utility = u        

    def update(self):
        """
        Updates the utility function parameters 
        """
        pass 
