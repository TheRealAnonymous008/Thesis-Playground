from typing import *
import numpy as np
from gymnasium import spaces, Env

class BaseEnv:
    def __init__(self, n_agents):
        # Validate payoff matrices
        self.n_agents = n_agents 


    def reset(self) -> dict[int, Any]:
        pass 

    def step(self, actions) -> tuple[Any, SupportsFloat, dict[int, Any], dict[int, Any]]:
        """
        """
        pass 

    def get_traits(self) -> np.ndarray:
        pass 

    def get_agents(self) -> list:
        pass 

    def get_beliefs(self) -> np.ndarray: 
        pass

    def set_beliefs(self, idx : int, belief : np.ndarray):
        pass 

    def get_belief_matrix(self) -> np.ndarray: 
        pass 