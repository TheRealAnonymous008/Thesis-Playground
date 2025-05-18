from typing import *
import numpy as np
from gymnasium import spaces, Env

import torch
from typing import Dict, Any, Union
import random
from .param_settings import ParameterSettings

class Graph:
    def __init__(self, n_agents: int):
        if n_agents <= 0:
            raise ValueError("n_agents must be a positive integer.")
        self.n_agents = n_agents
        self.adj: Dict[int, Dict[int, Dict[str, Any]]] = {u: {} for u in range(n_agents)}

    def add_edge(self, u: int, v: int, **attributes: Any) -> None:
        if not (0 <= u < self.n_agents) or not (0 <= v < self.n_agents):
            raise ValueError(f"Nodes must be between 0 and {self.n_agents - 1}.")
        self.adj[u][v] = {**self.adj[u].get(v, {}), **attributes}

    def remove_edge(self, u: int, v: int) -> None:
        if 0 <= u < self.n_agents and 0 <= v < self.n_agents:
            if v in self.adj[u]:
                del self.adj[u][v]

    def get_neighbors(self, u: int) -> Dict[int, Dict[str, Any]]:
        if not (0 <= u < self.n_agents):
            raise ValueError(f"Node {u} is out of range.")
        return {v: attrs.copy() for v, attrs in self.adj[u].items()}

    def random_init(self, p: float, **default_attrs: Any) -> None:
        for u in range(self.n_agents):
            for v in range(self.n_agents):
                if random.random() < p:
                    self.add_edge(u, v, **default_attrs)

eps = 1e-8

class BaseEnv:
    def __init__(self, n_agents, d_traits, d_beliefs, d_comm_state):
        # Validate payoff matrices
        self.n_agents = n_agents 
        self.d_traits = d_traits
        self.d_beliefs = d_beliefs 
        self.d_comm_state = d_comm_state

    def reset(self) -> dict[int, Any]:
        self.graph = Graph(self.n_agents)
        self.traits = np.zeros((self.n_agents, self.d_traits), dtype = np.float16)
        self.beliefs = np.zeros((self.n_agents, self.d_beliefs), dtype = np.float16)
        self.comm_state = np.zeros((self.n_agents, self.d_comm_state) , dtype = np.float16) 

    def step(self, actions) -> tuple[Any, SupportsFloat, dict[int, Any], dict[int, Any]]:
        pass 

    def get_traits(self) -> np.ndarray:
        pass 

    def get_agents(self) -> list:
        pass 

    def set_beliefs(self, belief :  torch.Tensor):
        self.beliefs = belief.detach().cpu().numpy()

    def get_belief_matrix(self) -> np.ndarray: 
        pass 