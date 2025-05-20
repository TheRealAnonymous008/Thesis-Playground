from typing import *
import numpy as np
import torch
import random

class Graph:
    def __init__(self, n_agents: int, d_edge: int):
        if n_agents <= 0:
            raise ValueError("n_agents must be a positive integer.")
        self.n_agents = n_agents
        self.d_edge = d_edge
        self.adj: Dict[int, Dict[int, np.ndarray]] = {u: {} for u in range(n_agents)}

    def add_edge(self, u: int, v: int, edge_tensor: np.ndarray) -> None:
        if not (0 <= u < self.n_agents) or not (0 <= v < self.n_agents):
            raise ValueError(f"Nodes must be between 0 and {self.n_agents - 1}.")
        if edge_tensor.shape != (self.d_edge,):
            raise ValueError(f"Edge tensor must have shape ({self.d_edge},), got {edge_tensor.shape}")
        self.adj[u][v] = edge_tensor

    def remove_edge(self, u: int, v: int) -> None:
        if 0 <= u < self.n_agents and 0 <= v < self.n_agents:
            if v in self.adj[u]:
                del self.adj[u][v]

    def get_neighbors(self, u: int) -> Dict[int, np.ndarray]:
        if not (0 <= u < self.n_agents):
            raise ValueError(f"Node {u} is out of range.")
        return {v: np.copy(tensor) for v, tensor in self.adj[u].items()}

    def update_edges(self, sources: np.ndarray, destinations: np.ndarray, new_edges: np.ndarray) -> None:
        if len(sources) != len(destinations) or len(sources) != new_edges.shape[0]:
            raise ValueError("Mismatched number of sources, destinations, and new_edges.")
        if new_edges.shape[1] != self.d_edge:
            raise ValueError(f"New edges must be of shape (N, {self.d_edge}), got {new_edges.shape}.")
        for u, v, edge in zip(sources, destinations, new_edges):
            if 0 <= u < self.n_agents and 0 <= v < self.n_agents:
                if v in self.adj[u]:
                    self.adj[u][v] = edge.copy()

eps = 1e-8

class BaseEnv:
    def __init__(self, n_agents, d_traits=1, d_beliefs=8, d_comm_state=8, d_relation=4):
        self.n_agents = n_agents 
        self.d_traits = d_traits
        self.d_beliefs = d_beliefs 
        self.d_comm_state = d_comm_state
        self.d_relation = d_relation

    def reset(self) -> dict[int, Any]:
        self.graph = Graph(self.n_agents, self.d_relation)
        self.traits = np.zeros((self.n_agents, self.d_traits), dtype=np.float32)
        self.beliefs = np.zeros((self.n_agents, self.d_beliefs), dtype=np.float32)
        self.comm_state = np.zeros((self.n_agents, self.d_comm_state), dtype=np.float32) 

    def step(self, actions) -> tuple[Any, SupportsFloat, dict[int, Any], dict[int, Any]]:
        pass 

    def get_traits(self) -> np.ndarray:
        pass 

    def get_agents(self) -> list:
        pass 

    def set_beliefs(self, belief: torch.Tensor):
        self.beliefs = belief.detach().cpu().numpy()

    def set_comm_state(self, indices, states : torch.Tensor):
        unique_indices, inverse_indices = torch.unique(indices, 
                                                    return_inverse=True,
                                                    return_counts=False)

        # Create mask for scatter operation
        mask = torch.zeros((len(indices), len(unique_indices)), 
                        device=states.device)
        mask[torch.arange(len(indices)), inverse_indices] = 1


        aggregated_zdj = torch.mm(mask.T, states)

        # Update the z tensor
        modified_z = states.clone()
        modified_z[unique_indices] = aggregated_zdj

        # Create final results
        modified_indices = unique_indices

        self.comm_state[modified_indices] = modified_z.detach().cpu().numpy()

    def update_edges(self, sources: torch.Tensor, destinations: torch.Tensor, new_edges: torch.Tensor) -> None:
        sources_np = sources.detach().cpu().numpy()
        destinations_np = destinations.detach().cpu().numpy()
        new_edges_np = new_edges.detach().cpu().numpy()
        self.graph.update_edges(sources_np, destinations_np, new_edges_np)

    def sample_neighbors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sampled_indices = []
        sampled_edges = []
        sampled_reverse_edges = []
        for u in range(self.n_agents):
            neighbors = self.graph.get_neighbors(u)
            if not neighbors:
                sampled_indices.append(u)
                edge = np.zeros((self.d_relation,), dtype=np.float32)
                reverse = np.zeros((self.d_relation,), dtype=np.float32)
            else:
                neighbor = random.choice(list(neighbors.keys()))
                sampled_indices.append(neighbor)
                edge = neighbors[neighbor]
                reverse =  self.graph.get_neighbors(neighbor).get(u, np.zeros((self.d_relation,), dtype=np.float32))

            sampled_edges.append(edge)
            sampled_reverse_edges.append(reverse)
        indices_tensor = torch.tensor(sampled_indices, dtype=torch.long)
        edges_tensor = torch.tensor(np.stack(sampled_edges), dtype=torch.float32)
        reverse_tensor = torch.tensor(np.stack(sampled_reverse_edges), dtype=torch.float32)
        return indices_tensor, edges_tensor, reverse_tensor