import numpy as np
from gymnasium import spaces
from models.base_env import BaseEnv
import torch

class BaselineEnvironment(BaseEnv):
    def __init__(self, n_agents, payoff_i, payoff_j, total_games = 1):
        super().__init__(n_agents, 1, 8, 1)

        # Validate payoff matrices
        assert len(payoff_i.shape) == 2 and len(payoff_j.shape) == 2, "Payoff matrices must be 2D"
        assert payoff_i.shape == payoff_j.shape, "Payoff matrices must have the same shape"
        
        self.payoff_i = payoff_i
        self.payoff_j = payoff_j
        self.num_actions = payoff_i.shape[0]
        self.total_games = total_games
        
        # Define observation space with flattened payoff matrices and an indexer so agents know which player they are.
        self.obs_size = 2 * (self.num_actions ** 2) + 1
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_size,),
            dtype=np.float32
        )
        
        # Agent identifiers
        self.agents = [i for i in range(n_agents)]
        
        self.total_steps = 0

    def reset(self):
        """Resets environment with zero-initialized payoff observations"""
        super().reset()
        self.total_steps = 0
        self.traits = np.array([[-1] if i % 2 ==0 else [1] for i in range(self.n_agents)], dtype = np.float16)

        for i in range(0, self.n_agents - 1, 2):
            self.graph.add_edge(i, i + 1, np.zeros((self.d_relation, ))) 
        for i in range(1, self.n_agents, 2):
            self.graph.add_edge(i, i - 1, np.zeros((self.d_relation, ))) 

        return {agent: np.zeros(self.obs_size, dtype=np.float32) for agent in self.agents}

    def step(self, actions):
        """
        Executes one timestep with pairwise interactions and payoff-based observations
        """
        # Validate actions
        for agent, action in actions.items():
            assert 0 <= action < self.num_actions, f"Invalid action {action} for {agent}"
        
        # Generate random pairs and initialize observations
        pairs = []
        observations = {agent: np.zeros(self.obs_size, dtype=np.float32) for agent in self.agents}
        rewards = {agent: 0.0 for agent in self.agents}

        # Create pairwise interactions
        for i in range(0, self.n_agents, 2):
            if i+1 >= self.n_agents:
                break
                
            a, b = i, i+1
            pairs.append((a, b))
            
            # Get actions and update counts
            action_a = actions[a]
            action_b = actions[b]

            # Calculate rewards
            rewards[a] = self.payoff_i[action_a, action_b]
            rewards[b] = self.payoff_j[action_a, action_b]
            
            # Generate observations with both payoff matrices
            obs_a = np.concatenate([
                self.payoff_i.flatten(),
                self.payoff_j.flatten(), 
                [self.traits[a][0]]
            ]).astype(np.float32)
            
            obs_b = np.concatenate([
                self.payoff_j.flatten(),
                self.payoff_i.flatten(),
                [self.traits[b][0]]
            ]).astype(np.float32)
            
            observations[a] = obs_a
            observations[b] = obs_b

        self.total_steps += 1
        done = self.total_steps >= self.total_games

        
        return observations, rewards, {agent: done for agent in self.agents}, {agent: {} for agent in self.agents}
    
    def get_agents(self):
        return self.agents
    
    def get_traits(self):
        return self.traits

class BaselineHeterogeneous(BaseEnv):
    def __init__(self, n_agents, n_types, type_payoffs, total_games=1):
        """
        The type payoffs should have shape
        (n_types, n_types, 2, n_actions, n_actions)
        """
        super().__init__(n_agents, d_traits = n_types, d_beliefs = 8, d_comm_state = 8, d_relation = 1)

        # Validate type_payoffs structure
        assert len(type_payoffs.shape) == 5
        assert type_payoffs.shape[0] == n_types and type_payoffs.shape[1] == n_types
        assert type_payoffs.shape[2] == 2
        self.num_actions = type_payoffs.shape[3]
        assert type_payoffs.shape[3] == type_payoffs.shape[4]
        
        self.n_types = n_types
        self.type_payoffs = type_payoffs
        self.total_games = total_games
        
        self.obs_size = 2  
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_size,),
            dtype=np.float32
        )
        
        # Agent identifiers and types
        self.agents = [i for i in range(n_agents)]
        self.agent_types = np.random.randint(0, n_types, size=n_agents)
        
        self.total_steps = 0

    def reset(self):
        """Resets environment with zero-initialized payoff observations and one-hot trait vectors"""
        super().reset()
        self.total_steps = 0
        # Generate one-hot encoded traits based on agent_types
        self.traits = np.eye(self.n_types, dtype=np.float16)[self.agent_types]
        return {agent: np.zeros(self.obs_size, dtype=np.float32) for agent in self.agents}

    def step(self, actions):
        """
        Executes one timestep with pairwise interactions and indexer-based observations
        """
        # Validate actions
        for agent, action in actions.items():
            assert 0 <= action < self.num_actions, f"Invalid action {action} for {agent}"
        
        # Generate random pairs and initialize observations
        pairs = []
        observations = {agent: np.zeros(self.obs_size, dtype=np.float32) for agent in self.agents}
        rewards = {agent: 0.0 for agent in self.agents}

        # Create pairwise interactions
        for i in range(0, self.n_agents, 2):
            if i+1 >= self.n_agents:
                break
                
            a, b = i, i+1
            pairs.append((a, b))
            
            # Get actions and update counts
            action_a = actions[a]
            action_b = actions[b]

            # Retrieve agent types and corresponding payoffs
            t1 = self.agent_types[a]
            t2 = self.agent_types[b]
            payoff_i = self.type_payoffs[t1, t2, 0]
            payoff_j = self.type_payoffs[t1, t2, 1]
            
            # Calculate rewards
            rewards[a] = payoff_i[action_a, action_b]
            rewards[b] = payoff_j[action_a, action_b]
            
            # Generate observations with indexer (agent's own type)
            observations[a] = np.array([self.agent_types[a], self.agent_types[b]], dtype=np.float32)
            observations[b] = np.array([self.agent_types[b], self.agent_types[a]], dtype=np.float32)

        self.total_steps += 1
        done = self.total_steps >= self.total_games
        
        return observations, rewards, {agent: done for agent in self.agents}, {agent: {} for agent in self.agents}
    
    def get_agents(self):
        return self.agents
    
    
class BaselineSimpleCommunication(BaseEnv):
    def __init__(self, n_agents, payoff_i, payoff_j, belief_dims = 8, total_games = 1):
        super().__init__(n_agents)

        # Validate payoff matrices
        assert len(payoff_i.shape) == 2 and len(payoff_j.shape) == 2, "Payoff matrices must be 2D"
        assert payoff_i.shape == payoff_j.shape, "Payoff matrices must have the same shape"
        
        self.payoff_i = payoff_i
        self.payoff_j = payoff_j
        self.num_actions = payoff_i.shape[0]
        self.total_games = total_games
        self.belief_dims = belief_dims
        
        # Define observation space with flattened payoff matrices and an indexer so agents know which player they are.
        self.obs_size = 2 * (self.num_actions ** 2) + 1
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_size,),
            dtype=np.float32
        )
        
        # Agent identifiers
        self.agents = [i for i in range(n_agents)]
        
        self.total_steps = 0

    def reset(self):
        """Resets environment with zero-initialized payoff observations"""
        self.total_steps = 0
        self.traits = np.array([[-1] if i % 2 ==0 else [1] for i in range(self.n_agents)], dtype = np.float16)
        self.beliefs = np.zeros((self.n_agents, self.belief_dims))
        return {agent: np.zeros(self.obs_size, dtype=np.float32) for agent in self.agents}

    def step(self, actions):
        """
        Executes one timestep with pairwise interactions and payoff-based observations
        """
        # Validate actions
        for agent, action in actions.items():
            assert 0 <= action < self.num_actions, f"Invalid action {action} for {agent}"
        
        # Generate random pairs and initialize observations
        pairs = []
        observations = {agent: np.zeros(self.obs_size, dtype=np.float32) for agent in self.agents}
        rewards = {agent: 0.0 for agent in self.agents}

        # Create pairwise interactions
        for i in range(0, self.n_agents, 2):
            if i+1 >= self.n_agents:
                break
                
            a, b = i, i+1
            pairs.append((a, b))
            
            # Get actions and update counts
            action_a = actions[a]
            action_b = actions[b]

            # Calculate rewards
            rewards[a] = self.payoff_i[action_a, action_b]
            rewards[b] = self.payoff_j[action_a, action_b]
            
            # Generate observations with both payoff matrices
            obs_a = np.concatenate([
                self.payoff_i.flatten(),
                self.payoff_j.flatten(), 
                [-1]
            ]).astype(np.float32)
            
            obs_b = np.concatenate([
                self.payoff_j.flatten(),
                self.payoff_i.flatten(),
                [1]
            ]).astype(np.float32)
            
            observations[a] = obs_a
            observations[b] = obs_b

        self.total_steps += 1
        done = self.total_steps >= self.total_games
        
        return observations, rewards, {agent: done for agent in self.agents}, {agent: {} for agent in self.agents}
    
    def get_agents(self):
        return self.agents
    
    def set_beliefs(self, i : int, belief : np.ndarray):
        self.beliefs[i] = belief