import numpy as np
from gymnasium import spaces
from models.base_env import BaseEnv

class BaselineSimpleCommunication(BaseEnv):
    def __init__(self, n_agents, payoff_i, payoff_j, belief_dims = 8, total_games = 1):
        super().__init__(n_agents, payoff_i.shape[0])

        # Validate payoff matrices
        assert len(payoff_i.shape) == 2 and len(payoff_j.shape) == 2, "Payoff matrices must be 2D"
        assert payoff_i.shape == payoff_j.shape, "Payoff matrices must have the same shape"
        
        self.payoff_i = payoff_i
        self.payoff_j = payoff_j
        self.total_games = total_games
        self.belief_dims = belief_dims
        
        # Define observation space with flattened payoff matrices and an indexer so agents know which player they are.
        self.obs_size = 2 * (self.n_actions ** 2) + 1
        self.action_space = spaces.Discrete(self.n_actions)
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
            assert 0 <= action < self.n_actions, f"Invalid action {action} for {agent}"
        
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