import numpy as np
from gymnasium import spaces

class BaselineEnvironment:
    def __init__(self, n_agents, payoff_i, payoff_j, total_games = 1):
        # Validate payoff matrices
        assert len(payoff_i.shape) == 2 and len(payoff_j.shape) == 2, "Payoff matrices must be 2D"
        assert payoff_i.shape == payoff_j.shape, "Payoff matrices must have the same shape"
        
        self.n_agents = n_agents
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
        
        # Initialize action counts
        self.action_counts = [np.zeros(self.num_actions, dtype=int) for _ in range(n_agents)]
        self.total_steps = 0

    def reset(self):
        """Resets environment with zero-initialized payoff observations"""
        for counts in self.action_counts:
            counts.fill(0)
        self.total_steps = 0
        return {agent: np.zeros(self.obs_size, dtype=np.float32) for agent in self.agents}

    def step(self, actions):
        """
        Executes one timestep with pairwise interactions and payoff-based observations
        """
        # Validate actions
        for agent, action in actions.items():
            assert 0 <= action < self.num_actions, f"Invalid action {action} for {agent}"
        
        # Generate random pairs and initialize observations
        agent_indices = np.random.permutation(self.n_agents)
        pairs = []
        observations = {agent: np.zeros(self.obs_size, dtype=np.float32) for agent in self.agents}
        rewards = {agent: 0.0 for agent in self.agents}

        # Create pairwise interactions
        for i in range(0, self.n_agents, 2):
            if i+1 >= self.n_agents:
                break
                
            a, b = agent_indices[i], agent_indices[i+1]
            pairs.append((a, b))
            
            # Get actions and update counts
            action_a = actions[a]
            action_b = actions[b]
            self.action_counts[a][action_a] += 1
            self.action_counts[b][action_b] += 1
            
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