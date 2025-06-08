import numpy as np
from gymnasium import spaces
from models.base_env import BaseEnv

class TypeInferenceEnvironment(BaseEnv):
    def __init__(self, n_agents, n_types, total_games=1, type_distribution=None):
        """
        Environment for type inference with private types and belief matrices.
        
        Args:
            n_agents (int): Number of agents (must be even)
            n_types (int): Size of type space |T|
            total_games (int): Number of games per episode
            type_distribution (np.ndarray): Probability distribution over types
        """
        super().__init__(n_agents, d_actions=n_types, d_traits=1, d_beliefs=n_types)
        
        assert n_agents % 2 == 0, "Number of agents must be even"
        self.n_types = n_types
        self.total_games = total_games
        
        # Type distribution (uniform by default)
        self.type_distribution = (
            type_distribution if type_distribution is not None 
            else np.ones(n_types) / n_types
        )
        
        # Define spaces
        self.obs_size = self.d_traits
        self.action_space = spaces.Discrete(n_types)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_size,),
            dtype=np.float32
        )
        
        # Agent identifiers
        self.agents = list(range(n_agents))
        self.current_step = 0
        self.types = np.zeros(n_agents, dtype=int)

        self.reset()

    def reset(self):
        """Reset environment with new private types and cleared beliefs"""
        super().reset()
        self.current_step = 0
        
        # Sample private types for all agents
        self.types = np.random.choice(
            self.n_types, 
            size=self.n_agents,
            p=self.type_distribution
        )
        
        # Set traits to agent types
        self.traits[:, 0] = self.types / self.n_types
        
        # Initialize beliefs as uniform distributions
        for i in range(self.n_agents):
            self.beliefs[i] = np.ones(self.n_types) / self.n_types
        
        # Create fixed pairwise connections
        for i in range(0, self.n_agents, 2):
            j = i + 1
            self.graph.add_edge(i, j, np.zeros((self.d_relation,)))
            self.graph.add_edge(j, i, np.zeros((self.d_relation,)))
        
        return self._get_observations()

    def step(self, actions):
        """
        Execute one timestep with type estimation and belief updates
        
        Args:
            actions (dict): Agent actions (type estimates)
            
        Returns:
            observations, rewards, dones, infos
        """
        rewards = {}
        dones = {}
        infos = {"true_types": self.types.copy()}
        
        # Calculate rewards based on estimation accuracy
        for i in range(0, self.n_agents, 2):
            j = i + 1
            
            # Reward = 1 if correct estimation, 0 otherwise
            a = 1.0 if actions[i] == self.types[j] else 0.0
            b = 1.0 if actions[j] == self.types[i] else 0.0

            rewards[i] = a 
            rewards[j] = b


        # Update step counter
        self.current_step += 1
        done = self.current_step >= self.total_games
        dones = {agent: done for agent in self.agents}
        
        return self._get_observations(), rewards, dones, {agent: {} for agent in self.agents}

    def _get_observations(self):
        observations = {}
        for agent in self.agents:
            obs = np.concatenate([
                [self.traits[agent, 0]],  # Own type
            ]).astype(np.float32)
            observations[agent] = obs
        return observations

    def get_agents(self):
        return self.agents

    def get_traits(self):
        return self.traits
    
import numpy as np
from gymnasium import spaces
from models.base_env import BaseEnv

class TypeDeceptionEnvironment(BaseEnv):
    def __init__(self, n_agents, n_types, total_games=1, type_distribution=None):
        """
        Environment for type inference with private types and belief matrices.
        
        Args:
            n_agents (int): Number of agents (must be even)
            n_types (int): Size of type space |T|
            total_games (int): Number of games per episode
            type_distribution (np.ndarray): Probability distribution over types
        """
        super().__init__(n_agents, d_actions=n_types, d_traits=1, d_beliefs=n_types)
        
        assert n_agents % 2 == 0, "Number of agents must be even"
        self.n_types = n_types
        self.total_games = total_games
        
        # Type distribution (uniform by default)
        self.type_distribution = (
            type_distribution if type_distribution is not None 
            else np.ones(n_types) / n_types
        )
        
        # Define spaces
        self.obs_size = self.d_traits
        self.action_space = spaces.Discrete(n_types)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_size,),
            dtype=np.float32
        )
        
        # Agent identifiers
        self.agents = list(range(n_agents))
        self.current_step = 0
        self.types = np.zeros(n_agents, dtype=int)

        self.reset()

    def reset(self):
        """Reset environment with new private types and cleared beliefs"""
        super().reset()
        self.current_step = 0
        
        # Sample private types for all agents
        self.types = np.random.choice(
            self.n_types, 
            size=self.n_agents,
            p=self.type_distribution
        )
        
        # Set traits to agent types
        self.traits[:, 0] = self.types / self.n_types
        
        # Initialize beliefs as uniform distributions
        for i in range(self.n_agents):
            self.beliefs[i] = np.ones(self.n_types) / self.n_types
        
        # Create fixed pairwise connections
        for i in range(0, self.n_agents, 2):
            j = i + 1
            self.graph.add_edge(i, j, np.zeros((self.d_relation,)))
            self.graph.add_edge(j, i, np.zeros((self.d_relation,)))
        
        return self._get_observations()

    def step(self, actions):
        """
        Execute one timestep with type estimation and belief updates
        
        Args:
            actions (dict): Agent actions (type estimates)
            
        Returns:
            observations, rewards, dones, infos
        """
        rewards = {}
        dones = {}
        infos = {"true_types": self.types.copy()}
        
        # Calculate rewards based on estimation accuracy
        for i in range(0, self.n_agents, 2):
            j = i + 1
            
            # Reward = 1 if correct estimation, 0 otherwise
            a = 1.0 if actions[i] == self.types[j] else 0.0
            b = 1.0 if actions[j] == self.types[i] else 0.0

            rewards[i] = -2 * b 
            rewards[j] = b

        # Update step counter
        self.current_step += 1
        done = self.current_step >= self.total_games
        dones = {agent: done for agent in self.agents}
        
        return self._get_observations(), rewards, dones, {agent: {} for agent in self.agents}

    def _get_observations(self):
        observations = {}
        for agent in self.agents:
            obs = np.concatenate([
                [self.traits[agent, 0]],  # Own type
            ]).astype(np.float32)
            observations[agent] = obs
        return observations

    def get_agents(self):
        return self.agents

    def get_traits(self):
        return self.traits