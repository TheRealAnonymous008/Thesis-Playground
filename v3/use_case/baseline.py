from models.base_env import *

class BaselineEnvironment(BaseEnv):
    def __init__(self, n_agents, payoff_i, payoff_j, total_games=1):
        super().__init__(n_agents, payoff_i.shape[0])

        # Validate payoff matrices
        assert len(payoff_i.shape) == 2 and len(payoff_j.shape) == 2, "Payoff matrices must be 2D"
        assert payoff_i.shape == payoff_j.shape, "Payoff matrices must have the same shape"
        
        self.payoff_i = payoff_i
        self.payoff_j = payoff_j
        self.total_games = total_games
        
        # Define observation space with flattened payoff matrices and an indexer so agents know which player they are.
        self.obs_size = 2 * (self.n_actions ** 2) + 2
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
        """Resets environment with payoff matrices and traits as observations"""
        super().reset()
        self.total_steps = 0
        self.traits = np.array([[-1] if i % 2 == 0 else [1] for i in range(self.n_agents)], dtype=np.float16)

        # Initialize observations with payoff matrices and traits
        observations = {}
        for agent in self.agents:
            if agent % 2 == 0:
                partner = agent + 1
                if partner < self.n_agents:
                    payoff_a = self.payoff_i.flatten()
                    payoff_b = self.payoff_j.flatten()
                    trait_self = self.traits[agent][0]
                    trait_partner = self.traits[partner][0]
                else:
                    # No partner, default partner trait to 0
                    payoff_a = self.payoff_i.flatten()
                    payoff_b = self.payoff_j.flatten()
                    trait_self = self.traits[agent][0]
                    trait_partner = 0.0
            else:
                partner = agent - 1
                if partner >= 0:
                    payoff_a = self.payoff_j.flatten()
                    payoff_b = self.payoff_i.flatten()
                    trait_self = self.traits[agent][0]
                    trait_partner = self.traits[partner][0]
                else:
                    # No partner (unlikely, handle for completeness)
                    payoff_a = self.payoff_j.flatten()
                    payoff_b = self.payoff_i.flatten()
                    trait_self = self.traits[agent][0]
                    trait_partner = 0.0

            obs = np.concatenate([
                payoff_a,
                payoff_b,
                [trait_self],
                [trait_partner]
            ]).astype(np.float32)
            observations[agent] = obs

        # Reset graph edges
        for i in range(0, self.n_agents, 2):
            if i + 1 >= self.n_agents:
                break
            self.graph.add_edge(i, i + 1, np.zeros((self.d_relation,)))
            self.graph.add_edge(i + 1, i, np.zeros((self.d_relation,)))

        return observations

    def step(self, actions):
        """
        Executes one timestep with pairwise interactions and payoff-based observations
        """
        # Generate random pairs and initialize observations
        pairs = []
        observations = {agent: np.zeros(self.obs_size, dtype=np.float32) for agent in self.agents}
        rewards = {agent: 0.0 for agent in self.agents}

        # Create pairwise interactions
        for i in range(0, self.n_agents, 2):
            if i + 1 >= self.n_agents:
                break
                
            a, b = i, i + 1
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
                [self.traits[a][0]],
                [self.traits[b][0]]
            ]).astype(np.float32)
            
            obs_b = np.concatenate([
                self.payoff_j.flatten(),
                self.payoff_i.flatten(),
                [self.traits[b][0]],
                [self.traits[a][0]]
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
    

    
    