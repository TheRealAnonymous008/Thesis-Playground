import numpy as np
from gymnasium import spaces
from models.base_env import BaseEnv

class BaselineHeterogeneous(BaseEnv):
    def __init__(self, n_agents, n_types, type_payoffs, total_games=1):
        """
        The type payoffs should have shape
        (n_types, n_types, 2, n_actions, n_actions)
        """
        super().__init__(n_agents, type_payoffs.shape[3], d_traits=n_types)

        # Validate type_payoffs structure
        assert len(type_payoffs.shape) == 5
        assert type_payoffs.shape[0] == n_types and type_payoffs.shape[1] == n_types
        assert type_payoffs.shape[2] == 2
        assert type_payoffs.shape[3] == type_payoffs.shape[4]
        
        self.n_types = n_types
        self.type_payoffs = type_payoffs
        self.total_games = total_games
        
        self.obs_size = 2  # Observation contains own type and opponent's type
        self.action_space = spaces.Discrete(self.n_actions)
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
        """Resets environment with agent's type and partner's type as observations"""
        super().reset()
        self.total_steps = 0
        # Generate one-hot encoded traits based on agent_types
        self.traits = np.eye(self.n_types, dtype=np.float16)[self.agent_types]

        # Create random pairs and add edges to the graph
        shuffled = np.random.permutation(self.n_agents)
        for i in range(0, len(shuffled), 2):
            if i + 1 >= len(shuffled):
                break
            a = shuffled[i]
            b = shuffled[i + 1]
            # Add edges in both directions
            self.graph.add_edge(a, b, np.zeros((self.d_relation,)))
            self.graph.add_edge(b, a, np.zeros((self.d_relation,)))

        # Build observations with own type and partner's type
        observations = {}
        for agent in self.agents:
            own_type = self.agent_types[agent]
            observations[agent] = np.array([own_type, -1], dtype=np.float32)
        
        return observations

    def step(self, actions):
        """
        Executes one timestep with pairwise interactions based on the graph structure
        """
        # Validate actions
        for agent, action in actions.items():
            assert 0 <= action < self.n_actions, f"Invalid action {action} for {agent}"
        
        # Generate pairs from the graph (u < v to avoid duplicates)
        pairs = []
        for u in self.agents:
            neighbors = self.graph.get_neighbors(u)
            for v in neighbors:
                if u < v:
                    pairs.append((u, v))
        
        observations = {agent: np.zeros(self.obs_size, dtype=np.float32) for agent in self.agents}
        rewards = {agent: 0.0 for agent in self.agents}

        # Process each pair based on the graph
        for a, b in pairs:
            # Retrieve actions
            action_a = actions[a]
            action_b = actions[b]

            # Get agent types and corresponding payoffs
            t1 = self.agent_types[a]
            t2 = self.agent_types[b]
            payoff_i = self.type_payoffs[t1, t2, 0]
            payoff_j = self.type_payoffs[t1, t2, 1]
            
            # Assign rewards
            rewards[a] = payoff_i[action_a, action_b]
            rewards[b] = payoff_j[action_a, action_b]
            
            # Set observations with own type and opponent's type
            observations[a] = np.array([t1, t2], dtype=np.float32)
            observations[b] = np.array([t2, t1], dtype=np.float32)

        self.total_steps += 1
        done = self.total_steps >= self.total_games
        
        return observations, rewards, {agent: done for agent in self.agents}, {agent: {} for agent in self.agents}
    
    def get_agents(self):
        return self.agents