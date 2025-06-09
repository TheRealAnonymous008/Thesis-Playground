from models.base_env import *

class NetBasedEnvironment(BaseEnv):
    def __init__(self, n_agents, num_types, k=4, p=0.1, episode_length = 10):
        """
        Environment for type-based matching with small-world graph structure.
        
        Args:
            n_agents (int): Number of agents
            num_types (int): Number of distinct types (T)
            k (int): Degree for small-world graph (must be even)
            p (float): Rewiring probability for small-world graph
            episode_length (int): Total steps per episode
        """
        # Action space: 0 (A) or 1 (B)
        super().__init__(n_agents, d_actions=2)
        
        self.n_types = num_types
        self.k = k
        self.p = p
        self.episode_length = episode_length
        
        # Observation: agent's own type (integer)
        self.obs_size = 1 + self.d_relation 
        self.action_space = spaces.Discrete(2)  # 2 actions: A(0), B(1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_size,),
            dtype=np.float32
        )
        
        self.agents = list(range(n_agents))
        self.total_steps = 0
        self.current_pairs = []  # Stores current agent pairs

    def reset(self):
        """Initialize environment with random types and small-world graph"""
        super().reset()
        self.total_steps = 0
        
        # Assign random types to agents
        self.traits = np.random.randint(0, self.n_types, size=(self.n_agents, 1)).astype(np.float16)
        self.types = self.traits[:, 0].astype(int)
        
        # Build small-world graph
        self._build_small_world_graph()
        
        # Sample initial pairs
        self._sample_pairs()
        
        # Observations: each agent sees only its own type
        return self._get_observations()

    def _build_small_world_graph(self):
        """Construct Watts-Strogatz small-world graph"""
        n = self.n_agents
        
        # Create ring lattice
        for i in range(n):
            for j in range(1, self.k//2 + 1):
                neighbors = [
                    (i + j) % n,
                    (i - j) % n
                ]
                for nb in neighbors:
                    self.graph.add_edge(i, nb, np.zeros((self.d_relation,)))
                    self.graph.add_edge(nb, i, np.zeros((self.d_relation,)))
        
        # Rewiring step
        for i in range(n):
            for j in range(1, self.k//2 + 1):
                if random.random() < self.p:
                    # Remove original edge
                    nb_original = (i + j) % n
                    self.graph.remove_edge(i, nb_original)
                    self.graph.remove_edge(nb_original, i)
                    
                    # Find new connection
                    possible_targets = [
                        k for k in range(n) 
                        if k != i
                    ]
                    if possible_targets:
                        new_nb = random.choice(possible_targets)
                        self.graph.add_edge(i, new_nb, np.zeros((self.d_relation,)))
                        self.graph.add_edge(new_nb, i, np.zeros((self.d_relation,)))

    def _sample_pairs(self):
        """Randomly pair agents using graph neighbors"""
        unpaired = set(self.agents)
        pairs = []
        
        # Process in random order
        agents = list(unpaired)
        random.shuffle(agents)
        
        for agent in agents:
            if agent not in unpaired:
                continue
                
            # Find available neighbors
            neighbors = [
                nb for nb in self.graph.get_neighbors(agent) 
                if nb in unpaired and nb != agent
            ]
            
            if neighbors:
                partner = random.choice(neighbors)
                pairs.append((agent, partner))
                unpaired.discard(agent)
                unpaired.discard(partner)
        
        self.current_pairs = pairs

    def step(self, actions):
        """
        Process actions and compute rewards based on:
        - Same type + mutual A: +1
        - Different type + B: +1
        - Other cases: -1
        """
        rewards = {agent: 0.0 for agent in self.agents}

        ctr = 0
        # Process current pairs (sampled in previous step)
        for i, j in self.current_pairs:
            a_i, a_j = actions[i], actions[j]
            t_i, t_j = self.types[i], self.types[j]
            
            if t_i == t_j:
                ctr += 1
                rewards[i] = 2.0 if a_i == 1 else -2.0
                rewards[j] = 2.0 if a_j == 1 else -2.0
            else:
                rewards[i] = 1.0 if a_i == 0 else -2.0
                rewards[j] = 1.0 if a_j == 0 else -2.0
        
        # Sample new pairs for next step
        self._sample_pairs()
        
        # Prepare observations (unchanged)
        observations = self._get_observations()
        
        # Update step counter
        self.total_steps += 1
        done = self.total_steps >= self.episode_length
        
        return (
            observations,
            rewards,
            {agent: done for agent in self.agents},
            {agent: {} for agent in self.agents}
        )
    
    def get_agents(self):
        return self.agents
    
    def get_traits(self):
        return self.traits
    
    def _get_observations(self):
        partner_of = {}
        for i, j in self.current_pairs:
            partner_of[i] = j
            partner_of[j] = i
        
        observations = {}
        for agent in self.agents:
            # Start with own type
            obs = np.zeros(self.obs_size, dtype=np.float32)
            obs[0] = self.types[agent]
            
            # Add relation vector if agent is paired
            if agent in partner_of:
                partner = partner_of[agent]
                neighbors = self.graph.get_neighbors(agent)
                if partner in neighbors:
                    obs[1:] = neighbors[partner]  # Use existing relation vector
            
            observations[agent] = obs
        return observations