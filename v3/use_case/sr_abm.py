from models.base_env import *

class DiseaseSpreadEnv(BaseEnv):
    def __init__(self, n_agents: int, d_relation: int = 4, beta: float = 0.5, 
                 k: int = 4, p: float = 0.1, episode_length: int = 50):
        """
        Environment for disease spread simulation using SI model.
        
        Args:
            n_agents: Number of agents
            d_relation: Dimension of edge features
            beta: Disease transmission rate
            k: Degree for small-world graph (must be even)
            p: Rewiring probability for small-world graph
            episode_length: Total steps per episode
        """
        super().__init__(
            n_agents=n_agents,
            d_actions=1,  # Continuous action (interaction duration)
            d_traits=3,    # [alpha, rho, p_s]
            d_beliefs=1,   # Unused (minimal size)
            d_comm_state=1,  # Unused (minimal size)
            d_relation=d_relation,
            obs_size=4 + d_relation  # [own_state, own_symptom, partner_state, partner_symptom] + edge
        )
        self.beta = beta
        self.k = k
        self.p = p
        self.episode_length = episode_length
        self.agents = list(range(n_agents))
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )

    def reset(self) -> Dict[int, np.ndarray]:
        super().reset()  # Initializes graph, traits, etc.
        self.total_steps = 0
        
        # Initialize agent states (0=susceptible, 1=infected)
        self.states = np.zeros(self.n_agents, dtype=np.float32)
        initial_infected = np.random.choice(self.n_agents, size=max(1, self.n_agents//20), replace=False)
        self.states[initial_infected] = 1.0
        
        # Initialize agent traits: [alpha, rho, p_s]
        self.traits = np.random.uniform(
            low=[0.1, 0.5, 0.1], 
            high=[1.0, 2.0, 0.9],
            size=(self.n_agents, 3)
        ).astype(np.float32)
        
        # Build social network
        self._build_small_world_graph()
        # Generate initial symptoms
        self.symptoms = self._generate_symptoms()
        # Form initial pairs
        self.current_pairs = self._sample_pairs()
        return self._get_observations()

    def _build_small_world_graph(self):
        n = self.n_agents
        # Create ring lattice
        for i in range(n):
            for j in range(1, self.k//2 + 1):
                neighbors = [(i+j) % n, (i-j) % n]
                for nb in neighbors:
                    edge_feat = np.zeros(self.d_relation, dtype=np.float32)
                    edge_feat[0] = 1.0  # First component as edge weight
                    self.graph.add_edge(i, nb, edge_feat)
                    self.graph.add_edge(nb, i, edge_feat)
        
        # Rewire edges with probability p
        for i in range(n):
            for j in range(1, self.k//2 + 1):
                if random.random() < self.p:
                    nb = (i+j) % n
                    self.graph.remove_edge(i, nb)
                    self.graph.remove_edge(nb, i)
                    
                    # Find new connection
                    possible_targets = [k for k in range(n) 
                                      if k != i and k not in self.graph.adj[i]]
                    if possible_targets:
                        new_nb = random.choice(possible_targets)
                        edge_feat = np.zeros(self.d_relation, dtype=np.float32)
                        edge_feat[0] = 1.0
                        self.graph.add_edge(i, new_nb, edge_feat)
                        self.graph.add_edge(new_nb, i, edge_feat)

    def _generate_symptoms(self) -> np.ndarray:
        symptoms = np.zeros(self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            if self.states[i] == 1 and np.random.rand() < self.traits[i, 2]:
                symptoms[i] = 1.0
        return symptoms

    def _sample_pairs(self) -> list:
        choices = {}
        # Step 1: Each agent chooses a neighbor
        for i in range(self.n_agents):
            neighbors = list(self.graph.adj[i].keys())
            if not neighbors:
                choices[i] = None
                continue
                
            # Extract logits from edge features
            logits = np.array([(self.graph.adj[i][j][0] + self.graph.adj[j][i][0]) / 2 for j in neighbors])
            
            # Convert logits to probabilities using softmax
            max_logit = np.max(logits)
            exp_logits = np.exp(logits - max_logit)
            probs = exp_logits / np.sum(exp_logits)
            
            # Ensure valid probability distribution
            probs = np.nan_to_num(probs, nan=1.0/len(neighbors))
            probs = np.clip(probs, 0, 1)
            probs /= probs.sum()  # Renormalize if needed
            
            choices[i] = np.random.choice(neighbors, p=probs)
        
        # Step 2: Form mutual pairs in random order
        pairs = []
        indices = list(range(self.n_agents))
        random.shuffle(indices)  # Shuffle the indices
        unpaired = set(indices)
        
        for i in indices:
            if i not in unpaired:
                continue
            j = choices[i]
            # Check if j exists, is unpaired, and reciprocally chose i
            if j is not None and j in unpaired:
                pairs.append((i, j))
                unpaired.discard(i)
                unpaired.discard(j)
                
        return pairs

    def _get_observations(self) -> Dict[int, np.ndarray]:
        partner_map = {}
        for i, j in self.current_pairs:
            partner_map[i] = j
            partner_map[j] = i
            
        obs_dict = {}
        for i in range(self.n_agents):
            obs = np.zeros(self.obs_size, dtype=np.float32)
            # Own state and symptom
            obs[0] = self.states[i]
            obs[1] = self.symptoms[i]
            
            # Partner info if exists
            if i in partner_map:
                j = partner_map[i]
                obs[2] = self.states[j]
                obs[3] = self.symptoms[j]
                # Edge features (if edge exists)
                if j in self.graph.adj[i]:
                    edge_feat = self.graph.adj[i][j]
                    obs[4:4+self.d_relation] = edge_feat
            obs_dict[i] = obs
        return obs_dict

    def step(self, actions: Dict[int, float]) -> Tuple:
        rewards = {i: 0.0 for i in range(self.n_agents)}
        new_infections = set()
        
        # Process each pair
        for i, j in self.current_pairs:
            a_i, a_j = actions[i], actions[j]
            duration = min(a_i, a_j)
            
            # Interaction rewards
            rewards[i] += self.traits[i, 0] * duration
            rewards[j] += self.traits[j, 0] * duration
            
            # Disease transmission
            if self.states[i] == 1 and self.states[j] == 0:
                if np.random.rand() < 1 - np.exp(-self.beta * duration):
                    new_infections.add(j)
            if self.states[j] == 1 and self.states[i] == 0:
                if np.random.rand() < 1 - np.exp(-self.beta * duration):
                    new_infections.add(i)
        
        # Infection penalties
        for i in range(self.n_agents):
            if self.states[i] == 1:
                rewards[i] -= self.traits[i, 1]
        
        # Update infection states
        for i in new_infections:
            self.states[i] = 1.0
        
        # Prepare for next step
        self.total_steps += 1
        done = self.total_steps >= self.episode_length
        
        if not done:
            self.symptoms = self._generate_symptoms()
            self.current_pairs = self._sample_pairs()
        next_obs = self._get_observations()
        
        dones = {i: done for i in range(self.n_agents)}
        infos = {i: {} for i in range(self.n_agents)}
        return next_obs, rewards, dones, infos

    def get_agents(self) -> list:
        return self.agents