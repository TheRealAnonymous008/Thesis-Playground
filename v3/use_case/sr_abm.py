from models.base_env import *
import networkx as nx

class DiseaseSpreadEnv(BaseEnv):
    def __init__(self, n_agents: int, d_relation: int = 4, 
                 beta_range: Tuple[float, float] = (0.2, 0.8), 
                 m_range: Tuple[int, int] = (2, 5), 
                 episode_length: int = 50):
        """
        Environment for disease spread simulation using SI model with scale-free network.
        
        Args:
            n_agents: Number of agents
            d_relation: Dimension of edge features
            beta_range: Range for disease transmission rate (sampled each reset)
            m_range: Range for scale-free graph parameter (number of edges to attach)
            episode_length: Total steps per episode
        """
        super().__init__(
            n_agents=n_agents,
            d_actions=1,  # Continuous action (interaction duration)
            d_traits=3,    # [alpha, rho, p_s]
            d_beliefs=1,   # Unused (minimal size)
            d_comm_state=1,  # Unused (minimal size)
            d_relation=d_relation,
            obs_size=5 + d_relation  # [own_state, own_symptom, partner_state, partner_symptom, degree] + edge
        )
        self.is_continuous = True
        self.beta_range = beta_range
        self.m_range = m_range
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
        
        # Reinitialize parameters each reset
        self.beta = np.random.uniform(*self.beta_range)
        self.m = np.random.randint(*self.m_range)
        self.p_min = np.random.uniform(0, 0.6)
        self.p_max = np.random.uniform(self.p_min, 1.0)
        
        # Initialize agent states (0=susceptible, 1=infected)
        self.states = np.zeros(self.n_agents, dtype=np.float32)
        initial_infected = np.random.choice(self.n_agents, size=max(1, self.n_agents//20), replace=False)
        self.states[initial_infected] = 1.0
        
        # Initialize agent traits: [alpha, rho, p_s]
        self.traits = np.random.uniform(
            low=[0.1, 1.0, self.p_min], 
            high=[5.0, 10.0, self.p_max],
            size=(self.n_agents, 3)
        ).astype(np.float32)
        
        # Build scale-free social network
        self._build_scale_free_graph()
        # Store degrees for observations
        self.degrees = np.array([len(self.graph.adj[i]) for i in range(self.n_agents)], dtype=np.float32)
        self.degrees /= self.n_agents
        # Generate initial symptoms
        self.symptoms = self._generate_symptoms()
        # Form initial pairs
        self.current_pairs = self._sample_pairs()
        return self._get_observations()

    def _build_scale_free_graph(self):
        """Constructs a scale-free network using Barabási-Albert model"""
        if self.n_agents < 2:
            return  # No edges possible
        
        # Generate undirected scale-free graph
        ba_graph = nx.barabasi_albert_graph(n=self.n_agents, m=self.m)
        
        # Add edges to our directed graph (with symmetric connections)
        for u, v in ba_graph.edges():
            edge_feat = np.zeros(self.d_relation, dtype=np.float32)
            edge_feat[0] = 1.0  # First component as edge weight
            self.graph.add_edge(u, v, edge_feat)
            self.graph.add_edge(v, u, edge_feat)

    def _generate_symptoms(self) -> np.ndarray:
        symptoms = np.zeros(self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            if self.states[i] == 1 and np.random.rand() < self.traits[i, 2]:
                symptoms[i] = 1.0
        return symptoms

    def _sample_pairs(self) -> list:
        # (Unchanged from original)
        choices = {}
        for i in range(self.n_agents):
            neighbors = list(self.graph.adj[i].keys())
            if not neighbors:
                choices[i] = None
                continue
                
            logits = np.array([(self.graph.adj[i][j][0] + self.graph.adj[j][i][0]) / 2 
                     for j in neighbors])
            
            max_logit = np.max(logits)
            exp_logits = np.exp(logits - max_logit)
            probs = exp_logits / np.sum(exp_logits)
            probs = np.nan_to_num(probs, nan=1.0/len(neighbors))
            probs = np.clip(probs, 0, 1)
            probs /= probs.sum()
            
            choices[i] = np.random.choice(neighbors, p=probs)
        
        pairs = []
        indices = list(range(self.n_agents))
        random.shuffle(indices)
        unpaired = set(indices)
        
        for i in indices:
            if i not in unpaired:
                continue
            j = choices[i]
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
                # Edge features
                if j in self.graph.adj[i]:
                    edge_feat = self.graph.adj[i][j]
                    obs[5:5+self.d_relation] = edge_feat
            
            # Add degree (always present)
            obs[4] = self.degrees[i] 
            
            obs_dict[i] = obs
        return obs_dict

    def step(self, actions: Dict[int, float]) -> Tuple:
        # (Unchanged from original)
        rewards = {i: 0.0 for i in range(self.n_agents)}
        new_infections = set()
        
        for i, j in self.current_pairs:
            a_i = actions[i]
            a_j = actions[j]
            duration = min(a_i, a_j)
            
            rewards[i] += self.traits[i][0] * duration
            rewards[j] += self.traits[j][0] * duration
            
            if self.states[i] == 1 and self.states[j] == 0:
                if np.random.rand() < 1 - np.exp(-self.beta * duration):
                    new_infections.add(j)
            if self.states[j] == 1 and self.states[i] == 0:
                if np.random.rand() < 1 - np.exp(-self.beta * duration):
                    new_infections.add(i)
        
        for i in range(self.n_agents):
            if self.states[i] == 1:
                rewards[i] -= self.traits[i][1]
        
        for i in new_infections:
            self.states[i] = 1.0
        
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
    
    def postprocess_actions(self, actions : torch.Tensor):
        actions = actions.nan_to_num(0, 0, 0)
        actions =  torch.clamp(actions.squeeze(), 1e-5, 10).cpu().detach().numpy().astype(np.float16)
        return actions