import numpy as np
import networkx as nx
from models.base_env import *

class InfluencerEnv(BaseEnv):
    def __init__(self, n_agents, num_influencers, d_idea=5, episode_length=10, m=2, perturbation_step=0.1, learning_rate=0.1):
        """
        Environment for idea propagation in a scale-free social network.
        
        Args:
            n_agents (int): Total number of agents
            num_influencers (int): Number of influencer agents
            d_idea (int): Dimensionality of idea vectors
            episode_length (int): Total steps per episode
            m (int): Attachment parameter for scale-free graph
            perturbation_step (float): Magnitude of idea vector perturbation
            learning_rate (float): Update rate for idea adoption
        """
        # Action space: 0 (remove) or 1 (add) for non-influencers, 0/1 for influencer perturbation
        super().__init__(n_agents, d_actions=2, d_relation=4)
        
        self.n_influencers = num_influencers
        self.d_idea = d_idea
        self.episode_length = episode_length
        self.m = m
        self.perturbation_step = perturbation_step
        self.learning_rate = learning_rate
        
        # Observation: idea vector + susceptibility/belief
        self.obs_size = d_idea + 1
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_size,),
            dtype=np.float32
        )
        
        self.agents = list(range(n_agents))
        self.total_steps = 0
        self.influencer_ids = []
        self.non_influencer_ids = []
        self.true_idea = None
        self.broadcast_idea = None
        self.susceptibility = None
        self.true_alpha = None
        self.belief_alpha = None

    def reset(self):
        """Initialize environment with scale-free graph and agent states"""
        super().reset()
        self.total_steps = 0
        
        # Assign influencer status randomly
        self.influencer_ids = np.random.choice(self.agents, self.n_influencers, replace=False)
        self.non_influencer_ids = [a for a in self.agents if a not in self.influencer_ids]
        
        # Build scale-free network
        self._build_scale_free_graph()
        
        # Initialize agent states
        self.true_idea = np.zeros((self.n_agents, self.d_idea), dtype=np.float32)
        self.broadcast_idea = np.zeros((self.n_agents, self.d_idea), dtype=np.float32)
        self.susceptibility = np.zeros(self.n_agents, dtype=np.float32)
        self.true_alpha = np.zeros(self.n_agents, dtype=np.float32)
        self.belief_alpha = np.zeros(self.n_agents, dtype=np.float32)
        
        # Initialize influencers (extreme positions ±1)
        for i in self.influencer_ids:
            self.true_idea[i] = np.random.choice([-1.0, 1.0], size=self.d_idea)
            self.broadcast_idea[i] = self.true_idea[i].copy()
            self.belief_alpha[i] = 0.5  # Initial belief
        
        # Initialize non-influencers (moderate positions)
        for j in self.non_influencer_ids:
            self.true_idea[j] = np.random.uniform(-0.5, 0.5, size=self.d_idea)
            self.susceptibility[j] = np.random.uniform(0, 1)
        
        # Compute influence scores (degree-normalized)
        degrees = np.array([len(self.graph.get_neighbors(a)) for a in self.agents])
        max_deg = np.max(degrees) if len(degrees) > 0 else 1
        for i in self.influencer_ids:
            self.true_alpha[i] = degrees[i] / max_deg
        
        return self._get_observations()

    def _build_scale_free_graph(self):
        """Construct scale-free graph using Barabási-Albert model"""
        # Create BA graph
        ba_graph = nx.barabasi_albert_graph(self.n_agents, self.m)
        
        # Add edges to our graph structure
        for u, v in ba_graph.edges():
            self.graph.add_edge(u, v, np.zeros((self.d_relation,)))
            self.graph.add_edge(v, u, np.zeros((self.d_relation,)))

    def _cosine_similarity(self, u, v):
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(u, v)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0 or norm_v == 0:
            return 0
        return dot_product / (norm_u * norm_v)

    def _sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def step(self, actions):
        """
        Execute one environment step:
        1. Influencers perturb and broadcast ideas
        2. Non-influencers update ideas based on influence
        3. Non-influencers rewire connections
        4. Compute rewards
        """
        # Phase 1: Influencers perturb their broadcasted ideas
        for i in self.influencer_ids:
            action = actions[i]
            dim = np.random.randint(0, self.d_idea)
            delta = (2 * action - 1) * self.perturbation_step
            self.broadcast_idea[i] = self.true_idea[i].copy()
            self.broadcast_idea[i, dim] = np.clip( self.broadcast_idea[i, dim] + delta, -1, 1)

        # Phase 2: Idea propagation and adoption
        for j in self.non_influencer_ids:
            neighbors = self.graph.get_neighbors(j)
            influencer_neighbors = [n for n in neighbors if n in self.influencer_ids]
            
            if influencer_neighbors:
                # Find most influential neighbor
                best_score = -10
                best_influencer = None
                for i in influencer_neighbors:
                    similarity = self._cosine_similarity(self.broadcast_idea[i],  self.true_idea[j])
                    score = self.true_alpha[i] * similarity
                    if score > best_score:
                        best_score = score
                        best_influencer = i
                
                # Adoption probability
                adoption_prob = self._sigmoid(best_score - self.susceptibility[j])
                if np.random.rand() < adoption_prob:
                    # Update idea vector
                    self.true_idea[j] = (1 - self.learning_rate) * self.true_idea[j] + self.learning_rate * self.broadcast_idea[best_influencer]
                    self.true_idea[j] = np.clip(self.true_idea[j], -1, 1)

        # Phase 3: Network rewiring by non-influencers
        random.shuffle(self.non_influencer_ids)
        for i in range(0, len(self.non_influencer_ids) - 1, 2):
            a1 = self.non_influencer_ids[i]
            a2 = self.non_influencer_ids[i+1]
            
            action1 = actions[a1]
            action2 = actions[a2]
            edge_exists = self.graph.has_edge(a1, a2)
            
            if edge_exists and (action1 == 0 or action2 == 0):
                # Remove connection
                self.graph.remove_edge(a1, a2)
                self.graph.remove_edge(a2, a1)
            elif not edge_exists and action1 == 1 and action2 == 1:
                # Create connection
                self.graph.add_edge(a1, a2, np.zeros((self.d_relation,)))
                self.graph.add_edge(a2, a1, np.zeros((self.d_relation,)))
        
        # Update influence scores after rewiring
        degrees = np.array([len(self.graph.get_neighbors(a)) for a in self.agents])
        max_deg = np.max(degrees) if len(degrees) > 0 else 1
        for i in self.influencer_ids:
            self.true_alpha[i] = degrees[i] / max_deg

        # Phase 4: Compute rewards
        rewards = {agent: 0.0 for agent in self.agents}
        mean_vector = np.mean(self.true_idea, axis=0)
        
        # Influencer reward: negative L2 distance from population mean
        for i in self.influencer_ids:
            dist = np.linalg.norm(self.true_idea[i] - mean_vector)
            rewards[i] = -dist
        
        # Prepare observations
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

    def _get_observations(self):
        """Get observations for all agents"""
        observations = {}
        for agent in self.agents:
            obs = np.zeros(self.obs_size, dtype=np.float32)
            obs[:self.d_idea] = self.true_idea[agent]
            
            if agent in self.influencer_ids:
                obs[self.d_idea] = self.belief_alpha[agent]
            else:
                obs[self.d_idea] = self.susceptibility[agent]
            
            observations[agent] = obs
        return observations

    def get_agents(self):
        return self.agents

    def get_traits(self):
        """Return idea vectors as traits"""
        return self.true_idea