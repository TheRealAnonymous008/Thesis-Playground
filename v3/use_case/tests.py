from .baseline import * 
from .baseline_het import * 
from .baseline_comm import *
from .baseline_net import *
from .sr_abm import *
from .influencer_abm import * 

def initialize_baseline(seed = 1337, n_agents = 1000, N = 10):
    np.random.seed(seed)
    payoff_i = np.random.randint(-10, 10, (N, N))
    payoff_j = np.random.randint(-10, 10, (N, N))

    # Initialize environment
    return BaselineEnvironment(n_agents, payoff_i, payoff_j, total_games = 5)

def initialize_baseline_hetero(seed = 1337, n_agents = 1000, n_types = 10, n_actions= 10):
    np.random.seed(seed)
    type_payoffs = np.random.uniform(-10, 10, (n_types, n_types, 2, n_actions, n_actions))
    return BaselineHeterogeneous(n_agents, n_types, type_payoffs, total_games = 4)

def initialize_type_inference(seed = 1337, n_agents = 1000, n_types = 10):
    np.random.seed(seed)
    return TypeInferenceEnvironment(n_agents, n_types, total_games=5)

def initialize_network_env(seed = 1337, n_agents = 1000, n_types = 2):
    np.random.seed(seed)
    return NetBasedEnvironment(n_agents, n_types, episode_length=10)

def initialize_sir_env(seed = 1337, n_agents = 1000, eps_length= 20):
    np.random.seed(seed)
    return DiseaseSpreadEnv(n_agents, episode_length=eps_length)

def initialize_influencer_env(seed = 1337, n_agents = 1000, eps_length = 20):
    np.random.seed(seed)

    influencers = int(0.01 * n_agents)
    return InfluencerEnv(n_agents = n_agents, num_influencers = influencers, episode_length= eps_length, m = 5)