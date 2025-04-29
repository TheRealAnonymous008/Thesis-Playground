from .losses import * 
from .model import * 
from .base_env import *
from tests.eval import *

import gymnasium as gym 
import torch 
from torch.distributions import Categorical
from dataclasses import dataclass 
from tqdm import tqdm 

@dataclass
class TrainingParameters:
    outer_loops: int = 5
    hypernet_training_loops: int = 5
    actor_training_loops: int = 5
    actor_learning_rate: float = 1e-3

    gamma: float = 0.99  # Discount factor
    experience_buffer_size : int = 3         # Warning: You shouldn't make this too big because you will have many agents in the env.
    entropy_coeff: float = 0.1

    # Exploration specific
    entropy_target: float = 0.5  # Target entropy for exploration
    noise_scale: float = 0.1     # Scale for parameter noise
    epsilon_start: float = 0.1   # Starting probability for epsilon-greedy
    epsilon_end: float = 0.01    # Ending probability for epsilon-greedy
    epsilon_decay: float = 0.99  # Decay rate for epsilon

    # PPO-specific parameters
    clip_epsilon: float = 0.2
    ppo_epochs: int = 4
    value_loss_coeff: float = 0.5

    # Hypernet specific parameters
    hypernet_learning_weight : float = 1e-3
    hypernet_entropy_weight : float = 0.1
    hypernet_performance_weight : float = 1.0
    hypernet_jsd_threshold: float = 0.5  
    hypernet_jsd_weight : float = 0.2
    hypernet_num_pair_samples : int = 500

def compute_returns(rewards: np.ndarray, gamma: float) -> torch.Tensor:
    """Compute discounted returns for each agent and timestep."""
    n_timesteps, n_agents = rewards.shape
    returns = np.zeros_like(rewards, dtype=np.float32)
    
    # Calculate returns for each agent
    for agent in range(n_agents):
        R = 0.0
        for t in reversed(range(n_timesteps)):
            R = rewards[t, agent] + gamma * R
            returns[t, agent] = R
    
    return torch.tensor(returns, dtype=torch.float32)

def add_exploration_noise(logits: torch.Tensor, params: TrainingParameters, epoch : int = 0):
    """Add multiple exploration strategies to policy outputs."""
    epsilon = max(params.epsilon_end, params.epsilon_start * (params.epsilon_decay ** epoch))
    if epsilon is None:
        epsilon = params.epsilon_start
    
    # Epsilon-greedy exploration
    if np.random.rand() < epsilon:
        return torch.tensor([np.log(1.0 / logits.shape[-1])] * logits.shape[-1], device = logits.device)
    
    # Parameter noise
    noise = torch.normal(0, params.noise_scale, size=logits.shape, device = logits.device)
    return logits + noise 

def collect_experiences(model : Model, env : BaseEnv, params : TrainingParameters):
    device = model.config.device
    obs = env.reset()
    batch_obs = []
    batch_actions = []
    batch_rewards = []
    batch_logits = []
    batch_lv = []
    batch_values = []
    batch_wh = []
    batch_belief = []
    batch_trait = []
    batch_com = []

    done = False

    for i in range(params.experience_buffer_size):
        
        obs_array = np.stack([obs[agent] for agent in env.get_agents()])
        obs_tensor = torch.FloatTensor(obs_array).to(device)
        
        # Hypernet forward
        belief_vector = torch.zeros((model.config.n_agents, 1), device=device)
        trait_vector = torch.tensor(env.get_traits(), device = device)
        com_vector = torch.zeros((model.config.n_agents, model.config.d_comm_state), device=device)

        lv, wh = model.hypernet(trait_vector, belief_vector)
        
        # Actor encoder forward
        Q, _, _ = model.actor_encoder.forward(
            obs_tensor, 
            belief_vector, 
            com_vector, 
            wh["policy"],
            wh["belief"], 
            wh["encoder"]
        )
        dists = Categorical(logits=Q)
        actions = dists.sample().cpu().numpy()
        actions_dict = {agent: int(actions[i]) for i, agent in enumerate(env.get_agents())}

        # Critic forward
        V = model.actor_encoder_critic.forward(
            obs_tensor, 
            belief_vector, 
            com_vector,
            wh["critic"]
        )
        values = V.squeeze(-1)  # Remove singleton dimension if needed
        # Environment step
        next_obs, rewards, dones, _ = env.step(actions_dict)
        rewards = np.array([rewards[agent] for agent in env.get_agents()])

        # Store experience
        batch_obs.append(obs_tensor)
        batch_actions.append(actions)
        batch_rewards.append(rewards)
        batch_logits.append(Q)
        batch_lv.append(lv)
        batch_wh.append(wh)
        batch_values.append(values)

        batch_belief.append(belief_vector)
        batch_trait.append(trait_vector)
        batch_com.append(com_vector)
        
        obs = next_obs
        if any(dones.values()):
            obs = env.reset()
    
    # Convert lists to tensors
    batch_obs = torch.stack(batch_obs)
    batch_actions = torch.tensor(np.array(batch_actions), device=device)
    batch_rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32, device=device)
    batch_logits = torch.stack(batch_logits)
    batch_lv = torch.stack(batch_lv)
    batch_values = torch.stack(batch_values)
    batch_wh = torch.stack(batch_wh)

    batch_belief = torch.stack(batch_belief)
    batch_trait = torch.stack(batch_trait)
    batch_com = torch.stack(batch_com)

    # Compute returns
    returns = compute_returns(batch_rewards.cpu().numpy(), params.gamma).to(device)
    return {
        'observations': batch_obs,
        'actions': batch_actions,
        'rewards': batch_rewards,
        'logits': batch_logits,
        'lv': batch_lv,
        'wh': batch_wh,
        'returns': returns,
        "values" : batch_values,
        'belief': batch_belief,
        'traits': batch_trait, 
        'com': batch_com,
    }


def train_model(model: Model, env: BaseEnv, params: TrainingParameters):
    hyper_optim = torch.optim.Adam(model.hypernet.parameters(), lr=params.actor_learning_rate)
    actor_optim = torch.optim.Adam(model.actor_encoder.parameters(), lr=params.actor_learning_rate)

    for i in range(params.outer_loops):
        print(f"Epoch {i}")
        model.requires_grad_(False)

        train_hypernet(model, env, params, hyper_optim)
        evaluate_policy(model, env)
        train_actor(model, env, params, actor_optim)
        evaluate_policy(model, env)

        # TODO: Add filter and decoder training


def compute_core_ppo_losses(new_logits: torch.Tensor, new_values: torch.Tensor, 
                            actions: torch.Tensor, advantages: torch.Tensor,
                            returns: torch.Tensor, old_log_probs: torch.Tensor,
                            params: TrainingParameters) -> tuple:
    """Compute PPO policy loss, value loss, and entropy."""
    new_dists = Categorical(logits=new_logits)
    new_log_probs = new_dists.log_prob(actions)
    entropy = new_dists.entropy().mean()

    ratios = torch.exp(new_log_probs - old_log_probs)
    surrogate1 = ratios * advantages
    surrogate2 = torch.clamp(ratios, 1 - params.clip_epsilon, 1 + params.clip_epsilon) * advantages
    policy_loss = -torch.min(surrogate1, surrogate2).mean()
    value_loss = torch.nn.functional.mse_loss(new_values, returns)
    
    return policy_loss, value_loss, entropy

def train_actor(model: Model, env: BaseEnv, params: TrainingParameters, optim):
    model.actor_encoder.requires_grad_(True)
    model.actor_encoder_critic.requires_grad_(True)

    for epoch in tqdm(range(params.actor_training_loops), desc = "Actor Training"):
        # Collect experiences once per outer loop
        exp = collect_experiences(model, env, params)
        logits = add_exploration_noise(exp["logits"], params, epoch)
        # Compute old log probabilities and values
        with torch.no_grad():
            old_dists = Categorical(logits=logits)
            old_log_probs = old_dists.log_prob(exp['actions'])
            old_values = exp['values']
            returns = exp['returns']
            advantages = returns - old_values
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actions = exp['actions']

        # PPO training loop
        for _ in range(params.ppo_epochs):
            new_logits = []
            new_values = []

            # Recompute logits and values with current parameters
            for i in range(params.experience_buffer_size):
                obs_i = exp["observations"][i]
                wh = exp['wh'][i]

                # Recompute actor outputs
                Q_i, _, _ = model.actor_encoder(
                    obs_i,
                    exp["belief"][i],
                    exp["com"][i],
                    wh["policy"],
                    wh["belief"],
                    wh["encoder"]
                )
                new_logits.append(Q_i)

                # Recompute critic outputs
                V_i = model.actor_encoder_critic(
                    obs_i,
                    exp["belief"][i],
                    exp["com"][i],
                    wh["critic"]
                )
                new_values.append(V_i.squeeze(-1))

            new_logits = torch.stack(new_logits)
            new_values = torch.stack(new_values)

            # Calculate policy losses
            new_dists = Categorical(logits=new_logits)
            entropy = new_dists.entropy().mean()

            policy_loss, value_loss, entropy = compute_core_ppo_losses(
                new_logits, new_values, exp['actions'], advantages,
                exp['returns'], old_log_probs, params
            )
            
            # Total loss
            total_loss = (policy_loss +
                        params.value_loss_coeff * value_loss -
                        params.entropy_coeff * entropy)

            # Update parameters
            optim.zero_grad()
            
            # Grad Clipping
            torch.nn.utils.clip_grad_norm_(model.actor_encoder.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(model.actor_encoder_critic.parameters(), max_norm=0.5)
            total_loss.backward()
            optim.step()
    

    model.actor_encoder.requires_grad_(False)
    model.actor_encoder_critic.requires_grad_(False)

def train_hypernet(model: Model, env: BaseEnv, params: TrainingParameters, optim):
    model.hypernet.requires_grad_(True)
    
    exp = collect_experiences(model, env, params)
    
    with torch.no_grad():
        logits = add_exploration_noise(exp['logits'], params)
        old_dists = Categorical(logits=logits)
        old_log_probs = old_dists.log_prob(exp['actions'])
        old_values = exp['values']
        returns = exp['returns']
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    actions = exp['actions']
    observations = exp['observations']
    beliefs = exp['belief']
    traits = exp['traits']
    coms = exp['com']
    
    for _ in tqdm(range(params.hypernet_training_loops), desc="Hypernet Loop"):
        for _ in range(params.ppo_epochs):
            new_logits = []
            new_values = []
            lv_list = []
            
            for i in range(params.experience_buffer_size):
                traits_i = traits[i]
                belief_i = beliefs[i]
                lv_i, wh_i = model.hypernet(traits_i, belief_i)
                lv_list.append(lv_i)
                Q_i, _, _ = model.actor_encoder(
                    observations[i],
                    belief_i,
                    coms[i],
                    wh_i["policy"],
                    wh_i["belief"],
                    wh_i["encoder"]
                )
                new_logits.append(Q_i)
                V_i = model.actor_encoder_critic(
                    observations[i],
                    belief_i,
                    coms[i],
                    wh_i["critic"]
                ).squeeze(-1)
                new_values.append(V_i)
            
            new_logits = torch.stack(new_logits)
            new_values = torch.stack(new_values)
            lv_all = torch.stack(lv_list)
            
            policy_loss, value_loss, entropy = compute_core_ppo_losses(
                new_logits, new_values, exp['actions'], advantages,
                exp['returns'], old_log_probs, params
            )
            
            performance_loss = policy_loss + params.value_loss_coeff * value_loss - params.entropy_coeff * entropy
            entropy_loss_val = entropy_loss(lv_all).mean()
            
            # JSD loss computation with sampled pairs
            # Sample pairs of indices
            num_pairs = min(env.n_agents **2, params.hypernet_num_pair_samples)
            indices = torch.randint(0, params.experience_buffer_size, (num_pairs, 2), device=lv_all.device)
            
            # Get latent variables for the sampled pairs
            lv_pairs = lv_all[indices]  # Shape (num_pairs, 2, latent_dim)
            
            # Compute cosine similarity between each pair's latent variables
            similarities = torch.nn.functional.cosine_similarity(lv_pairs[:, 0], lv_pairs[:, 1], dim=-1)
            
            # Get corresponding logits for each pair
            logits_p = new_logits[indices[:, 0]]
            logits_q = new_logits[indices[:, 1]]
            
            # Compute thresholded JSD loss
            jsd_loss = threshed_jsd_loss(logits_p, logits_q, similarities, params.hypernet_jsd_threshold)
            
            total_loss = params.hypernet_learning_weight * (
                params.hypernet_performance_weight * performance_loss +
                params.hypernet_entropy_weight * entropy_loss_val +
                params.hypernet_jsd_weight * jsd_loss
            )
            
            optim.zero_grad()

            # Grad Clipping
            torch.nn.utils.clip_grad_norm_(model.hypernet.parameters(), max_norm=0.5)
            total_loss.backward()
            optim.step()
    
    model.hypernet.requires_grad_(False)