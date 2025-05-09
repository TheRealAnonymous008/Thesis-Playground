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
    critic_learning_rate : float = 1e-3

    gamma: float = 0.99  # Discount factor
    experience_buffer_size : int = 3         # Warning: You shouldn't make this too big because you will have many agents in the env.
    actor_performance_weight : int = 1.0
    entropy_coeff: float = 0.1
    grad_clip_norm = 0.5

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
    hypernet_learning_rate : float = 1e-3
    hypernet_entropy_weight : float = 0.1
    hypernet_performance_weight : float = 1.0
    hypernet_jsd_threshold: float = 0.5  
    hypernet_jsd_weight : float = 0.2
    hypernet_num_pair_samples : int = 5000

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

def add_exploration_noise(logits: torch.Tensor, params: TrainingParameters, epoch: int = 0):
    """Add independent exploration noise per agent and timestep"""
    device = logits.device
    buffer_size, n_agents, n_actions = logits.shape
    epsilon = max(params.epsilon_end, params.epsilon_start * (params.epsilon_decay ** epoch))
    
    # Create exploration mask [buffer, agents]
    exploration_mask = torch.rand((buffer_size, n_agents), device=device) < epsilon
    
    # Create uniform logits for entire batch [buffer, agents, actions]
    uniform_logits = torch.log(torch.ones_like(logits) / n_actions)
    
    # Apply epsilon-greedy mask
    modified_logits = torch.where(
        exploration_mask.unsqueeze(-1),  # Expand to [buffer, agents, 1]
        uniform_logits,
        logits
    )
    
    # Add independent Gaussian noise per agent-timestep pair
    noise = torch.randn_like(modified_logits) * params.noise_scale
    return modified_logits + noise

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
    batch_ld_means = []
    batch_ld_std = []

    done = False

    for i in range(params.experience_buffer_size):
        
        obs_array = np.stack([obs[agent] for agent in env.get_agents()])
        obs_tensor = torch.FloatTensor(obs_array).to(device)
        
        # Hypernet forward
        belief_vector = torch.zeros((model.config.n_agents, 1), device=device)
        trait_vector = torch.tensor(env.get_traits(), device = device)
        com_vector = torch.zeros((model.config.n_agents, model.config.d_comm_state), device=device)

        lv, wh, mean, std = model.hypernet.forward(trait_vector, belief_vector)
        
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

        batch_ld_means.append(mean)
        batch_ld_std.append(std)
        
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

    batch_means = torch.stack(batch_ld_means)
    batch_std = torch.stack(batch_ld_std)

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
        "means": batch_means,
        "std": batch_std,
    }


def train_model(model: Model, env: BaseEnv, params: TrainingParameters):
    hyper_optim = torch.optim.Adam(model.hypernet.parameters(), lr=params.hypernet_learning_rate)
    actor_optim = torch.optim.Adam(model.actor_encoder.parameters(), lr=params.actor_learning_rate)
    critic_optim = torch.optim.Adam(model.actor_encoder_critic.parameters(), lr = params.critic_learning_rate)

    for i in range(params.outer_loops):
        print(f"Epoch {i}")
        model.requires_grad_(False)

        if params.hypernet_training_loops > 0: 
            train_hypernet(model, env, params, hyper_optim)
            evaluate_policy(model, env)
        if params.actor_training_loops > 0:
            train_actor(model, env, params, actor_optim, critic_optim)
            evaluate_policy(model, env)

        # TODO: Add filter and decoder training


def compute_core_ppo_losses(new_logits: torch.Tensor, new_values: torch.Tensor, 
                            actions: torch.Tensor, advantages: torch.Tensor,
                            returns: torch.Tensor, old_log_probs: torch.Tensor,
                            params: TrainingParameters) -> tuple:
    """Compute PPO policy loss, value loss, and entropy with per-agent calculations."""
    new_dists = Categorical(logits=new_logits)
    new_log_probs = new_dists.log_prob(actions)

    
    # Per-agent entropy (shape [timesteps, agents])
    entropy = new_dists.entropy()
    
    # Independent policy optimization per agent
    ratios = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1-params.clip_epsilon, 1+params.clip_epsilon) * advantages
    

    # Agent-wise policy loss (mean over time first)
    policy_loss = -torch.min(surr1, surr2).mean(dim=0)  # [agents]

    # Agent-wise value loss
    value_loss = torch.nn.functional.huber_loss(new_values.mean(dim=0), returns.mean(dim=0), reduction='none', delta = 1.0) # [agents]
    # Apply per-agent entropy bonus
    entropy_loss = entropy.mean(dim=0)  # [agents]
    
    # Return unaggregated agent-wise losses
    return policy_loss, value_loss, entropy_loss

def train_actor(model: Model, env: BaseEnv, params: TrainingParameters, actor_optim, critic_optim):
    model.actor_encoder.requires_grad_(True)
    model.actor_encoder_critic.requires_grad_(True)

    avg_entropy_loss = 0
    avg_policy_loss = 0
    avg_value_loss = 0

    for epoch in tqdm(range(params.actor_training_loops), desc="Actor Training"):
        exp = collect_experiences(model, env, params)
        logits = add_exploration_noise(exp["logits"], params, epoch)
        
        with torch.no_grad():
            old_dists = Categorical(logits=logits)
            old_log_probs = old_dists.log_prob(exp['actions'])
            returns = exp['returns']
            advantages = returns - exp['values']
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        # PPO training loop
        for _ in range(params.ppo_epochs):
            # Regenerate outputs with current parameters
            new_logits, new_values = [], []
            for i in range(params.experience_buffer_size):
                obs_i = exp["observations"][i]
                wh = exp['wh'][i]
                
                # Actor forward
                Q_i, _, _ = model.actor_encoder(
                    obs_i,
                    exp["belief"][i],
                    exp["com"][i],
                    wh["policy"],
                    wh["belief"],
                    wh["encoder"]
                )
                new_logits.append(Q_i)
                
                # Critic forward
                V_i = model.actor_encoder_critic(
                    obs_i,
                    exp["belief"][i],
                    exp["com"][i],
                    wh["critic"]
                )
                new_values.append(V_i.squeeze(-1))

            new_logits = torch.stack(new_logits)
            new_values = torch.stack(new_values)

            # Calculate losses
            policy_loss, value_loss, entropy = compute_core_ppo_losses(
                new_logits, new_values, exp['actions'], advantages,
                returns, old_log_probs, params
            )

            # Actor update
            actor_loss = (params.actor_performance_weight * policy_loss - params.entropy_coeff * entropy).mean()
            
            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor_encoder.parameters(), params.grad_clip_norm)
            actor_optim.step()

            # Critic update
            critic_loss = params.value_loss_coeff * value_loss.mean()
            
            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor_encoder_critic.parameters(), params.grad_clip_norm)
            critic_optim.step()

            # Track metrics
            avg_policy_loss += policy_loss.mean().item()
            avg_entropy_loss += entropy.mean().item()
            avg_value_loss += value_loss.mean().item()

    print(f"""
    Average Policy Loss: {avg_policy_loss / (params.actor_training_loops * params.ppo_epochs)}
    Average Value Loss: {avg_value_loss / (params.actor_training_loops * params.ppo_epochs)}
    Average Entropy Loss: {avg_entropy_loss / (params.actor_training_loops * params.ppo_epochs)}
    """)

    model.actor_encoder.requires_grad_(False)
    model.actor_encoder_critic.requires_grad_(False)

def train_hypernet(model: Model, env: BaseEnv, params: TrainingParameters, optim):
    model.hypernet.requires_grad_(True)
    
    avg_entropy_loss = 0
    avg_policy_loss = 0
    avg_jsd_loss = 0
    # TODO: Possibly modify this so that the actor network is trained a few times to see if it is actually good.
    for _ in tqdm(range(params.hypernet_training_loops), desc="Hypernet Loop"):        
        exp = collect_experiences(model, env, params)
        with torch.no_grad():
            logits = add_exploration_noise(exp['logits'], params)
            old_dists = Categorical(logits=logits)
            old_log_probs = old_dists.log_prob(exp['actions'])
            old_values = exp['values']
            returns = exp['returns']
            advantages = returns - old_values
            advantages_mean = advantages.mean(dim=0, keepdim=True)
            advantages_std = advantages.std(dim=0, keepdim=True) + 1e-8
            advantages = (advantages - advantages_mean) / advantages_std

        actions = exp['actions']
        observations = exp['observations']
        beliefs = exp['belief']
        traits = exp['traits']
        coms = exp['com']

        new_logits = []
        new_values = []
        lv_list = []
        
        for i in range(params.experience_buffer_size):
            traits_i = traits[i]
            belief_i = beliefs[i]
            lv_i, wh_i, _, _= model.hypernet.forward(traits_i, belief_i)
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
        
        policy_loss, value_loss, entropy = compute_core_ppo_losses(
            new_logits, new_values, exp['actions'], advantages,
            exp['returns'], old_log_probs, params
        )

        num_agents = policy_loss.shape[0]
        agent_weights = torch.softmax(torch.randn(num_agents, device=model.device), dim=-1)
        performance_loss = (
            (agent_weights * (policy_loss - params.entropy_coeff * entropy)).sum() +
            params.value_loss_coeff * (agent_weights * value_loss).sum()
        )
        
        entropy_loss_val = entropy_loss(exp["std"])

        # JSD loss computation with sampled agent pairs within the same timestep
        num_pairs = min(params.hypernet_num_pair_samples, params.experience_buffer_size * (model.config.n_agents * (model.config.n_agents - 1)))

        # Generate timestep indices and agent pairs ensuring i != j
        timesteps = torch.randint(0, params.experience_buffer_size, (num_pairs,), device=model.device)
        agent_i = torch.randint(0, model.config.n_agents, (num_pairs,), device=model.device)
        agent_j = torch.randint(0, model.config.n_agents - 1, (num_pairs,), device=model.device)
        agent_j[agent_j >= agent_i] += 1  # Ensure j != i

        # Get trait vectors for each agent pair
        traits_p = exp["traits"][timesteps, agent_i]
        traits_q = exp["traits"][timesteps, agent_j]

        # Compute cosine similarity between trait vectors
        similarities = torch.nn.functional.cosine_similarity(traits_p, traits_q, dim=1)
        # Get logits for each agent in the pairs
        logits_p = new_logits[timesteps, agent_i]
        logits_q = new_logits[timesteps, agent_j]
        # Compute JSD loss
        jsd_loss = threshed_jsd_loss(logits_p, logits_q, similarities, params.hypernet_jsd_threshold)
        avg_policy_loss += policy_loss 
        avg_entropy_loss += entropy_loss_val
        avg_jsd_loss += jsd_loss

        
        total_loss = params.hypernet_performance_weight * performance_loss 
        + params.hypernet_entropy_weight * entropy_loss_val 
        - params.hypernet_jsd_weight * jsd_loss
        
        optim.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.hypernet.parameters(), max_norm=params.grad_clip_norm)
        total_loss.backward()
        optim.step()
    
    print(f"""
    Average Policy Loss {avg_policy_loss.mean() / params.hypernet_training_loops}
    Average Entropy Loss: {avg_entropy_loss / params.hypernet_training_loops}
    Average JSD Loss: {avg_jsd_loss / params.hypernet_training_loops}
    """)
    model.hypernet.requires_grad_(False)