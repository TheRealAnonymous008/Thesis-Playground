from .losses import * 
from .model import * 
from .base_env import *
from tests.eval import *

import gymnasium as gym 
import torch 
from tensordict import cat
from torch.distributions import Categorical
from dataclasses import dataclass 
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

# Temporary fix to avoid OMP duplicates. Not ideal though.
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

@dataclass
class TrainingParameters:
    outer_loops: int = 5
    actor_learning_rate: float = 1e-3
    critic_learning_rate : float = 1e-3

    gamma: float = 0.99  # Discount factor
    experience_buffer_size : int = 3         # Warning: You shouldn't make this too big because you will have many agents in the env
    actor_performance_weight : float = 1.0
    experience_sampling_steps : int = 100
    grad_clip_norm = 10.0

    # Exploration specific
    entropy_target: float = 0.1  # Target entropy for exploration. Used in the hypernet.
    noise_scale: float = 0.1     # Scale for parameter noise
    epsilon_start: float = 0.8   # Starting probability for epsilon-greedy
    epsilon_end: float = 0.2    # Ending probability for epsilon-greedy
    epsilon_decay: float = 0.99  # Decay rate for epsilon
    epsilon_period: int = 250    # period for cosine scheduling. If 0, doesn't use cosine scheduling
    
    
    # PPO-specific parameters
    clip_epsilon: float = 0.2
    ppo_epochs: int = 15
    value_loss_coeff: float = 1.0
    entropy_coeff: float = 0.2

    # Hypernet specific parameters
    hypernet_learning_rate : float = 1e-3
    hypernet_entropy_weight : float = 0.1
    hypernet_jsd_threshold: float = 0.5  
    hypernet_jsd_weight : float = 0.2
    hypernet_num_pair_samples : int = 5000
    hypernet_steps : int = 15

    sampled_agents_proportion : float = 1.0

    filter_learning_rate : float = 1e-3
    decoder_learning_rate : float = 1e-3

    # Control training flow here
    should_train_hypernet : bool = True,
    should_train_actor : bool = True 
    verbose : bool = True

    # Do not change this
    global_steps : int = 0
    epsilon : float = 0 

def normalize_tensor(rewards : torch.Tensor):
    reward_mean = rewards.mean(dim=1, keepdim=True)
    reward_std = rewards.std(dim=1, keepdim=True) + 1e-8
    rewards = (rewards - reward_mean) / reward_std
    return rewards

def compute_returns(rewards: np.ndarray, dones : torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute discounted returns for each agent and timestep."""
    n_timesteps, n_agents = rewards.shape
    returns = np.zeros_like(rewards, dtype=np.float32)
    
    # Calculate returns for each agent
    for agent in range(n_agents):
        for t in reversed(range(n_timesteps)):
            if dones[t]:
                R = 0.0
            R = rewards[t, agent] + gamma * R          # Augment rewards to incentivize maximization
            returns[t, agent] = R
    
    return torch.tensor(returns, dtype=torch.float32)

def add_exploration_noise(logits: torch.Tensor, params: TrainingParameters, epoch: int = 0):
    """Add independent exploration noise per agent and timestep"""
    device = logits.device
    n_agents, n_actions = logits.shape

    if params.epsilon_period > 1: 
        epoch = epoch % params.epsilon_period
        params.epsilon = max(params.epsilon_end, params.epsilon_start * (params.epsilon_decay ** epoch))
    
    # Create exploration mask [buffer, agents]
    exploration_mask = torch.rand((n_agents), device=device) < params.epsilon
    
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

def select_weights(wh : TensorDict, indices : list) -> TensorDict:
    return TensorDict(
    {
        key: TensorDict(
            {
                "weight": wh[key]["weight"][indices],
                "bias": wh[key]["bias"][indices]
            }, 
            batch_size=[len(indices)]
        )
        for key in wh.keys()
    }, 
    batch_size=[len(indices)],
    device=wh.device
    )

def collect_experiences(model : Model, env : BaseEnv, params : TrainingParameters, epoch = 1):
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
    batch_dones = []

    sampled_agents = int(params.sampled_agents_proportion * env.n_agents)
    indices = np.random.choice(env.n_agents, size = sampled_agents, replace = False)
    for i in range(params.experience_sampling_steps):
        
        obs_array = np.stack([obs[agent] for agent in env.get_agents()])
        obs_tensor = torch.FloatTensor(obs_array).to(device)
        
        # Hypernet forward
        belief_vector = torch.tensor(env.get_beliefs(), device = device)
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
        Q = add_exploration_noise(Q, params, epoch)

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

        obs = next_obs
        done = np.any(dones.values()) or i == params.experience_sampling_steps - 1

        if done:
            obs = env.reset()

        # Store experience
        batch_obs.append(obs_tensor[indices])
        batch_actions.append(actions[indices])
        batch_rewards.append(rewards[indices])
        batch_logits.append(Q[indices])
        batch_lv.append(lv[indices])
        batch_wh.append(select_weights(wh, indices))
        batch_values.append(values[indices])

        batch_belief.append(belief_vector[indices])
        batch_trait.append(trait_vector[indices])
        batch_com.append(com_vector[indices])

        batch_ld_means.append(mean[indices])
        batch_ld_std.append(std[indices])
        batch_dones.append(torch.tensor(done, dtype = torch.bool))
        
    
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
    batch_dones = torch.stack(batch_dones)

    # Compute returns

    return {
        'observations': batch_obs,
        'actions': batch_actions,
        'rewards': batch_rewards,
        'logits': batch_logits,
        'lv': batch_lv,
        'wh': batch_wh,
        "values" : batch_values,
        'belief': batch_belief,
        'traits': batch_trait, 
        'com': batch_com,
        "means": batch_means,
        "std": batch_std,
        "done": batch_dones
    }

def compute_core_ppo_losses(new_logits: torch.Tensor, 
                            new_values: torch.Tensor, 
                            actions: torch.Tensor, 
                            advantages: torch.Tensor,
                            returns: torch.Tensor, 
                            old_log_probs: torch.Tensor,
                            params: TrainingParameters
                            ) -> tuple:
    """Compute PPO policy loss, value loss, and entropy with per-agent calculations."""
    new_dists = Categorical(logits=new_logits)
    new_log_probs = new_dists.log_prob(actions)
    
    
    # Independent policy optimization per agent
    ratios = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1-params.clip_epsilon, 1+params.clip_epsilon) * advantages

    # Agent-wise policy loss (mean over time first)
    policy_loss = -torch.min(surr1, surr2).mean(dim=0)  # [agents]

    # Agent-wise value loss
    value_loss = torch.nn.functional.huber_loss(new_values.mean(dim=0), returns.mean(dim=0), reduction='none', delta = 10.0) # [agents]

    entropy = new_dists.entropy()
    entropy_loss = entropy.mean(dim=0)  # [agents]
    
    # Return unaggregated agent-wise losses
    return policy_loss, value_loss, entropy_loss

def train_model(model: Model, env: BaseEnv, params: TrainingParameters):
    if params.verbose:
        writer = SummaryWriter()
    else:
        writer = None 

    optim = torch.optim.Adam([
        {'params': model.actor_encoder.parameters(), 'lr': params.actor_learning_rate, 'eps' : 1e-5},
        {'params': model.actor_encoder_critic.parameters(), 'lr': params.critic_learning_rate, 'eps' : 1e-5},
        {'params': model.hypernet.parameters(), 'lr': params.hypernet_learning_rate, 'eps' : 1e-5},
        {'params': model.filter.parameters(), 'lr': params.filter_learning_rate, 'eps' : 1e-5}, 
        {'params': model.decoder_update.parameters(), 'lr': params.decoder_learning_rate, 'eps' : 1e-5}
    ])  

    params.global_steps = 0
    experiences = TensorDict({})  # Initialize empty experience buffer
    for i in tqdm(range(params.outer_loops)):
        model.requires_grad_(True)
        
        # Collect new experiences and explicitly detach+clone
        new_exp = collect_experiences(model, env, params, i)
        
        # Create detached clone of all tensors in the experience
        detached_exp = TensorDict({
            k: v.detach().clone() for k, v in new_exp.items()
        }, batch_size=[params.experience_sampling_steps])
        
        # Append to buffer
        if len(experiences) == 0:
            experiences = detached_exp
        else:
            experiences = torch.cat([experiences, detached_exp], dim=0)
        
        # Trim buffer while maintaining computational graph isolation
        if len(experiences) > params.experience_buffer_size:
            keep_from = len(experiences) - params.experience_buffer_size
            experiences = TensorDict(
                {k: v[keep_from:] for k, v in experiences.items()},
                batch_size=[params.experience_buffer_size]
            )
        
        total_loss = 0
        if params.should_train_actor:
            total_loss += train_actor(model, env, experiences, params, writer=writer)
        if params.should_train_hypernet:
            total_loss += train_hypernet(model, env, experiences, params, writer=writer)

        
        if writer is not None:
            writer.add_scalar('State/Epsilon', params.epsilon, global_step = params.global_steps)

        optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip_norm)
        optim.step()
        
        model.requires_grad_(False)
        evaluate_policy(model, env, writer=writer, global_step=params.global_steps)
        params.global_steps += 1

    writer.close()

def train_actor(model: Model, env: BaseEnv, exp: TensorDict, params: TrainingParameters, writer : SummaryWriter =None):
    old_logits = exp["logits"]
    old_dists = Categorical(logits=old_logits)
    old_log_probs = old_dists.log_prob(exp['actions'])

    rewards = normalize_tensor(exp["rewards"])
    returns = compute_returns(rewards.detach().cpu().numpy(), exp["done"], params.gamma).to(exp["rewards"].device)
    values = normalize_tensor(exp["values"])

    advantages = normalize_tensor(returns - values)

    # Reshape all inputs to combine buffer_size and agent_no into a single batch dimension
    buffer_size, agent_no, *_ = exp["traits"].shape  # Get dimensions
    traits_all = exp["traits"].view(-1, exp["traits"].shape[-1])
    belief_hyper = exp["belief"].view(-1, exp["belief"].shape[-1])

    # Compute hypernet outputs for all agents and buffer entries in one batch
    _, wh_all, _, _ = model.hypernet.forward(traits_all, belief_hyper)

    # Reshape other inputs for actor and critic
    obs_all = exp["observations"].view(-1, exp["observations"].shape[-1])
    belief_actor = exp["belief"].view(-1, exp["belief"].shape[-1])
    com_all = exp["com"].view(-1, exp["com"].shape[-1])

    # Actor forward pass for all entries
    Q_all, _, _ = model.actor_encoder(
        obs_all,
        belief_actor,
        com_all,
        wh_all["policy"],
        wh_all["belief"],
        wh_all["encoder"]
    )

    # Reshape Q outputs to (buffer_size, agent_no, ...)
    new_logits = Q_all.view(buffer_size, agent_no, -1)  # Adjust dimensions as needed

    # Critic forward pass
    V_all = model.actor_encoder_critic(
        obs_all,
        belief_actor,
        com_all,
        wh_all["critic"]
    ).squeeze(-1)

    # Reshape V outputs to (buffer_size, agent_no)
    new_values = V_all.view(buffer_size, agent_no)
    # Calculate losses
    policy_loss, value_loss, entropy = compute_core_ppo_losses(
        new_logits, new_values, exp['actions'], advantages, returns, old_log_probs, params
    )

    policy_loss =  params.actor_performance_weight * policy_loss.mean()
    value_loss = params.value_loss_coeff * value_loss.mean()
    
    entropy = entropy.mean()
    entropy_regularization = params.entropy_coeff * entropy

    actor_loss = policy_loss - entropy_regularization
    critic_loss = value_loss
    total_loss = actor_loss + critic_loss

    if writer is not None:
        writer.add_scalars('Actor/Metrics', {
            'Policy Loss' : policy_loss.item(), 
            'Actor Loss': actor_loss.item(),
            'Critic Loss': critic_loss.item(),
            'Entropy': entropy_regularization.item(),
            'Total Loss': total_loss.item()
        }, global_step=params.global_steps)

    return total_loss

def train_hypernet(model: Model, env: BaseEnv, exp: TensorDict, params: TrainingParameters, writer : SummaryWriter =None):
    num_agents = int(env.n_agents * params.sampled_agents_proportion)
    entropy_loss_val = entropy_loss(exp["means"], exp["std"], params.entropy_target)

    old_logits = exp["logits"]

    # JSD loss computation with sampled agent pairs within the same timestep
    buffer_length = len(exp)
    num_pairs = min(params.hypernet_num_pair_samples, buffer_length * (num_agents * (num_agents - 1)))

    # Generate timestep indices and agent pairs ensuring i != j
    timesteps = torch.randint(0, buffer_length, (num_pairs,), device=model.device)
    agent_i = torch.randint(0, num_agents, (num_pairs,), device=model.device)
    agent_j = torch.randint(0, num_agents - 1, (num_pairs,), device=model.device)
    agent_j[agent_j >= agent_i] += 1  # Ensure j != i

    # Get trait vectors for each agent pair
    traits_p = exp["traits"][timesteps, agent_i]
    traits_q = exp["traits"][timesteps, agent_j]

    # Compute cosine similarity between trait vectors normalized to be between 0 and 1
    similarities = 0.5 * (1 + torch.nn.functional.cosine_similarity(traits_p, traits_q, dim=1))
    # Get logits for each agent in the pairs
    logits_p = old_logits[timesteps, agent_i]
    logits_q = old_logits[timesteps, agent_j]
    # Compute JSD loss
    jsd_loss = threshed_jsd_loss(logits_p, logits_q, similarities, params.hypernet_jsd_threshold)
    
    total_loss =params.hypernet_entropy_weight * entropy_loss_val  - params.hypernet_jsd_weight * jsd_loss

    if writer is not None:
        writer.add_scalar('Hypernet/Entropy Loss', entropy_loss_val.item(), global_step = params.global_steps)
        writer.add_scalar('Hypernet/JSD Loss', jsd_loss.item(), global_step = params.global_steps)

    return total_loss

