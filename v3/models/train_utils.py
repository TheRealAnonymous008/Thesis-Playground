import numpy as np 
import torch 
from tensordict import TensorDict
from .param_settings import TrainingParameters
from .base_env import * 

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

def add_exploration_noise(env : BaseEnv, logits: torch.Tensor, params: TrainingParameters, epoch: int = 0):
    """Add independent exploration noise per agent and timestep"""
    device = logits.device
    n_agents, n_actions = logits.shape

    if params.epsilon_period > 1: 
        epoch = epoch % params.epsilon_period
        params.epsilon = max(params.epsilon_end, params.epsilon_start * (params.epsilon_decay ** epoch))
    
    # Create exploration mask [buffer, agents]
    exploration_mask = torch.rand((n_agents), device=device) < params.epsilon
    
    # Create uniform logits for entire batch [buffer, agents, actions]
    uniform_logits = env.sample_action(device = device)
    with torch.no_grad():
        noise = torch.randn_like(logits) * params.noise_scale 


    # Apply epsilon-greedy mask
    modified_logits = torch.where(
        exploration_mask.unsqueeze(-1),  # Expand to [buffer, agents, 1]
        uniform_logits + noise,
        logits
    )

    return modified_logits

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
