from .losses import * 
from .model import * 
from .base_env import *
from tests.eval import *

import torch 
from torch.utils.tensorboard import SummaryWriter
from .param_settings import TrainingParameters

def train_filter(model: SACModel | PPOModel, env: BaseEnv, exp: TensorDict, params: TrainingParameters, writer: SummaryWriter = None):
    total_loss = torch.tensor(0.0, device=params.device)
    
    messages = exp["messages"]
    logits = exp["pair_logits"]
    dones = exp["done"]
    
    # Determine valid timesteps where the next step is within the same episode
    if dones.size(0) < 2:
        return total_loss  # Not enough steps to form any pairs
    
    # Mask for valid t (from 0 to T-2) where dones[t] is False
    valid_t_mask = ~dones[:-1].squeeze() if dones.dim() > 1 else ~dones[:-1]
    valid_t_indices = torch.where(valid_t_mask)[0]
    
    if len(valid_t_indices) == 0:
        return total_loss  # No valid transitions
    
    n_valid = len(valid_t_indices)
    n_agents = messages.size(1)
    
    # Extract valid data slices
    messages_valid = messages[valid_t_indices]  # [n_valid, n_agents, d_messages]
    logits_next = logits[valid_t_indices + 1]   # [n_valid, n_agents, d_logits]
    
    # Collect all (message, next_logit) pairs across agents and timesteps
    p_list, q_list = [], []
    for agent_i in range(n_agents):
        # Messages from agent i at valid timesteps
        msg_i = messages_valid[:, agent_i, :]
        # Logits of the target agents at the next timestep
        logit_j = logits_next[torch.arange(n_valid, device=params.device), agent_i, :]
        
        p_list.append(msg_i)
        q_list.append(logit_j)
    
    # Concatenate all samples
    p = torch.cat(p_list, dim=0)
    q = torch.cat(q_list, dim=0)
    
    # Compute mutual information loss
    mi = mi_loss(p, q)
    total_loss = mi
    
    # Log to tensorboard if writer is provided
    if writer is not None:
        writer.add_scalar('Filter/mi_loss', mi.item(), params.global_steps)
    
    return total_loss