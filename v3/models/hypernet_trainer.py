from .losses import * 
from .model import * 
from .base_env import *
from tests.eval import *

import torch 
from torch.utils.tensorboard import SummaryWriter
from .param_settings import TrainingParameters
from .ppo_trainer import *
from .sac_trainer import *

def train_hypernet(model: SACModel, env: BaseEnv, exp: TensorDict, params: TrainingParameters, writer: SummaryWriter = None):
    num_agents = int(env.n_agents * params.sampled_agents_proportion)
    entropy_loss_val = entropy_loss(exp["means"], exp["std"], params.entropy_target)

    # JSD loss computation with sampled agent pairs within the same timestep
    buffer_length = len(exp)
    num_pairs = params.hypernet_samples
    
    # Generate timestep indices and agent pairs ensuring i != j
    timesteps = torch.randint(0, buffer_length, (num_pairs,), device=model.device)
    agent_i = torch.randint(0, num_agents, (num_pairs,), device=model.device)
    agent_j = torch.randint(0, num_agents - 1, (num_pairs,), device=model.device)
    agent_j[agent_j >= agent_i] += 1  # Ensure j != i

    with torch.no_grad():
        # Get trait vectors for each agent pair
        traits_p = exp["traits"][timesteps, agent_i]
        traits_q = exp["traits"][timesteps, agent_j]

        # Compute cosine similarity between trait vectors normalized to be between 0 and 1
        similarities = 0.5 * (1 + torch.nn.functional.cosine_similarity(traits_p, traits_q, dim=1))
        
        # Compute Q_i and Q_j for each agent pair using current model and stored weights
        # For agent i
        obs = exp["observations"][timesteps, agent_i]
        belief = exp["belief"][timesteps, agent_i]
        com = exp["com"][timesteps, agent_i]


    wh_policy_i = exp["wh"]["policy"][timesteps, agent_i]
    wh_policy_j = exp["wh"]["policy"][timesteps, agent_j]
    
    temp_Q, _, _ = model.actor_encoder.homogeneous_forward(obs, belief, com)
    Q_i = apply_heterogeneous_weights(temp_Q, wh_policy_i, sigmoid=False)
    Q_j = apply_heterogeneous_weights(temp_Q, wh_policy_j, sigmoid=False)
    
    # Compute JSD loss with new Q values
    jsd_loss = threshed_jsd_loss(Q_i, Q_j, similarities, params.hypernet_jsd_threshold)
    # Diversity loss computation using latent variables
    lv_i = exp["lv"][timesteps, agent_i]
    lv_j = exp["lv"][timesteps, agent_j]
    div_sim = 1 - 0.5 * (1 + torch.nn.functional.cosine_similarity(lv_i, lv_j, dim=1))
    div_sim = div_sim.mean()

    e_loss = params.hypernet_entropy_weight * entropy_loss_val
    j_loss = params.hypernet_jsd_weight * jsd_loss
    d_loss = params.hypernet_diversity_weight * div_sim
    total_loss = -j_loss

    if writer is not None:
        writer.add_scalar('Hypernet/Entropy Loss', e_loss.item(), global_step=params.global_steps)
        writer.add_scalar('Hypernet/JSD Loss', j_loss.item(), global_step=params.global_steps)
        writer.add_scalar('Hypernet/Diversity', d_loss.item(), global_step=params.global_steps)
        writer.add_scalar('Hypernet/Total Loss', total_loss.item(), global_step=params.global_steps)

    return total_loss