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
    # JSD loss computation with sampled agent pairs within the same timestep
    buffer_length = len(exp)
    num_pairs = params.hypernet_samples
    
    # Generate timestep indices and agent pairs ensuring i != j
    timesteps = torch.randint(0, buffer_length, (num_pairs,), device=model.device)
    agent_i = torch.randint(0, num_agents, (num_pairs,), device=model.device)
    agent_j = torch.randint(0, num_agents - 1, (num_pairs,), device=model.device)
    agent_j[agent_j >= agent_i] += 1  # Ensure j != i

    traits_i = exp["traits"][timesteps, agent_i]
    traits_j = exp["traits"][timesteps, agent_j]

    # Get the weights to be used
    lv_i, weights_i, mu_i, sigma_i = model.hypernet.forward(
        traits_i,  
        exp["observations"][timesteps, agent_i], 
        exp["belief"][timesteps, agent_i], 
        exp["com"][timesteps, agent_i]
    )
    lv_j, weights_j, mu_j, sigma_j = model.hypernet.forward(
        traits_j, 
        exp["observations"][timesteps, agent_j], 
        exp["belief"][timesteps, agent_j], 
        exp["com"][timesteps, agent_j]
    )

    # Compute the JSD loss
    with torch.no_grad():
        # Get trait vectors for each agent pair
        traits_p = exp["traits"][timesteps, agent_i]
        traits_q = exp["traits"][timesteps, agent_j]

        # Compute cosine similarity between trait vectors normalized to be between 0 and 1
        similarities = 0.5 * (1 + torch.nn.functional.cosine_similarity(traits_p, traits_q, dim=1))

    wh_policy_i = weights_i["policy"]
    wh_policy_j = weights_j["policy"]

    with torch.no_grad():
        temp_Q_i, _, _ = model.actor_encoder.homogeneous_forward(
            exp["observations"][timesteps, agent_i], 
            exp["belief"][timesteps, agent_i], 
            exp["com"][timesteps, agent_i])
        
        
        temp_Q_j, _, _ = model.actor_encoder.homogeneous_forward(
            exp["observations"][timesteps, agent_j], 
            exp["belief"][timesteps, agent_j], 
            exp["com"][timesteps, agent_j])
        
    Q_ii = apply_heterogeneous_weights(temp_Q_i, wh_policy_i, sigmoid=False)
    Q_ij = apply_heterogeneous_weights(temp_Q_i, wh_policy_j, sigmoid=False)
    Q_ji = apply_heterogeneous_weights(temp_Q_j, wh_policy_i, sigmoid=False)
    Q_jj = apply_heterogeneous_weights(temp_Q_j, wh_policy_j, sigmoid=False)
    jsd_loss = (threshed_jsd_loss(Q_ii, Q_ij, similarities, params.hypernet_jsd_threshold) + \
        threshed_jsd_loss(Q_ji, Q_jj, similarities, params.hypernet_jsd_threshold)) / 2.0
    
    # Diversity loss computation using latent variables
    div_sim = 1 - 0.5 * (1 + torch.nn.functional.cosine_similarity(lv_i, lv_j, dim=1))
    div_sim = div_sim.mean()

    # Entropy loss
    means = torch.cat([mu_i, mu_j])
    stds = torch.cat([sigma_i, sigma_j])
    entropy_loss_val = entropy_loss(means, stds, params.entropy_target)


    e_loss = params.hypernet_entropy_weight * entropy_loss_val
    j_loss = params.hypernet_jsd_weight * jsd_loss
    d_loss = params.hypernet_diversity_weight * div_sim
    total_loss = e_loss - d_loss - j_loss

    if writer is not None:
        writer.add_scalar('Hypernet/Entropy Loss', e_loss.item(), global_step=params.global_steps)
        writer.add_scalar('Hypernet/JSD', j_loss.item(), global_step=params.global_steps)
        writer.add_scalar('Hypernet/Diversity', d_loss.item(), global_step=params.global_steps)
        writer.add_scalar('Hypernet/Total Loss', total_loss.item(), global_step=params.global_steps)

    return total_loss