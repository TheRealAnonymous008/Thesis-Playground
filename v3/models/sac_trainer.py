from .model import * 
from .base_env import *

from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from .param_settings import TrainingParameters
from .train_utils import * 


def compute_core_sac_losses(current_q1: torch.Tensor,
                            current_q2: torch.Tensor,
                            target_q: torch.Tensor,
                            policy_loss: torch.Tensor,
                            alpha: torch.Tensor,
                            log_probs: torch.Tensor,
                            target_entropy: float,
                            automatic_entropy_tuning: bool) -> tuple:
    """Compute SAC Q losses, policy loss, and alpha loss if applicable."""
    q1_loss = torch.nn.functional.mse_loss(current_q1, target_q)
    q2_loss = torch.nn.functional.mse_loss(current_q2, target_q)

    
    if automatic_entropy_tuning:
        alpha_loss = -(log_probs + target_entropy).detach() * alpha
        alpha_loss = alpha_loss.mean()
    else:
        alpha_loss = torch.tensor(0.0, device=current_q1.device)
    
    return q1_loss, q2_loss, policy_loss, alpha_loss

def train_sac_actor(model: SACModel, env: BaseEnv, exp: TensorDict, params: TrainingParameters, writer: SummaryWriter = None):
    # Reshape inputs for hypernetwork
    traits_all = exp["traits"].view(-1, exp["traits"].shape[-1])
    belief_hyper = exp["belief"].view(-1, exp["belief"].shape[-1])
    
    # Compute hypernet outputs for all agents and buffer entries
    _, wh_all, _, _ = model.hypernet.forward(traits_all, belief_hyper)
    
    # Reshape inputs for networks
    obs_all = exp["observations"].view(-1, exp["observations"].shape[-1])
    next_obs_all = exp["next_observations"].view(-1, exp["next_observations"].shape[-1])
    belief_actor = exp["belief"].view(-1, exp["belief"].shape[-1])
    com_all = exp["com"].view(-1, exp["com"].shape[-1])
    rewards = normalize_tensor(exp["rewards"]).view(-1, 1)
    dones = exp["done"].view(-1, 1)
    
    # Compute next actions and log probabilities (target policy)
    with torch.no_grad():
        next_actions, _, _ = model.actor_encoder(
            next_obs_all, belief_actor, com_all,
            wh_all["policy"], wh_all["belief"], wh_all["encoder"]
        )

        dones = dones.float().repeat_interleave(env.n_agents, dim=0).to(model.device)  # Reshape here

        dists = Categorical(logits = next_actions)
        Q =  dists.sample().view(-1, 1)

        # Compute target Q values
        # TODO: This is placeholder. It really should be the next belief and com state.
        target_q1 = model.target_q1(next_obs_all,  belief_actor, com_all, wh_all["q1"])
        target_q2 = model.target_q2(next_obs_all,  belief_actor, com_all, wh_all["q2"])
        target_q = torch.min(target_q1, target_q2) - model.alpha * Q
        target_q = rewards + params.gamma * (1 - dones) * target_q
    
    # Compute current Q estimates
    current_q1 = model.q1(obs_all,  belief_actor, com_all, wh_all["q1"])
    current_q2 = model.q2(obs_all, belief_actor, com_all , wh_all["q2"])
    
    # Freeze Q parameters for policy update
    for param in model.q1.parameters():
        param.requires_grad = False
    for param in model.q2.parameters():
        param.requires_grad = False
    
    # Sample new actions for policy update
    Q, _, _  = model.actor_encoder(
        obs_all, belief_actor, com_all,
        wh_all["policy"], wh_all["belief"], wh_all["encoder"]
    )
    
    # Compute Q values for new actions
    q1_new = model.q1(obs_all, belief_actor, com_all, wh_all["q1"])
    q2_new = model.q2(obs_all, belief_actor, com_all, wh_all["q2"])
    q_new = torch.min(q1_new, q2_new)
    
    # Policy loss
    policy_loss = (model.alpha * Q - q_new).mean()
    
    # Unfreeze Q parameters
    for param in model.q1.parameters():
        param.requires_grad = True
    for param in model.q2.parameters():
        param.requires_grad = True
    
    # Compute losses
    q1_loss, q2_loss, policy_loss, alpha_loss = compute_core_sac_losses(
        current_q1, current_q2, target_q.detach(), policy_loss,
        model.alpha,
        Q, params.target_entropy, params.automatic_entropy_tuning
    )
    
    # Logging
    if writer is not None:
        writer.add_scalar('Actor/Q1', q1_loss.item(), params.global_steps)
        writer.add_scalar('Actor/Q2', q2_loss.item(), params.global_steps)
        writer.add_scalar('Actor/Policy', policy_loss.item(), params.global_steps)
        writer.add_scalar('Actor/Alpha', model.alpha, params.global_steps)
        writer.add_scalar('Actor/Alpha_Loss', alpha_loss.item(), params.global_steps)
    
    total_loss = q1_loss + q2_loss + policy_loss + alpha_loss
    return total_loss