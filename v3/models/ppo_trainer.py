from .model import * 
from .base_env import *

from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from .param_settings import TrainingParameters
from .train_utils import * 


def compute_core_ppo_losses(new_logits: torch.Tensor, 
                            new_values: torch.Tensor, 
                            actions: torch.Tensor, 
                            advantages: torch.Tensor,
                            returns: torch.Tensor, 
                            old_logits: torch.Tensor,
                            params: TrainingParameters
                            ) -> tuple:
    """Compute PPO policy loss, value loss, and entropy with per-agent calculations."""

    old_dists = Categorical(logits=old_logits)
    old_log_probs = old_dists.log_prob(actions)

    new_dists = Categorical(logits=new_logits)
    new_log_probs = new_dists.log_prob(actions)
    
    # Independent policy optimization per agent
    ratios = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1-params.clip_epsilon, 1+params.clip_epsilon) * advantages

    # Agent-wise policy loss (mean over time first)
    policy_loss = -torch.min(surr1, surr2).sum(dim=0)  # [agents]

    # Agent-wise value loss
    value_loss = torch.nn.functional.huber_loss(new_values.sum(dim=0), returns.sum(dim=0), reduction='none', delta=10.0)  # [agents]

    entropy = new_dists.entropy()
    entropy_loss = entropy.mean(dim=0)  # [agents]
    
    # Return unaggregated agent-wise losses
    return policy_loss, value_loss, entropy_loss


def compute_gae(rewards: torch.Tensor, 
                values: torch.Tensor, 
                dones: torch.Tensor, 
                gamma: float, 
                gae_lambda: float) -> tuple:
    """
    Compute Generalized Advantage Estimation (GAE) with per-agent handling.
    Inputs:
        rewards: (timesteps, agents)
        values: (timesteps, agents)
        dones: (timesteps,)
    """
    buffer_size, agent_no = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    # Convert dones to agent-wise shape (timesteps, 1) for broadcasting
    dones = dones.float().unsqueeze(1).to(device = rewards.device)  # (timesteps, 1)

    # Initialize last advantage and return
    next_value = torch.zeros(agent_no, device=rewards.device)
    gae = torch.zeros(agent_no, device=rewards.device)

    for t in reversed(range(buffer_size)):
        # Compute delta using current and next timestep values
        if t == buffer_size - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value = torch.zeros_like(values[t])
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_value = values[t + 1]
        
        reward = rewards[t]

        delta = reward + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae

    # Compute returns as advantages + values
    returns = advantages + values
    return advantages, returns



def train_ppo_actor(model: PPOModel, env: BaseEnv, exp: TensorDict, params: TrainingParameters, writer: SummaryWriter = None):
    old_logits = exp["logits"]
    rewards = normalize_tensor(exp["rewards"])
    values = normalize_tensor(exp["values"])
    dones = exp["done"]
    
    # Compute GAE advantages and returns
    advantages, returns = compute_gae(rewards, values, dones, params.gamma, params.gae_lambda)
    advantages = normalize_tensor(advantages)

    # Reshape all inputs to combine buffer_size and agent_no into a single batch dimension
    buffer_size, agent_no, *_ = exp["traits"].shape
    traits_all = exp["traits"].view(-1, exp["traits"].shape[-1])
    belief_all = exp["belief"].view(-1, exp["belief"].shape[-1])
    obs_all = exp["observations"].view(-1, exp["observations"].shape[-1])
    com_all = exp["com"].view(-1, exp["com"].shape[-1])

    # Compute hypernet outputs in one batch
    _, wh_all, _, _ = model.hypernet.forward(traits_all, obs_all, belief_all, com_all)

    # Actor forward pass
    Q_all, h_all, z_all = model.actor_encoder.forward(
        obs_all,
        belief_all,
        com_all,
        wh_all["policy"],
        wh_all["belief"],
        wh_all["encoder"]
    )
    new_logits = Q_all.view(buffer_size, agent_no, -1)

    # Critic forward pass
    V_all = model.actor_encoder_critic(
        obs_all,
        belief_all,
        com_all,
        wh_all["critic"]
    ).squeeze(-1)
    new_values = V_all.view(buffer_size, agent_no)

    # Calculate losses
    policy_loss, value_loss, entropy = compute_core_ppo_losses(
        new_logits, new_values, exp['actions'], advantages, returns, old_logits, params
    )

    policy_loss = params.actor_performance_weight * policy_loss.mean()
    value_loss = params.value_loss_coeff * value_loss.mean()
    entropy = entropy.mean()
    entropy_regularization = params.entropy_coeff * entropy

    actor_loss = policy_loss - entropy_regularization
    critic_loss = value_loss
    total_loss = actor_loss + critic_loss

    if writer is not None:
        writer.add_scalar('Actor/Metrics/Policy Loss', policy_loss.item(), global_step=params.global_steps)
        writer.add_scalar('Actor/Metrics/Actor Loss', actor_loss.item(), global_step=params.global_steps)
        writer.add_scalar('Actor/Metrics/Critic Loss', critic_loss.item(), global_step=params.global_steps)
        writer.add_scalar('Actor/Metrics/Entropy', entropy_regularization.item(), global_step=params.global_steps)
        writer.add_scalar('Actor/Metrics/Total Loss', total_loss.item(), global_step=params.global_steps)

    return total_loss