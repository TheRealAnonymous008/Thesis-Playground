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
    policy_loss = -torch.min(surr1, surr2).mean(dim=0)  # [agents]

    # Agent-wise value loss
    value_loss = torch.nn.functional.huber_loss(new_values.mean(dim=0), returns.mean(dim=0), reduction='none', delta = 10.0) # [agents]

    entropy = new_dists.entropy()
    entropy_loss = entropy.mean(dim=0)  # [agents]
    
    # Return unaggregated agent-wise losses
    return policy_loss, value_loss, entropy_loss


def train_ppo_actor(model: SACModel, env: BaseEnv, exp: TensorDict, params: TrainingParameters, writer : SummaryWriter =None):
    old_logits = exp["logits"]
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
    V_all = model.q1(
        obs_all,
        belief_actor,
        com_all,
        wh_all["critic"]
    ).squeeze(-1)

    # Reshape V outputs to (buffer_size, agent_no)
    new_values = V_all.view(buffer_size, agent_no)
    # Calculate losses
    policy_loss, value_loss, entropy = compute_core_ppo_losses(
        new_logits, new_values, exp['actions'], advantages, returns, old_logits, params
    )

    policy_loss =  params.actor_performance_weight * policy_loss.mean()
    value_loss = params.value_loss_coeff * value_loss.mean()
    
    entropy = entropy.mean()
    entropy_regularization = params.entropy_coeff * entropy

    actor_loss = policy_loss - entropy_regularization
    critic_loss = value_loss
    total_loss = actor_loss + critic_loss

    if writer is not None:
        writer.add_scalar('Actor/Metrics/Policy Loss', policy_loss.item(), global_step= params.global_steps)
        writer.add_scalar('Actor/Metrics/Actor Loss', actor_loss.item(), global_step= params.global_steps)
        writer.add_scalar('Actor/Metrics/Critic Loss', critic_loss.item(), global_step= params.global_steps)
        writer.add_scalar('Actor/Metrics/Entropy', entropy_regularization.item(), global_step= params.global_steps)
        writer.add_scalar('Actor/Metrics/Total Loss', total_loss.item(), global_step= params.global_steps)

    return total_loss
