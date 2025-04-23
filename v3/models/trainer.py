from .losses import * 
from .model import * 
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
    learning_rate: float = 1e-3
    gamma: float = 0.99  # Discount factor
    jsd_threshold: float = 0.5  # Threshold for JSD loss
    experience_buffer_size : int = 3         # Warning: You shouldn't make this too big because you will have many agents in the env.
    entropy_coeff: float = 0

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

def collect_experiences(model : Model, env : gym.Env, params):
    device = model.config.device
    obs = env.reset()
    batch_obs = []
    batch_actions = []
    batch_rewards = []
    batch_logits = []
    batch_lv = []
    batch_values = []
    batch_wh = []
    done = False

    for i in range(params.experience_buffer_size):
        
        obs_array = np.stack([obs[agent] for agent in env.agents])
        obs_tensor = torch.FloatTensor(obs_array).to(device)
        
        # Hypernet forward
        belief_vector = torch.ones((model.config.n_agents, 1), device=device)
        trait_vector = torch.ones((model.config.n_agents, 1), device=device)
        com_vector = torch.zeros((model.config.n_agents, model.config.d_comm_state), device=device)
        lv, wh = model.hypernet(trait_vector, belief_vector)
        
        # Actor encoder forward
        Q, _, _ = model.actor_encoder.forward(
            obs_tensor, 
            belief_vector, 
            com_vector, 
            (torch.ones_like(wh["policy"][0]), torch.ones_like(wh["policy"][1])), 

            wh["belief"], 
            wh["encoder"]
        )

        dists = Categorical(logits=Q)
        actions = dists.sample().cpu().numpy()
        actions_dict = {agent: int(actions[i]) for i, agent in enumerate(env.agents)}

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
        rewards = np.array([rewards[agent] for agent in env.agents])

        # Store experience
        batch_obs.append(obs_tensor)
        batch_actions.append(actions)
        batch_rewards.append(rewards)
        batch_logits.append(Q)
        batch_lv.append(lv)
        batch_wh.append(wh)
        batch_values.append(values)
        
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
    # Compute returns
    returns = compute_returns(batch_rewards.cpu().numpy(), params.gamma).to(device)
    return {
        'observations': batch_obs,
        'actions': batch_actions,
        'rewards': batch_rewards,
        'logits': batch_logits,
        'lv': batch_lv,
        'returns': returns,
        "values" : batch_values,
    }


def train_model(model: Model, env: gym.Env, params: TrainingParameters):
    hyper_optim = torch.optim.Adam(model.hypernet.parameters(), lr=params.learning_rate)
    actor_optim = torch.optim.Adam(model.actor_encoder.parameters(), lr=params.learning_rate)

    for i in range(params.outer_loops):
        print(f"Epoch {i}")
        model.requires_grad_(False)

        train_actor(model, env, params, actor_optim)
        evaluate_policy(model, env)
        # train_hypernet(model, env, params, hyper_optim)

        # TODO: Add filter and decoder training

def train_actor(model: Model, env: gym.Env, params: TrainingParameters, optim):
    # Enable gradients for both actor and critic components
    model.actor_encoder.requires_grad_(True)
    model.actor_encoder_critic.requires_grad_(True)
    
    for _ in tqdm(range(params.actor_training_loops), desc="Actor-Critic Loop"):
        exp = collect_experiences(model, env, params)  
        
        # Calculate policy gradient loss with advantages
        dists = Categorical(logits=exp['logits'])
        log_probs = dists.log_prob(exp['actions'])
        entropy = dists.entropy().mean()
        
        # Compute advantages using critic's value estimates
        advantages = exp['returns'] - exp['values'].detach()
        actor_loss_pg = -(log_probs * advantages).mean()
        actor_loss = (1 - params.entropy_coeff) * actor_loss_pg + params.entropy_coeff * entropy
        
        # Calculate critic loss (value function MSE)
        critic_loss = torch.nn.functional.mse_loss(exp["returns"], exp["values"])
        
        # Combine losses and update
        total_loss = actor_loss + critic_loss
        
        optim.zero_grad()
        total_loss.backward()
        optim.step()
    
    # Disable gradients after training
    model.actor_encoder.requires_grad_(False)
    model.actor_encoder_critic.requires_grad_(False)

def train_hypernet(model : Model, env : gym.Env, params : TrainingParameters, optim):
    model.hypernet.requires_grad_(True)
    for _ in tqdm(range(params.hypernet_training_loops), desc = "Hypernet Loop"):
        exp = collect_experiences(model, env, params)
        
        # Recompute logits with current hypernet
        belief = torch.ones((model.config.n_agents, 1), device=model.config.device)
        trait = torch.ones((model.config.n_agents, 1), device=model.config.device)
        com_vector = torch.zeros((model.config.n_agents, model.config.d_comm_state), 
                                device=model.config.device)
        
        lv, wh = model.hypernet(trait, belief)
        Q, _, _ = model.actor_encoder.forward(
            exp['observations'], 
            belief, 
            com_vector, 
            wh["policy"], 
            wh["belief"], 
            wh["encoder"]
        )
        
        # Calculate losses
        dists = Categorical(logits=Q)
        log_probs = dists.log_prob(exp['actions'])
        rl_loss = -(log_probs * exp['returns']).mean()
        entropy_loss_val = entropy_loss(lv).mean()
        
        # JSD between first and second half of episode
        p = Q[:len(Q)//2]
        q = Q[len(Q)//2:]
        s = torch.norm(exp['observations'][:len(Q)//2] - exp['observations'][len(Q)//2:], dim=-1)
        jsd_loss = threshed_jsd_loss(p, q, s, params.jsd_threshold)
        
        total_loss = rl_loss + entropy_loss_val + jsd_loss
        
        optim.zero_grad()
        total_loss.backward()
        optim.step()
    model.hypernet.requires_grad_(False)
