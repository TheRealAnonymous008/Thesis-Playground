from .losses import * 
from .model import * 

import gymnasium as gym 
import torch 
from torch.distributions import Categorical
from dataclasses import dataclass 
from tqdm import tqdm 

def run_model(model : Model, env : gym.Env):
    # Hypernet Steps 
    for _ in tqdm(range(1000)): 
        params = model.config
        obs = env.reset()

        device = params.device

        # Network steps
        belief_vector = torch.ones((params.n_agents, 1), device = device)
        trait_vector = torch.ones((params.n_agents, 1), device= device)
        com_vector = torch.zeros((params.n_agents, params.d_comm_state), device = device)
        lv, wh = model.hypernet.forward(trait_vector, belief_vector)
        
        p_weights, b_weights, e_weights, f_weights, d_weights, u_weights  = wh 
        
        # Actor encoder phase 
        for _ in range(1000): 
            obs_array = np.stack([obs[agent] for agent in env.agents])  # Convert to ndarray
            obs = torch.FloatTensor(obs_array).to(device)

            Q, h, ze = model.actor_encoder.forward(obs, belief_vector, com_vector, p_weights, b_weights, e_weights)
            dists = Categorical(logits=Q)
            actions = dists.sample().cpu().numpy()
            actions = {agent: int(actions[i]) 
                        for i, agent in enumerate(env.agents)}
            
            # Environment step
            next_obs, rewards, _, _ = env.step(actions)
            obs = next_obs

@dataclass
class TrainingParameters:
    outer_loops: int = 5
    hypernet_training_loops: int = 5
    actor_training_loops: int = 5
    learning_rate: float = 1e-3
    gamma: float = 0.99  # Discount factor
    jsd_threshold: float = 0.5  # Threshold for JSD loss
    experience_buffer_size : int = 1000

def compute_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """JIT-optimized version for long sequences."""
    rewards = torch.Tensor(rewards)
    returns = torch.empty_like(rewards)
    R = torch.tensor(0.0, device=rewards.device)
    for i in reversed(range(rewards.size(0))):
        R = rewards[i] + gamma * R
        returns[i] = R
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def collect_experiences(model, env, params):
    device = model.config.device
    obs = env.reset()
    batch_obs = []
    batch_actions = []
    batch_rewards = []
    batch_logits = []
    batch_lv = []
    batch_wh = []
    done = False

    for _ in range(1000):  # Collect 1000 steps per batch
        obs_array = np.stack([obs[agent] for agent in env.agents])
        obs_tensor = torch.FloatTensor(obs_array).to(device)
        
        # Hypernet forward
        belief_vector = torch.ones((model.config.n_agents, 1), device=device)
        trait_vector = torch.ones((model.config.n_agents, 1), device=device)
        com_vector = torch.zeros((model.config.n_agents, model.config.d_comm_state), device=device)
        lv, wh = model.hypernet(trait_vector, belief_vector)
        
        # Actor encoder forward
        Q, h, ze = model.actor_encoder(obs_tensor, belief_vector, com_vector, *wh[:3])
        dists = Categorical(logits=Q)
        actions = dists.sample().cpu().numpy()
        actions_dict = {agent: int(actions[i]) for i, agent in enumerate(env.agents)}
        
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
        
        obs = next_obs
        if any(dones.values()):
            break
    
    # Convert lists to tensors
    batch_obs = torch.stack(batch_obs)
    batch_actions = torch.tensor(np.array(batch_actions), device=device)
    batch_rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32, device=device)
    batch_logits = torch.stack(batch_logits)
    batch_lv = torch.stack(batch_lv)
    
    # Compute returns
    returns = compute_returns(batch_rewards.cpu().numpy(), params.gamma).to(device)
    
    return {
        'observations': batch_obs,
        'actions': batch_actions,
        'rewards': batch_rewards,
        'logits': batch_logits,
        'lv': batch_lv,
        'returns': returns,
    }


def train_model(model: Model, env: gym.Env, params: TrainingParameters):
    hyper_optim = torch.optim.Adam(model.hypernet.parameters(), lr=params.learning_rate)
    actor_optim = torch.optim.Adam(model.actor_encoder.parameters(), lr=params.learning_rate)

    for i in range(params.outer_loops):
        print(f"Epock {i}")
        model.hypernet.requires_grad_(False)
        model.actor_encoder.requires_grad_(False)
        model.decoder_update.requires_grad_(False)
        model.filter.requires_grad_(False)

        train_actor(model, env, params, actor_optim)
        evaluation  = evaluate_policy(model, env)
        print(f"Average Return: {evaluation}")
        # train_hypernet(model, env, params, hyper_optim)

        # TODO: Add filter and decoder training

def train_actor(model: Model, env: gym.Env, params: TrainingParameters, optim):
    model.actor_encoder.requires_grad_(True)
    for _ in tqdm(range(params.actor_training_loops), desc = "Actor Loop"):
        exp = collect_experiences(model, env, params)
        
        # Calculate policy gradient loss
        dists = Categorical(logits=exp['logits'])
        log_probs = dists.log_prob(exp['actions'])
        actor_loss = -(log_probs * exp['returns']).mean()
        
        optim.zero_grad()
        actor_loss.backward()
        optim.step()
    model.actor_encoder.requires_grad_(False)

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
        Q, _, _ = model.actor_encoder(
            exp['observations'], belief, com_vector, *wh[:3]
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


def evaluate_policy(model, env, num_episodes=10):
    """Evaluate current policy and return average episode return"""
    total_returns = []
    device = model.config.device
    
    with torch.no_grad():
        for _ in range(num_episodes):
            obs = env.reset()
            episode_return = 0
            done = False
            
            while not done:
                obs_array = np.stack([obs[agent] for agent in env.agents])
                obs_tensor = torch.FloatTensor(obs_array).to(device)
                
                # Generate hypernet weights
                belief = torch.ones((model.config.n_agents, 1), device=device)
                trait = torch.ones((model.config.n_agents, 1), device=device)
                com_vector = torch.zeros((model.config.n_agents, model.config.d_comm_state), device=device)
                lv, wh = model.hypernet(trait, belief)
                
                # Get action distribution
                Q, _, _ = model.actor_encoder(obs_tensor, belief, com_vector, *wh[:3])
                actions = Q.argmax(dim=-1).cpu().numpy()
                
                # Step environment
                next_obs, rewards, dones, _ = env.step(
                    {agent: int(actions[i]) for i, agent in enumerate(env.agents)}
                )
                episode_return += sum(rewards.values())
                done = any(dones.values())
                obs = next_obs
                
            total_returns.append(episode_return)
    
    return np.mean(total_returns)

def evaluate_policy(model, env, num_episodes=10):
    """Evaluate current policy and return average episode return"""
    total_returns = []
    device = model.config.device
    
    with torch.no_grad():
        for _ in range(num_episodes):
            obs = env.reset()
            episode_return = 0
            done = False
            
            while not done:
                obs_array = np.stack([obs[agent] for agent in env.agents])
                obs_tensor = torch.FloatTensor(obs_array).to(device)
                
                # Generate hypernet weights
                belief = torch.ones((model.config.n_agents, 1), device=device)
                trait = torch.ones((model.config.n_agents, 1), device=device)
                com_vector = torch.zeros((model.config.n_agents, model.config.d_comm_state), device=device)
                lv, wh = model.hypernet(trait, belief)
                
                # Get action distribution
                Q, _, _ = model.actor_encoder(obs_tensor, belief, com_vector, *wh[:3])
                actions = Q.argmax(dim=-1).cpu().numpy()
                
                # Step environment
                next_obs, rewards, dones, _ = env.step(
                    {agent: int(actions[i]) for i, agent in enumerate(env.agents)}
                )
                episode_return += sum(rewards.values())
                done = any(dones.values())
                obs = next_obs
                
            total_returns.append(episode_return)
    
    return np.mean(total_returns)