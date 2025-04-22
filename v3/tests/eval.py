import numpy as np 
import torch
from models.model import Model

def find_pure_equilibria(p1_payoff, p2_payoff):
    """Find pure strategy Nash equilibria"""
    pure_eq = []
    rows, cols = p1_payoff.shape
    
    for i in range(rows):
        for j in range(cols):
            if (p1_payoff[i,j] == np.max(p1_payoff[:,j]) and 
                p2_payoff[i,j] == np.max(p2_payoff[i,:])):
                pure_eq.append(((i,j), (float(p1_payoff[i,j]), float(p2_payoff[i,j]))))
    
    return pure_eq



def evaluate_policy(model : Model, env, num_episodes=10):
    """Evaluate current policy and return average episode return"""
    total_returns = []
    actions_array = []
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
                actions_array.append(actions)
                
                # Step environment
                next_obs, rewards, dones, _ = env.step(
                    {agent: int(actions[i]) for i, agent in enumerate(env.agents)}
                )
                episode_return += sum(rewards.values()) / len(env.agents)
                done = any(dones.values())
                obs = next_obs
                
            total_returns.append(episode_return)
    
    mean_returns = np.mean(total_returns)
    total_returns = np.sum(total_returns)
    mean_action_dist = np.histogram(actions, [i for i in range(0, model.config.d_action + 1)])
    print(f"Average Return: {mean_returns}")
    print(f"Total returns: {total_returns}")
    print(f"Action Dist, {mean_action_dist}")