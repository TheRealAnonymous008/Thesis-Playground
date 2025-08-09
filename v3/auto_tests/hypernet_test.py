from collections import defaultdict
import numpy as np 
import torch
from models.model import *
from use_case.baseline_het import BaselineHeterogeneous
from torch.utils.tensorboard import SummaryWriter
from collections import Counter

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

def hypernet_test(model: PPOModel, 
        env : BaselineHeterogeneous,
        num_episodes=10,
        temperature=-1,
        set_agent_types=None):
    
    device = model.config.device
    all_episode_actions = []
    action_distributions = {}
    
    with torch.no_grad():
        for _ in range(num_episodes):
            obs = env.reset()
            if set_agent_types:
                set_agent_types()
            done = False
            episode_actions = []
            
            # Get agent types (one-hot encoded) and convert to type indices
            traits_np = env.traits
            type_indices = np.argmax(traits_np, axis=1)

            while not done:
                obs_array = np.stack([obs[agent] for agent in env.agents])
                obs_tensor = torch.FloatTensor(obs_array).to(device)

                trait_vector = torch.tensor(env.traits, device=device)
                belief_vector = torch.tensor(env.beliefs, device=device)
                com_vector = torch.tensor(env.comm_state, device=device)
                lv, wh, _, _ = model.hypernet(trait_vector, obs_tensor, belief_vector, com_vector)

                Q, h, z = model.actor_encoder.forward(
                    obs_tensor, 
                    belief_vector, 
                    com_vector, 
                    wh["policy"], 
                    wh["belief"], 
                    wh["encoder"]
                )
                
                if temperature < 0:
                    actions = model.get_argmax_action(Q, is_continuous=env.is_continuous)
                else:
                    actions = model.get_action(Q, temperature, env.is_continuous)
                
                actions = env.postprocess_actions(actions)
                episode_actions.append(actions.copy())
                
                # Record actions per type
                for agent_idx, (action, type_idx) in enumerate(zip(actions, type_indices)):
                    if type_idx not in action_distributions:
                        action_distributions[type_idx] = []
                    action_distributions[type_idx].append(action)
                
                next_obs, rewards, dones, _ = env.step(
                    {agent: actions[i] for i, agent in enumerate(env.get_agents())}
                )
                done = np.any(list(dones.values()))
                obs = next_obs

                source_indices = torch.arange(0, env.n_agents, dtype=torch.long)
                neighbor_indices, Mij, reverses = env.sample_neighbors()
                Mij = Mij.to(model.device)
                reverses = reverses.to(model.device)
                messages = model.filter.forward(z, Mij, wh["filter"])

                zdj, Mji = model.decoder_update.forward(messages, reverses,  wh["decoder"], wh["update_mean"], wh["update_std"])
                
                env.set_beliefs(h)
                env.set_comm_state(neighbor_indices, zdj)
                env.update_edges(neighbor_indices, source_indices, Mji)
            
            all_episode_actions.append(episode_actions)
    
    return all_episode_actions, action_distributions