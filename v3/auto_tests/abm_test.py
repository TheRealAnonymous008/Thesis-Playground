from collections import defaultdict
import numpy as np 
import torch
from models.model import *
from use_case.sr_abm import DiseaseSpreadEnv
from use_case.influencer_abm import InfluencerEnv
from torch.utils.tensorboard import SummaryWriter
from collections import Counter


def si_eval_loop(model: PPOModel, 
        env : DiseaseSpreadEnv,
        num_episodes=10,
        temperature=-1,
        set_agent_types=None):
    
    device = model.config.device
    all_episode_actions = []
    all_episode_stats = []  # Store stats per episode
    
    with torch.no_grad():
        for _ in range(num_episodes):
            obs = env.reset()
            if set_agent_types:
                set_agent_types()
            done = False
            episode_actions = []
            episode_stats = []  # Store stats per timestep for this episode
            
            # Record initial state
            states = env.states
            episode_stats.append({
                'susceptible': np.sum(states == 0),
                'infected': np.sum(states == 1)
            })
            
            trait_vector = torch.tensor(env.get_traits(), device=device)
            traits_np = trait_vector.cpu().numpy()

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
                    actions = model.get_argmax_action(Q, is_continuous= env.is_continuous)
                else:
                    actions = model.get_action(Q, temperature, env.is_continuous)
                
                actions = env.postprocess_actions(actions)
                episode_actions.append(actions.copy())
                
                next_obs, rewards, dones, _ = env.step(
                    {agent: actions[i] for i, agent in enumerate(env.get_agents())}
                )
                done = np.any(list(dones.values()))
                obs = next_obs

                # Record state after step
                states = env.states
                episode_stats.append({
                    'susceptible': np.sum(states == 0),
                    'infected': np.sum(states == 1)
                })

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
            all_episode_stats.append(episode_stats)
    
    return all_episode_actions, all_episode_stats


def influencer_eval_loop(model: PPOModel, 
        env : InfluencerEnv,  # Updated env type
        num_episodes=10,
        temperature=-1,
        set_agent_types=None):
    
    device = model.config.device
    all_episode_actions = []
    all_episode_ideas = []  # Store idea vectors per episode
    
    with torch.no_grad():
        for _ in range(num_episodes):
            obs = env.reset()
            if set_agent_types:
                set_agent_types()
            done = False
            episode_actions = []
            episode_ideas = []  # Store idea vectors per timestep
            
            # Record initial idea vectors
            episode_ideas.append(env.true_idea.copy())  # Store copy of initial ideas
            
            trait_vector = torch.tensor(env.get_traits(), device=device)
            traits_np = trait_vector.cpu().numpy()

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
                
                next_obs, rewards, dones, _ = env.step(
                    {agent: actions[i] for i, agent in enumerate(env.get_agents())}
                )
                done = np.any(list(dones.values()))
                obs = next_obs

                # Record idea vectors after step
                episode_ideas.append(env.true_idea.copy())  # Store copy of current ideas

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
            all_episode_ideas.append(episode_ideas)  # Store all timesteps for episode
    
    return all_episode_actions, all_episode_ideas