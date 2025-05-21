from .losses import * 
from .model import * 
from .base_env import *
from tests.eval import *

import torch 
from torch.distributions import Categorical
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from .param_settings import TrainingParameters
from .ppo_trainer import *
from .sac_trainer import *
from .hypernet_trainer import * 
from .gnn_trainer import *
from .filter_trainer import *

# Temporary fix to avoid OMP duplicates. Not ideal though.
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def collect_experiences(model : PPOModel, env : BaseEnv, params : TrainingParameters, epoch = 1):
    device = model.config.device
    obs = env.reset()

    # Basic Stuff
    batch_obs = []
    batch_actions = []
    batch_rewards = []

    # Actpr
    batch_logits = []
    batch_belief = []
    batch_values = []
    batch_trait = []
    batch_com = []
    batch_dones = []

    # Heterogeneous Weights
    batch_lv = []
    batch_wh = []
    batch_ld_means = []
    batch_ld_std = []

    # Nexts
    batch_next_obs = []
    batch_next_belief = []
    batch_next_com = []

    # Additional data for GNN loss
    batch_Mij = []
    batch_Mji = []
    batch_ze = []
    batch_zd = []

    # Additional Data for Filter
    batch_messages = []

    sampled_agents = int(params.sampled_agents_proportion * env.n_agents)
    indices = np.random.choice(env.n_agents, size = sampled_agents, replace = False)
    
    for i in range(params.experience_sampling_steps):
        obs_array = np.stack([obs[agent] for agent in env.get_agents()])
        obs_tensor = torch.FloatTensor(obs_array).to(device)
        
        # Hypernet forward
        belief_vector = torch.tensor(env.beliefs, device = device)
        trait_vector = torch.tensor(env.traits, device = device)
        com_vector = torch.tensor(env.comm_state , device=device)
        lv, wh, mean, std = model.hypernet.forward(trait_vector, obs_tensor, belief_vector, com_vector)
        
        # Actor encoder forward
        Q, h, z = model.actor_encoder.forward(
            obs_tensor, 
            belief_vector, 
            com_vector, 
            wh["policy"],
            wh["belief"], 
            wh["encoder"]
        )
        Q = add_exploration_noise(Q, params, epoch)

        dists = Categorical(logits=Q)
        actions = dists.sample().cpu().numpy()

        actions_dict = {agent: int(actions[i]) for i, agent in enumerate(env.get_agents())}

        # Critic forward
        V = model.actor_encoder_critic.forward(
            obs_tensor, 
            belief_vector, 
            com_vector,
            wh["critic"]
        )
        values = V.squeeze(-1)
        
        # Environment step
        next_obs, rewards, dones, _ = env.step(actions_dict)
        rewards = np.array([rewards[agent] for agent in env.get_agents()])

        obs = next_obs
        done = np.any(dones.values()) or i == params.experience_sampling_steps - 1

        if done:
            obs = env.reset()

        next_obs_tensor = torch.FloatTensor(np.stack([obs[agent] for agent in env.get_agents()])).to(device)

        # Communication processing
        source_indices = torch.arange(0, env.n_agents, dtype=torch.long)
        neighbor_indices, Mij, reverses = env.sample_neighbors()
        Mij = Mij.to(model.device)
        reverses = reverses.to(model.device)
        messages = model.filter.forward(z, Mij, wh["filter"])
        
        # Decoder update
        zdj, Mji = model.decoder_update.forward(messages, reverses,  wh["decoder"], wh["update_mean"], wh["update_std"])
        
        # Update environment states
        env.set_beliefs(h)
        env.set_comm_state(neighbor_indices, zdj)
        env.update_edges(source_indices, neighbor_indices, Mji)

        # Store experience for sampled agents
        batch_obs.append(obs_tensor[indices])
        batch_next_obs.append(next_obs_tensor[indices])
        batch_actions.append(actions[indices])
        batch_rewards.append(rewards[indices])
        batch_logits.append(Q[indices])

        batch_lv.append(lv[indices])
        batch_wh.append(select_weights(wh, indices))
        batch_values.append(values[indices])
        batch_next_belief.append(h[indices])
        batch_next_com.append(z[indices])

        batch_belief.append(belief_vector[indices])
        batch_trait.append(trait_vector[indices])
        batch_com.append(com_vector[indices])

        batch_ld_means.append(mean[indices])
        batch_ld_std.append(std[indices])
        batch_dones.append(torch.tensor(done, dtype = torch.bool))

        batch_Mij.append(Mij[indices])
        batch_Mji.append(Mji[indices])

        batch_ze.append(z[indices])
        batch_zd.append(zdj[indices])

        batch_messages.append(messages[indices])
    
    # Convert to tensors
    experiences = {
        'observations': torch.stack(batch_obs),
        'actions': torch.tensor(np.array(batch_actions), device=device),
        'rewards': torch.tensor(np.array(batch_rewards), dtype=torch.float32, device=device),
        'logits': torch.stack(batch_logits),
        'lv': torch.stack(batch_lv),
        'wh': torch.stack(batch_wh),
        "values" : torch.stack(batch_values),
        'belief': torch.stack(batch_belief),
        'traits': torch.stack(batch_trait), 
        'com': torch.stack(batch_com),
        "means": torch.stack(batch_ld_means),
        "std": torch.stack(batch_ld_std),
        "done": torch.stack(batch_dones),
        "next_observations": torch.stack(batch_next_obs),
        "next_com" : torch.stack(batch_next_com),
        "next_belief" : torch.stack(batch_next_belief),

        "M_ij" : torch.stack(batch_Mij),
        "M_ji" : torch.stack(batch_Mji),

        'z_e' : torch.stack(batch_ze),
        'z_d' : torch.stack(batch_zd),
        'messages' : torch.stack(batch_messages)
    }
    
    return TensorDict(experiences, batch_size=params.experience_sampling_steps)

def train_sac_model(model: SACModel, env: BaseEnv, params: TrainingParameters):
    if params.verbose:
        writer = SummaryWriter()
    else:
        writer = None 

    optim = torch.optim.Adam([
        {'params': model.actor_encoder.parameters(), 'lr': params.actor_learning_rate, 'eps' : 1e-5},
        {'params': model.q1.parameters(), 'lr': params.critic_learning_rate, 'eps' : 1e-5},
        {'params': model.hypernet.parameters(), 'lr': params.hypernet_learning_rate, 'eps' : 1e-5},
        {'params': model.filter.parameters(), 'lr': params.filter_learning_rate, 'eps' : 1e-5}, 
        {'params': model.decoder_update.parameters(), 'lr': params.decoder_learning_rate, 'eps' : 1e-5},
        {'params': model.q2.parameters(), 'lr': params.critic_learning_rate, 'eps' : 1e-5},
        {'params': [model.log_alpha], 'lr': params.actor_learning_rate, 'eps' : 1e-5},
    ])  

    params.global_steps = 0
    experiences = TensorDict({})  # Initialize empty experience buffer

    # Initialize the model
    model.set_alpha(params.alpha)
    model.to(params.device)

    for i in tqdm(range(params.outer_loops)):
        model.train()
        model.requires_grad_(True)
        
        # Collect new experiences and explicitly detach+clone
        new_exp = collect_experiences(model, env, params, i)
        
        # Create detached clone of all tensors in the experience
        detached_exp = TensorDict({
            k: v.detach().clone() for k, v in new_exp.items()
        }, batch_size=[params.experience_sampling_steps])
        
        # Append to buffer
        if len(experiences) == 0:
            experiences = detached_exp
        else:
            experiences = torch.cat([experiences, detached_exp], dim=0)
        
        # Trim buffer while maintaining computational graph isolation
        if len(experiences) > params.experience_buffer_size:
            keep_from = len(experiences) - params.experience_buffer_size
            experiences = TensorDict(
                {k: v[keep_from:] for k, v in experiences.items()},
                batch_size=[params.experience_buffer_size]
            )
        
        total_loss = torch.tensor(0.0, requires_grad = True)

        if params.should_train_actor:
            total_loss = total_loss + train_sac_actor(model, env, experiences, params, writer=writer)
        
        if params.should_train_hypernet:
            total_loss = total_loss + train_hypernet(model, env, experiences, params, writer=writer)

        if params.should_train_gnn:
            total_loss = total_loss + train_gnn(model, env, experiences, params, writer= writer)
        
        if params.should_train_filter:
            total_loss = total_loss + train_filter(model, env, experiences, params, writer = writer)
            
        if writer is not None:
            writer.add_scalar('State/Epsilon', params.epsilon, global_step = params.global_steps)

        optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip_norm)
        optim.step()
        
        # Update target networks
        with torch.no_grad():
            for t_param, param in zip(model.target_q1.parameters(), model.q1.parameters()):
                t_param.data.mul_(1 - params.tau).add_(params.tau * param.data)
            for t_param, param in zip(model.target_q2.parameters(), model.q2.parameters()):
                t_param.data.mul_(1 - params.tau).add_(params.tau * param.data)


        model.requires_grad_(False)
        model.train(False)
        evaluate_policy(model, env, writer=writer, global_step=params.global_steps, temperature=params.eval_temp, k = params.eval_k)
        params.global_steps += 1

    writer.close()

def train_ppo_model(model: PPOModel, env: BaseEnv, params: TrainingParameters):
    if params.verbose:
        writer = SummaryWriter()
    else:
        writer = None 

    optim = torch.optim.Adam([
        {'params': model.actor_encoder.parameters(), 'lr': params.actor_learning_rate, 'eps' : 1e-5},
        {'params': model.actor_encoder_critic.parameters(), 'lr': params.critic_learning_rate, 'eps' : 1e-5},
        {'params': model.hypernet.parameters(), 'lr': params.hypernet_learning_rate, 'eps' : 1e-5},
        {'params': model.filter.parameters(), 'lr': params.filter_learning_rate, 'eps' : 1e-5}, 
        {'params': model.decoder_update.parameters(), 'lr': params.decoder_learning_rate, 'eps' : 1e-5},
    ])  

    params.global_steps = 0
    experiences = TensorDict({})  # Initialize empty experience buffer

    # Initialize the model
    model.to(params.device)

    for i in tqdm(range(params.outer_loops)):
        model.train()
        model.requires_grad_(True)
        
        # Collect new experiences and explicitly detach+clone
        new_exp = collect_experiences(model, env, params, i)
        
        # Create detached clone of all tensors in the experience
        detached_exp = TensorDict({
            k: v.detach().clone() for k, v in new_exp.items()
        }, batch_size=[params.experience_sampling_steps])
        
        # Append to buffer
        if len(experiences) == 0:
            experiences = detached_exp
        else:
            experiences = torch.cat([experiences, detached_exp], dim=0)
        
        # Trim buffer while maintaining computational graph isolation
        if len(experiences) > params.experience_buffer_size:
            keep_from = len(experiences) - params.experience_buffer_size
            experiences = TensorDict(
                {k: v[keep_from:] for k, v in experiences.items()},
                batch_size=[params.experience_buffer_size]
            )
        
        total_loss = torch.tensor(0.0, requires_grad = True)

        if params.should_train_actor:
            total_loss = total_loss + train_ppo_actor(model, env, experiences, params, writer=writer)
        
        if params.should_train_hypernet:
            total_loss = total_loss + train_hypernet(model, env, experiences, params, writer=writer)
        
        if params.should_train_gnn:
            total_loss = total_loss + train_gnn(model, env, experiences, params, writer= writer)
            
        if writer is not None:
            writer.add_scalar('State/Epsilon', params.epsilon, global_step = params.global_steps)

        optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip_norm)
        optim.step()

        model.train(False)
        model.requires_grad_(False)
        evaluate_policy(model, env, writer=writer, global_step=params.global_steps, temperature=params.eval_temp, k = params.eval_k)
        params.global_steps += 1

    if writer != None: 
        writer.close()


def train_model(model: SACModel | PPOModel, env: BaseEnv, params: TrainingParameters):
    if type(model) is SACModel:
        train_sac_model(model, env, params)
    else: 
        train_ppo_model(model, env, params)

