from collections import defaultdict
import numpy as np 
import torch
from models.model import SACModel
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

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

def kmeans(data, k=3, max_iters=100):
    n_samples = data.shape[0]
    if n_samples == 0 or k == 0:
        return np.array([]), np.array([])
    # Adjust k if there are fewer samples than k
    k = min(k, n_samples)
    centroids = data[np.random.choice(n_samples, k, replace=False)]
    for _ in range(max_iters):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = []
        for i in range(k):
            cluster_data = data[labels == i]
            if len(cluster_data) == 0:
                # Re-initialize empty cluster centroid randomly
                new_centroid = data[np.random.choice(n_samples, 1)[0]]
            else:
                new_centroid = cluster_data.mean(axis=0)
            new_centroids.append(new_centroid)
        new_centroids = np.array(new_centroids)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids
def evaluate_policy(model: SACModel, env, num_episodes=10, k=2, writer: SummaryWriter = None, global_step=None, temperature=-1):
    """Evaluate current policy and return average episode return with trait cluster breakdown, including action distributions per cluster."""
    total_returns = []
    episode_actions = []  # List to store actions per episode
    agents_per_episode = []  # Track number of agents per episode
    all_traits = []
    all_rewards = []
    device = model.config.device

    with torch.no_grad():
        for _ in range(num_episodes):
            obs = env.reset()
            episode_return = 0
            done = False
            agent_returns = {agent: 0.0 for agent in env.agents}
            trait_vector = torch.tensor(env.get_traits(), device=device)
            traits_np = trait_vector.cpu().numpy()
            agents_in_episode = len(env.agents)
            agents_per_episode.append(agents_in_episode)
            current_episode_actions = []

            while not done:
                obs_array = np.stack([obs[agent] for agent in env.agents])
                obs_tensor = torch.FloatTensor(obs_array).to(device)

                # Generate hypernet weights
                belief_vector = torch.tensor(env.get_beliefs(), device=device)
                com_vector = torch.zeros((model.config.n_agents, model.config.d_comm_state), device=device)
                lv, wh, _, _ = model.hypernet(trait_vector, belief_vector)

                # Get action distribution
                Q, _, _ = model.actor_encoder.forward(
                    obs_tensor, 
                    belief_vector, 
                    com_vector, 
                    wh["policy"], 
                    wh["belief"], 
                    wh["encoder"]
                )
                if temperature < 0:
                    actions = Q.argmax(dim=-1).cpu().numpy()
                else:
                    dist = Categorical(logits=Q / temperature)
                    actions = dist.sample().cpu().numpy()
                
                current_episode_actions.append(actions)

                # Step environment
                next_obs, rewards, dones, _ = env.step(
                    {agent: int(actions[i]) for i, agent in enumerate(env.agents)}
                )
                episode_return += sum(rewards.values()) / len(env.agents)
                done = any(dones.values())
                obs = next_obs

                # Accumulate individual agent rewards
                for agent in env.agents:
                    agent_returns[agent] += rewards[agent]

            # Store episode actions and agent data
            episode_actions.append(current_episode_actions)
            for i, agent in enumerate(env.agents):
                all_traits.append(traits_np[i])
                all_rewards.append(agent_returns[agent])

            total_returns.append(episode_return)

    # Process clustering and print breakdown
    median_returns = np.median(total_returns)
    actions_flat = np.concatenate([np.concatenate(ep_acts) for ep_acts in episode_actions], dtype=np.int16) if episode_actions else np.array([])
    rewards_dist = np.array(all_rewards)

    if len(all_traits) > 0:
        data = np.array(all_traits)
        rewards = np.array(all_rewards)
        labels, _ = kmeans(data, k)
        if len(labels) > 0:
            cluster_rewards = {}
            cluster_actions = defaultdict(list)
            # Compute episode indices for agent mapping
            current = 0
            episode_indices = []
            for count in agents_per_episode:
                episode_indices.append((current, current + count))
                current += count

            # Map each agent to cluster and collect actions
            for i, (label, reward) in enumerate(zip(labels, rewards)):
                # Update cluster rewards
                if label not in cluster_rewards:
                    cluster_rewards[label] = []
                cluster_rewards[label].append(reward)

                # Find episode and position for current agent
                for e, (start, end) in enumerate(episode_indices):
                    if start <= i < end:
                        break
                else:
                    continue  # Skip if not found (shouldn't happen)
                j = i - start  # Agent's index in the episode

                # Collect actions for the agent
                if e < len(episode_actions):
                    for t in range(len(episode_actions[e])):
                        action = episode_actions[e][t][j]
                        cluster_actions[label].append(action)

            # Log metrics per cluster
            for cluster in sorted(cluster_rewards.keys()):
                avg_return = np.median(cluster_rewards[cluster])
                header = f"Eval/cluster_{cluster}"
                if writer is not None:
                    writer.add_scalar(f'{header}/median_return', avg_return, global_step)
                    # Log action distribution if available
                    if cluster in cluster_actions and len(cluster_actions[cluster]) > 0:
                        actions_tensor = torch.tensor(cluster_actions[cluster], dtype=torch.int32)
                        writer.add_histogram(f'{header}/action_distribution', actions_tensor, global_step)
        else:
            print("\nNo clusters formed due to insufficient data.")
    else:
        print("\nNo agent traits collected.")

    # Log overall metrics
    if writer is not None:
        writer.add_scalar(f'Eval/median_rewards', median_returns, global_step)
        if actions_flat.size > 0:
            writer.add_histogram(f'Eval/action_distribution', torch.tensor(actions_flat), global_step)
        writer.add_histogram('Eval/agent_total_rewards', torch.tensor(rewards_dist), global_step)
    return median_returns