import numpy as np 
import torch
from models.model import Model
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

def evaluate_policy(model: Model, env, num_episodes=10, k = 10, writer : SummaryWriter =None, global_step=None, temperature = 0.9):
    """Evaluate current policy and return average episode return with trait cluster breakdown"""
    total_returns = []
    actions_array = []
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

            while not done:
                obs_array = np.stack([obs[agent] for agent in env.agents])
                obs_tensor = torch.FloatTensor(obs_array).to(device)

                # Generate hypernet weights
                belief_vector = torch.tensor(env.get_beliefs(), device = device)
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
                if temperature > 0:
                    actions = Q.argmax(dim=-1).cpu().numpy()
                else : 
                    dist = Categorical(logits = Q / temperature)
                    actions = dist.sample().cpu().numpy()
                
                actions_array.append(actions)

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

            # Store traits and rewards for this episode's agents
            for i, agent in enumerate(env.agents):
                all_traits.append(traits_np[i])
                all_rewards.append(agent_returns[agent])

            total_returns.append(episode_return)

    # Process clustering and print breakdown
    if len(all_traits) > 0:
        data = np.array(all_traits)
        rewards = np.array(all_rewards)
        labels, _ = kmeans(data, k)
        if len(labels) > 0:
            cluster_rewards = {}
            for label, reward in zip(labels, rewards):
                if label not in cluster_rewards:
                    cluster_rewards[label] = []
                cluster_rewards[label].append(reward)
            for cluster in sorted(cluster_rewards.keys()):
                avg_return = np.median(cluster_rewards[cluster])
                header = f"Eval/cluster_{cluster}"
                if writer is not None:
                    writer.add_scalar(f'{header}/median_return', avg_return, global_step)

                
        else:
            print("\nNo clusters formed due to insufficient data.")
    else:
        print("\nNo agent traits collected.")

    # Original outputs
    median_returns = np.median(total_returns)
    actions_flat = np.concatenate(actions_array, dtype = np.int16) if actions_array else np.array([])
    
    rewards_dist = np.array(all_rewards)

     # Log overall metrics
    if writer is not None:
        writer.add_scalar(f'Eval/median_rewards', median_returns, global_step)
        writer.add_histogram(f'Eval/action_distribution', torch.tensor(actions_flat), global_step)

        writer.add_histogram('Eval/agent_total_rewards', torch.tensor(rewards_dist), global_step)
    return median_returns