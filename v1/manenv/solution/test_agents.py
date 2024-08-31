from __future__ import annotations
import os
import glob

from stable_baselines3 import PPO
import supersuit as ss

from manenv.gym_wrapper import MARLFactoryEnvironment

def test_agents(env : MARLFactoryEnvironment, games : int = 100, seed : int = 0):
    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={games})"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    env.reset()
    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
    for i in range(games):
        obs, info = env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)
        for agent in env.agents:
            rewards[agent]== 0
        
        for _ in range(MARLFactoryEnvironment.MAX_GAME_STEPS): 
            action = {}
            for agent in env.agents:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample()
                else:
                    act = model.predict(obs[agent], deterministic=True)[0]
                action[agent] = act
            obs, reward, termination, truncation, info = env.step(action)
            for agent in env.agents:
                rewards[agent] += reward[agent]

    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    return avg_reward