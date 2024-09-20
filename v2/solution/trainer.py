from __future__ import annotations
import copy
import os
import glob

from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy, MultiInputPolicy 
import supersuit as ss

from .custom_gym import CustomGymEnviornment
import time

from ..models.base import BaseModel

def train_loop(_env : CustomGymEnviornment, model : BaseModel, games : int = 100, seed : int = 0):
    """
    Train an agent for run_count number of games (no. of iters per game is dictated by env)
    """
    env = copy.deepcopy(_env)
    print(f"Training on {str(env.metadata['name'])}.")
    env.reset(seed=seed)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 20, num_cpus=4, base_class="stable_baselines3")
    env.reset()
    steps = games * _env._max_time_steps
    steps_per_checkpt = 100
    checkpts = int(steps / steps_per_checkpt)

    for i in range(checkpts):
        model = model.learn(total_timesteps=steps_per_checkpt)
        model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")
        print("Model has been saved.")

        # test_agents(_env, 1)

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def test_agents(env : CustomGymEnviornment, model : BaseModel, games : int = 100, seed : int = 0):
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

    model.load(latest_policy)

    env.reset()
    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
    for i in range(games):
        obs, info = env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)
        for agent in env.agents:
            rewards[agent]== 0
        
        for _ in range(env._max_time_steps): 
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