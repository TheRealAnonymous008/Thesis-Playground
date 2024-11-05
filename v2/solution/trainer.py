from __future__ import annotations
import copy
import os
import glob

from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy, MultiInputPolicy 
import supersuit as ss

from .custom_gym import CustomGymEnviornment
import time

from tqdm import tqdm

from models.base import BaseModel
import numpy as np

def train_loop(
        env: CustomGymEnviornment, 
        model: BaseModel,  
        games: int = 100, 
        optimization_passes: int = 10,
        seed: int = 0):
    """
    Train an agent for run_count number of games (no. of iters per game is dictated by env)
    """
    print(f"Training on {str(env.metadata['name'])}.")
    steps = games * env._max_time_steps
    steps_per_checkpt = 100
    checkpts = int(steps / steps_per_checkpt)
    avg_rewards = []

    # Wrap the training loop in tqdm for progress tracking
    for i in tqdm(range(checkpts), desc="Training Progress"):
        model.learn(total_timesteps=steps_per_checkpt, optimization_passes=optimization_passes)
        # TODO: Uncomment this 
        # model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")
        print("Model has been saved.")

        avg_rewards.append(test_agents(env, model, 1))

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")
    
    env.close()

    return avg_rewards


def test_agents(env : CustomGymEnviornment, model :BaseModel, games : int = 100,  seed : int = 0,  load_from_disk : bool = False):
    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={games})"
    )

    if load_from_disk: 
        try:
            latest_policy = max(
                glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
            )
        except ValueError:
            print("Policy not found.")

        model.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    for i in range(games):
        obs, info = env.reset(seed=i)
        for agent in env.agents:
            rewards[agent]== 0
        
        for _ in range(env._max_time_steps): 
            obs = model.feature_extractor(obs)
            action = model.select_joint_action(obs, deterministic=True)
            obs, reward, termination, truncation, info = env.step(action)
            for agent in env.agents:
                rewards[agent] += reward[agent]

    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}  std: {np.std(rewards.values())}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    # print("Full rewards: ", rewards)
    return avg_reward