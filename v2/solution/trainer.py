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
import torch 

def set_seed(seed: int | None):
    """
    Set random seed for reproducibility across numpy and torch if seed is non-negative.
    """
    if seed != None :
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using CUDA, set all CUDA seeds

def train_loop(
        env: CustomGymEnviornment, 
        model: BaseModel,  
        games: int = 100, 
        optimization_passes: int = 10,
        seed: int = None):
    """
    Train an agent for run_count number of games (no. of iters per game is dictated by env)
    """
    print(f"Training on {str(env.metadata['name'])}.")
    steps = games * env._max_time_steps
    steps_per_checkpt = 100
    checkpts = int(steps / steps_per_checkpt)
    avg_rewards = []

    set_seed(seed)
    model.env.reset(seed= seed)

    # Wrap the training loop in tqdm for progress tracking
    for i in tqdm(range(checkpts), desc="Training Progress"):
        model.learn(total_timesteps=steps_per_checkpt, optimization_passes=optimization_passes)
        # TODO: Uncomment this 
        print("Model has been saved.")

        avg_rewards.append(test_agents(model.env, model, 1))

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")
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

    model._model.eval()
    rewards = {agent: 0 for agent in env.possible_agents}

    for i in range(games):
        obs, info = env.reset(seed=i)
        for agent in env.agents:
            rewards[agent]== 0
        
        for _ in range(env._max_time_steps): 
            obs = model.feature_extractor(obs, model.device)
            action = model.select_joint_action(obs, deterministic=True)
            obs, reward, termination, truncation, info = env.step(action)
            for agent in env.agents:
                rewards[agent] += reward[agent]

    env.close()

    avg_reward_per_agent = {
        agent: rewards[agent] / games for agent in env.possible_agents
    }

    rewards = [reward for reward in rewards.values()]

    mean = np.mean(rewards)
    std = np.std(rewards)
    coeff_variation = std / mean 

    print(f"Avg reward: {mean}  std: {std}  coeff : {coeff_variation}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    # print("Full rewards: ", rewards)
    return mean