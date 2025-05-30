from __future__ import annotations
import copy
import os
import glob

from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy, MultiInputPolicy 
import supersuit as ss

from models.custom_gym import CustomGymEnviornment
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
        games_per_checkpoint: int = 50, 
        checkpoints : int = 10,
        optimization_passes: int = 10,
        difficulty_rate : int = 4,
        verbose : bool = False,
        seed: int = None):
    """
    Train an agent for run_count number of games (no. of iters per game is dictated by env)
    """
    print(f"Training on {str(env.metadata['name'])}.")
    avg_rewards = []

    set_seed(seed)
    model.env.reset(seed= seed)
    model.reset_difficulty()
    
    time_steps = 0

    # Wrap the training loop in tqdm for progress tracking
    for i in tqdm(range(checkpoints), desc="Training Progress"):
        model.learn(total_timesteps=games_per_checkpoint * env._max_time_steps, optimization_passes=optimization_passes, verbose = verbose)
        model.update_difficulty()
        time_steps += 1

        # TODO: Uncomment this 
        model.save(f"{env.unwrapped.metadata.get('name')}_{i}_{time.strftime('%Y%m%d-%H%M%S')}.zip")
        if verbose: 
            print("Model has been saved.")

        avg_rewards.append(test_agents(model.env, model, 1, verbose = verbose))
    
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")
    
    env.close()

    return avg_rewards


def test_agents(env : CustomGymEnviornment, model :BaseModel, games : int = 100,  seed : int = 0,  load_from_disk : bool = False, verbose : bool = True ):
    if verbose: 
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
    # While it is called reward -- we can actually have it be anything we want (for example, the actual goal we are optimizing for )
    rewards = {agent: 0 for agent in env.possible_agents}

    for i in range(games):
        obs, info = env.reset(seed=i)
        for agent in env.agents:
            rewards[agent] = 0
        
        for _ in range(env._max_time_steps): 
            _, _, reward, obs, _, _ = model.step(obs)
            for agent in env.agents:
                r = env._world.get_agent(agent)._current_state.just_rescued_victim / env._world.get_param("total_victims") * 100 
                rewards[agent] += r

    env.close()

    avg_reward_per_agent = {
        agent: rewards[agent] / games for agent in env.possible_agents
    }

    rewards = [reward for reward in rewards.values()]

    mean = np.mean(rewards)
    std = np.std(rewards)
    coeff_variation = std / mean 

    if verbose: 
        print(f"Avg reward: {mean}  std: {std}  coeff : {coeff_variation}")
        print("Avg reward per agent, per game: ", avg_reward_per_agent)
    # print("Full rewards: ", rewards)
    return mean