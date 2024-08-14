from __future__ import annotations
import os
import glob

from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy, MultiInputPolicy 
import supersuit as ss

from manenv.gym_wrapper import MARLFactoryEnvironment
import time

def train_loop(env : MARLFactoryEnvironment, games : int = 100, seed : int = 0):
    """
    Train an agent for run_count number of games (no. of iters per game is dictated by env)
    """
    print(f"Starting training on {str(env.metadata['name'])}.")
    env.reset(seed=seed)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")
    env.reset()
    
    model = PPO(
        MultiInputPolicy,
        env,
        verbose=3,
        batch_size=256,
    )
    steps = games * MARLFactoryEnvironment.MAX_GAME_STEPS

    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()