from factory_gym.factorygym import FactoryGym
from factory_gym.factorysim import FactorySimulation
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


if __name__ == "__main__":
    env = FactoryGym()
    # Run the environment
    state, info = env.reset(42)
    total_reward = 0
    iters = 100
    for _ in range(iters):
        action = env.action_space.sample()

        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        # Check if the episode is done
        if done:
            print("Episode finished. Reward = ", total_reward)
            total_reward = 0
            state, info = env.reset(42)
            break

    # Close the environment
    env.close()
