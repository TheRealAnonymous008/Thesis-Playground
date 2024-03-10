from factory_gym.factorygym import FactoryGym
from factory_gym.factorysim import FactorySimulation
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


if __name__ == "__main__":
    env = FactoryGym()
    print(check_env(env))
    env.close()

