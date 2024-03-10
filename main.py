from factory_gym.factorygym import FactoryGym
from factory_gym.factorysim import FactorySimulation
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


if __name__ == "__main__":
    env = FactoryGym()
    
    # Initialize the environment and get the initial state
    state = env.reset()
    max_iter = 1000

    env = FactoryGym()
    # print(check_env(env))

    # Initialize the PPO model
    model = PPO("MultiInputPolicy", env, verbose=1)

    # TODO: Save the model :) 

    # Train the model
    model.learn(total_timesteps=10000)

    model.save("test.zip")
    
    # Run the environment
    state, info = env.reset()
    for _ in range(0, max_iter):
        env.render()
        action, _states = model.predict(state, deterministic=True)
        state, reward, done, truncated, info = env.step(action)

        # Check if the episode is done
        if done:
            print("Episode finished.")
            break

    # Close the environment
    env.close()

