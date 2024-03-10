from gym.factorygym import FactoryGym
from gym.factorysim import FactorySimulation
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


if __name__ == "__main__":
    env = FactoryGym()
    
    # Initialize the environment and get the initial state
    state = env.reset()
    max_iter = 100

    env = FactoryGym()
    print(check_env(env))

    # Initialize the PPO model
    model = PPO("MultiInputPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=10000)

    # Save the trained model
    model.save("ppo_factory_gym")

    # Load the trained model
    model = PPO.load("ppo_factory_gym")
    
    # Run the environment
    state = env.reset()
    for _ in range(0, max_iter):
        action, _states = model.predict(state, deterministic=True)
        state, reward, done, info = env.step(action)
        env.render()

        # Check if the episode is done
        if done:
            print("Episode finished.")
            break

    # Close the environment
    env.close()

