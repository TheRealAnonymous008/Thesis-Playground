import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pygame

from environment.world import World
from environment.constants import BOUNDS, BLOCK_SIZE

class FactoryGym(gym.Env):
    def __init__(self):
        super(FactoryGym, self).__init__()

        # Define observation and action space 
        self.observation_space = spaces.Box(low=0, high=255, shape=(600, 800, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(6)

        self.DISPLAY_WIDTH = 800
        self.DISPLAY_HEIGHT = 600
        
        self.running = True 
        
        pygame.init()
        pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
        self.world = World(BOUNDS.x, BOUNDS.y, BLOCK_SIZE)

        self.reset()

    def reset(self):
        # Reset the environment to its initial state
        self.state = np.zeros((600, 800, 3))
        return self.state

    def step(self, action):
        self.state += action
        reward = 0 
        done = False 

        self.world.update()

        info = {} 
        return self.state, reward, done, info

    def render(self, mode='human'):
        # Render the environment to the screen
        if mode == 'human':
            self.world.draw(pygame.display.get_surface())
            pygame.display.flip()
        

    def close(self):
        # Clean up the environment
        pygame.quit()
