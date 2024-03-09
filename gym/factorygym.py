import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pygame

from environment.world import World
from environment.constants import BOUNDS, BLOCK_SIZE

from environment.resource import TOTAL_RESOURCE_TYPES
from environment.components import TOTAL_COMPONENT_TYPES

class FactoryGym(gym.Env):
    def __init__(self):
        super(FactoryGym, self).__init__()
        
        self.DISPLAY_WIDTH = 800
        self.DISPLAY_HEIGHT = 600

        self.WORLD_WIDTH = BOUNDS.x 
        self.WORLD_HEIGHT = BOUNDS.y

        """
        The state space is specified as consisting of the foollowing:
        All masks have a shape of (width, height, x) where x varies depending on the mask type

        1. World Mask - determines whether or not a tile is empty.
        x = 1 

        2. Resource MAsk - determines the resource type per tile and any linkages in the four cardinal
        directions
        x = 1 + 4 = 5

        3. Factory Mask - determines the component type per tile, its rotation, and an additional param for 
        spawners that determine the resource type it spawns. Components where rotation doesn't matter are given 0 rotation
        x = 1 + 1 + 1 = 3 
        """
        self.observation_space = spaces.Dict({
            "world_mask": 
                spaces.Box(low = 0, high = 1, shape = (self.WORLD_WIDTH, self.WORLD_HEIGHT, 1), dtype=np.int8),
            "resource_mask": spaces.Tuple([
                spaces.Box(low = 0, high = TOTAL_RESOURCE_TYPES, shape = (self.WORLD_WIDTH, self.WORLD_HEIGHT, 1), dtype = np.int8), 
                spaces.Box(low = 0, high = 1, shape = (self.WORLD_WIDTH, self.WORLD_HEIGHT, 4), dtype = np.int8)
            ]),
            "factory_mask": spaces.Tuple([
                spaces.Box(low = 0, high = TOTAL_COMPONENT_TYPES, shape = (self.WORLD_WIDTH, self.WORLD_HEIGHT, 1), dtype = np.int8), 
                spaces.Box(low = 0, high = 4, shape = (self.WORLD_WIDTH, self.WORLD_HEIGHT, 1), dtype = np.int8),
                spaces.Box(low = 0, high = TOTAL_RESOURCE_TYPES, shape = (self.WORLD_WIDTH, self.WORLD_HEIGHT, 1), dtype = np.int8)
            ]),
        })


        self.action_space = spaces.Discrete(6)
        
        self.running = True 
        
        pygame.init()
        pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
        self.world = World(self.WORLD_WIDTH, self.WORLD_HEIGHT, BLOCK_SIZE)

        self.reset()

    def reset(self):
        # Reset the environment to its initial state
        self.state = self.world.get_state()
        return self.state

    def step(self, action):
        # TODO: Process the action here
        self.state = self.world.get_state()
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
