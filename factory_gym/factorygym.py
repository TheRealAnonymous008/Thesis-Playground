import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pygame
from enum import Enum

from environment.world import World
from environment.constants import BOUNDS, BLOCK_SIZE

from environment.resource import TOTAL_RESOURCE_TYPES
from environment.components import TOTAL_COMPONENT_TYPES
from environment.direction import Direction

class ActionEnum(Enum):
    IDLE = 0 
    MOVE_NORTH = 1
    MOVE_SOUTH = 2 
    MOVE_EAST = 3 
    MOVE_WEST = 4
    ROTATE_CW = 5
    ROTATE_CCW = 6 
    SWITCH_MODE = 7 

TOTAL_AGENT_ACTIONS = len(ActionEnum) + 1
MAX_ITERS = 320

class FactoryGym(gym.Env):
    def __init__(self):
        super(FactoryGym, self).__init__()
        
        self.DISPLAY_WIDTH = 800
        self.DISPLAY_HEIGHT = 600

        self.WORLD_WIDTH = BOUNDS.x 
        self.WORLD_HEIGHT = BOUNDS.y

        self.n_agents = 1

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

        4. Assembler Mask - determines information about assemmblers, their rotation and the current mode they are operating on (PUSH / PULL)
        x = 1 + 1 + 1 = 3
        """
        self.world = World(self.WORLD_WIDTH, self.WORLD_HEIGHT, BLOCK_SIZE)
        low = np.array([
            # World Mask
            0, 
            # Resource Mask 
            0, 
            0, 0, 0, 0, 
            # Factory Mask 
            0, 
            0, 
            0, 
            # Assembler Mask 
            0, 
            0, 
            0,
        ], dtype = np.int8)
        low = np.repeat(low.reshape(1, -1), self.WORLD_HEIGHT, axis=0)
        low = np.repeat(low.reshape(1, self.WORLD_HEIGHT,  -1), self.WORLD_WIDTH, axis=0)

        high = np.array([
            # World Mask
            1, 
            # Resource Mask 
            TOTAL_RESOURCE_TYPES, 
            1, 1, 1, 1, 
            # Factory Mask 
            TOTAL_COMPONENT_TYPES, 
            4, 
            TOTAL_RESOURCE_TYPES, 
            # Assembler Mask 
            1, 
            4, 
            1,
        ], dtype = np.int8)
        high = np.repeat(high.reshape(1, -1), self.WORLD_HEIGHT, axis=0)
        high = np.repeat(high.reshape(1, self.WORLD_HEIGHT,  -1), self.WORLD_WIDTH, axis=0)

        self.observation_space = spaces.Box(low, high, dtype = np.int8) 

        """
        The action space is specified as a tuple of discrete spaces per agent 
        """
        self.n_agents = len(self.world.factory.assembler_list)
        self.action_space = spaces.MultiDiscrete([TOTAL_AGENT_ACTIONS for _ in range(0, self.n_agents)])
        
        self.running = True 
        
        pygame.init()
        pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))

        self.iters = 0

    def reset(self, seed = 0):
        # Reset the environment to its initial state
        self.world.init()
        self.state = self.world.get_state()
        self.iters = 0

        info = {}
        return self.state, info

    def step(self, actions : ActionEnum):
        # Each action is assumed (for now) to be a single agent's action
        self.iters += 1
        reward = 0

        for idx in range(0, self.n_agents): 
            action = actions[idx]
            assembler = self.world.factory.assembler_list[idx]

            match(action):
                case ActionEnum.IDLE.value: 
                    pass
                case ActionEnum.MOVE_NORTH.value:
                    assembler.move_direction(self.world, Direction.NORTH)
                case ActionEnum.MOVE_SOUTH.value:
                    assembler.move_direction(self.world, Direction.SOUTH)
                case ActionEnum.MOVE_EAST.value:
                    assembler.move_direction(self.world, Direction.EAST)
                case ActionEnum.MOVE_WEST.value:
                    assembler.move_direction(self.world, Direction.WEST)
                case ActionEnum.ROTATE_CW.value:
                    assembler.rotate_cw()  
                case ActionEnum.ROTATE_CCW.value: 
                    assembler.rotate_ccw() 
                case ActionEnum.SWITCH_MODE.value: 
                    assembler.switch_mode()  

        self.world.update()
        reward += self.world.global_reward 
        self.state = self.world.get_state()
        done = True if self.iters >= MAX_ITERS else False 
        info = {} 
        return self.state, reward, done, done, info

    def render(self, mode='human'):
        # Render the environment to the screen
        if mode == 'human':
            self.world.draw(pygame.display.get_surface())
            pygame.display.flip()
        

    def close(self):
        # Clean up the environment
        pygame.quit()

