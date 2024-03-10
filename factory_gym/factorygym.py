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

        4. Assembler Mask - determines information about assemmblers, their rotation and the current mode they are operating on (PUSH / PULL)
        x = 1 + 1 + 1 = 3
        """
        self.observation_space = spaces.Dict({
            "world_mask": spaces.Box(low = 0, high = 1, shape = (self.WORLD_WIDTH, self.WORLD_HEIGHT), dtype=np.int8),
            "resource_mask_type":  spaces.Box(low = 0, high = TOTAL_RESOURCE_TYPES, shape = (self.WORLD_WIDTH, self.WORLD_HEIGHT), dtype = np.int8), 
            "resource_link_mask": spaces.Box(low = 0, high = 1, shape = (self.WORLD_WIDTH, self.WORLD_HEIGHT, 4), dtype = np.int8),
            "factory_mask_type": spaces.Box(low = 0, high = TOTAL_COMPONENT_TYPES, shape = (self.WORLD_WIDTH, self.WORLD_HEIGHT), dtype = np.int8), 
            "factory_mask_direction": spaces.Box(low = 0, high = 4, shape = (self.WORLD_WIDTH, self.WORLD_HEIGHT), dtype = np.int8),
            "factory_mask_resource_type": spaces.Box(low = 0, high = TOTAL_RESOURCE_TYPES, shape = (self.WORLD_WIDTH, self.WORLD_HEIGHT), dtype = np.int8),
            "assembler_mask_is_present": spaces.Box(low = 0, high = 1, shape = (self.WORLD_WIDTH, self.WORLD_HEIGHT), dtype = np.int8), 
            "assembler_mask_direction": spaces.Box(low = 0, high = 4, shape = (self.WORLD_WIDTH, self.WORLD_HEIGHT), dtype = np.int8),
            "assembler_mask_mode": spaces.Box(low = 0, high = 1, shape = (self.WORLD_WIDTH, self.WORLD_HEIGHT), dtype = np.int8)
        })
        """
        The action space is specified as a discrete space

        """
        self.action_space = spaces.Box(low = 0, high = TOTAL_AGENT_ACTIONS, dtype= np.int8)
        
        self.running = True 
        
        pygame.init()
        pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
        self.world = World(self.WORLD_WIDTH, self.WORLD_HEIGHT, BLOCK_SIZE)

        self.assembler = self.world.factory.assemblers[3][4]

    def reset(self, seed):
        # Reset the environment to its initial state
        self.world.init()
        self.state = self.world.get_state()
        self.assembler = self.world.factory.assemblers[3][4]

        info = {}
        return self.state, info

    def step(self, action : ActionEnum):
        # Each action is assumed (for now) to be a single agent's action
        match(action):
            case ActionEnum.IDLE.value: 
                pass 
            case ActionEnum.MOVE_NORTH.value:
                self.assembler.move_direction(self.world, Direction.NORTH)
            case ActionEnum.MOVE_SOUTH.value:
                self.assembler.move_direction(self.world, Direction.SOUTH)
            case ActionEnum.MOVE_EAST.value:
                self.assembler.move_direction(self.world, Direction.EAST)
            case ActionEnum.MOVE_WEST.value:
                self.assembler.move_direction(self.world, Direction.WEST)
            case ActionEnum.ROTATE_CW.value:
                self.assembler.rotate_cw()  
            case ActionEnum.ROTATE_CCW.value: 
                self.assembler.rotate_ccw() 
            case ActionEnum.SWITCH_MODE.value: 
                self.assembler.switch_mode()  

        self.world.update()
        reward = self.world.global_reward - 1
        self.state = self.world.get_state()
        done = False 
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
