import functools
import random
from copy import copy 

from gymnasium.spaces.space import Space
import numpy as np 
from gymnasium.spaces import Discrete, Box, Dict 

from pettingzoo import ParallelEnv


from core.agent import Agent, AgentState
from core.action import Direction
from core.render import render_world
from core.world import World

def take_action(action_code : int, agent : Agent):
    match (action_code) : 
        case 0 : agent.move(Direction.NORTH) 
        case 1 : agent.move(Direction.SOUTH) 
        case 2 : agent.move(Direction.EAST) 
        case 3 : agent.move(Direction.WEST)

        case 4 : agent.pick_up(Direction.NORTH) 
        case 5 : agent.pick_up(Direction.SOUTH) 
        case 6 : agent.pick_up(Direction.EAST) 
        case 7 : agent.pick_up(Direction.WEST) 

        case _ : pass # Do nothing, placeholder for now.

class CustomGymEnviornment(ParallelEnv):
    def __init__(self, world : World):
        """
        Define the initial parameters of the environment

        `world` is an instance of the environment simulation
        """

        self._world = world

    def reset(self, seed):
        """
        Reset the environemnt
        """
    
    def step(self, action):
        """
        Take an action for the current agent 
        """

    def render(self, update : callable):
        """
        Render the environment 
        """
        render_world(self._world, (800, 800), update_fn=update, delay_s=0)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent : Agent) -> Space:
        """
        Return the observation space of the agent 
        """
        
        # Each agent should know its agent state in some way 
        
        return Dict({
            "vision" : Box(0, self._world.total_resource_types, (2 * agent._visibility_range + 1, 2 * agent._visibility_range + 1))
        })

    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent : Agent):
        """
        Return the action space of the agent 
        """

        # The action space shall be defined as follows
        # 0 - 3 - MOVE N / S / E / W
        # 4 - 7 - PICK UP N / S / E / W
        # 8 - 11 - PUT_DOWN  N / S / E / W

        return Discrete(12)