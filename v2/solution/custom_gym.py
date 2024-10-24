import functools
import random
from copy import copy 

from gymnasium.spaces.space import Space
import numpy as np 
from gymnasium.spaces import Discrete, Box, Dict 

from pettingzoo import ParallelEnv


from core.agent import Agent, AgentState
from core.action import BaseActionParser
from core.render import render_world
from core.world import BaseWorld
from core.direction import Direction 

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
    def __init__(self, 
                 world : BaseWorld, 
                 action_interpreter : BaseActionParser, 
                 time_step_upper = 100):
        """
        Define the initial parameters of the environment

        `world` is an instance of the environment simulation
        `time_step_upper` dictates the truncation time step
        """

        self._world = world
        self._max_time_steps = time_step_upper
        self._action_interpreter = action_interpreter 

        self.render_mode = None
        self.metadata = {
            "name" : "thesis"
        }

    def reset(self, seed = 42, options = None):
        """
        Reset the environemnt. Returns observation and infos
        """
        np.random.seed(seed)
        self._world.reset()
        self.agents : list[str] = self._world.agent_aliases
        self.possible_agents = copy(self.agents)
        
        # Observations
        self._world.update()
        observations = self.get_observations()
        infos = {
            a : {}
            for a in self.agents 
        }

        return observations, infos
    
    def step(self, actions : dict[int, int]):
        """
        Take an action for the current agent 

        `action` key, values correspond to agent ids and action codes.
        """
        for agent_id, action in actions.items():
            agent = self._world.get_agent(agent_id)
            self._action_interpreter.take_action(action, agent)
        
        self._world.update()
        terminations = {a: False for a in self.agents}
        rewards = {a: self._world.get_agent(a).utility for a in self.agents}
        observations = self.get_observations()
        infos = {a : {} for a in self.agents}

        if self.is_finished:
            truncations = {a : True for a in self.agents}
            self.agents= []
        else: 
            truncations = {a: False for a in self.agents}

        return observations, rewards, terminations, truncations, infos


    def render(self, update : callable):
        """
        Render the environment 
        """
        render_world(self._world, (800, 800), update_fn=update, delay_s=0)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_id : int) -> Space:
        """
        Return the observation space of the agent 
        """
        
        # Each agent should know its agent state in some way 
        agent = self._world.get_agent(agent_id)
        return Dict({
            "vision" : Box(0, self._world.total_resource_types, (2 * agent._visibility_range + 1, 2 * agent._visibility_range + 1))
        })

    
    def get_observations(self):
        observations = { 
            a : self.get_observation(a)
            for a in self.agents 
        }
        return observations

    def get_observation(self, agent_id : int): 
        """
        Returns a dictionary of agent observations
        """
        agent = self._world.get_agent(agent_id)
        return {
            "vision" : agent.local_observation.resource_types
        }
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent : int):
        """
        Return the action space of the agent 
        """

        # The action space shall be defined as follows
        # 0 - 3 - MOVE N / S / E / W
        # 4 - 7 - PICK UP N / S / E / W
        # 8 - 11 - PUT_DOWN  N / S / E / W

        return Discrete(12)
    
    @property
    def is_finished(self):
        return self._world._time_step > self._max_time_steps