from __future__ import annotations

from core.action import *
from core.direction import *
from sar.sar_agent import *
from sar.sar_env_params import * 
from gymnasium.spaces import * 

import torch.nn as nn

class SARActionInterpreter(BaseActionParser):
    def __init__(self, belief_dims : int):
        super().__init__()
        self._belief_dims = belief_dims

    
    def take_action(self, action_code : int, agent : SARAgent):
        match (action_code) : 
            case 0 : agent.move(Direction.NORTH) 
            case 1 : agent.move(Direction.SOUTH) 
            case 2 : agent.move(Direction.EAST) 
            case 3 : agent.move(Direction.WEST)
            case _: 
                pass 
    
    def get_action_space(self, agent : SARAgent):
        return Discrete(4)

    def get_observation_space(self, agent : SARAgent):
        vis = agent._traits._visibility_range
        return Dict({
            "Belief": Box(-1, 1, (self._belief_dims, )),
            "Victims" : Box(0, 1, (2 * vis + 1, 2 * vis + 1)),
            "Energy": Box(0, MAX_ENERGY + 1),
        })

    def get_observation(self, agent : SARAgent):
        obs : SARObservation = agent.local_observation
        
        return {
            "Belief": agent._current_belief,
            "Victims" : obs.victim_map,
            "Energy": agent._current_state.current_energy,
        }