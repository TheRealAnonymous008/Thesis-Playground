from __future__ import annotations

from core.action import *
from core.direction import *
from sar.sar_agent import *
from gymnasium.spaces import * 

class SARActionInterpreter(BaseActionParser):
    def __init__(self):
        super().__init__()

    
    def take_action(self, action_code : int, agent : SARAgent):
        match (action_code) : 
            case 0 : agent.move(Direction.NORTH) 
            case 1 : agent.move(Direction.SOUTH) 
            case 2 : agent.move(Direction.EAST) 
            case 3 : agent.move(Direction.WEST)
            case _ : pass # Do nothing, placeholder for now.
    
    def get_action_space(self, agent : SARAgent):
        return Discrete(4)

    def get_observation_space(self, agent : SARAgent):
        vis = agent._traits._visibility_range
        return Dict({
            # "vision" : Box(0, 1, (2 * vis + 1, 2 * vis + 1))
            "Victims" : Box(0, 1, (2 * vis + 1, 2 * vis + 1))
        })