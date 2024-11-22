from __future__ import annotations

from core.action import *
from core.direction import *
from sar.sar_agent import *
from sar.sar_env_params import * 
from sar.sar_world  import *
from gymnasium.spaces import * 

import torch.nn as nn

ACTION_DIMS = 4

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
    
    def get_action_mask(self, agent: SARAgent, world: SARWorld, device : str = "cpu"):
        """
        Returns an action mask indicating the valid actions for the agent in the current world state.
        Valid actions are marked as 1, and invalid actions are marked as 0.
        """
        # Get the current position of the agent
        current_position = agent.current_position_const
        # Initialize the action mask as a torch tensor with zeros
        action_mask = torch.ones(ACTION_DIMS, dtype=torch.float32, device=device) * -torch.inf

        # Iterate through possible actions and check traversability
        for dir in DIRECTION_MAP.keys():
            new_position = current_position + Direction.get_direction_of_movement(dir)
            if world.is_traversable(new_position):
                action_mask[dir.value - 1] = 0  # Mark as valid

        return action_mask


    def get_observation_space(self, agent : SARAgent):
        vis = agent._traits._visibility_range
        return Dict({
            "Belief": Box(-1, 1, (self._belief_dims, )),
            "Traits": Box(low = 0, high = np.inf),
            "Vision" : Box(0, 1, (2 * vis + 1, 2 * vis + 1)),
            "Terrain": Box(0, 1, (2 * vis + 1, 2 * vis + 1)),
            "State": Box(low = 0, high = np.inf),
        })

    def get_observation(self, agent : SARAgent):
        obs : SARObservation = agent.local_observation
        
        return {
            "Belief": agent._current_belief,
            "Traits": agent.trait_as_tensor,
            "Vision" : obs.victim_map,
            "Terrain" : obs.terrain_map,
            "State": agent.state_as_tensor
        }