from __future__ import annotations

import numpy as np
from enum import Enum

from .observation import LocalObservation

from .action import *

@dataclass
class AgentSate:
    """
    Contains attributes and misc. information about an agent's internal state.

    `current_energy` - current energy of the agent
    """
    current_energy : float = 0

    def can_move(self):
        return self.current_energy > 0

class Agent:
    def __init__(self):
        """
        Initializes a simple agent 
        """
        self._id = -1
        self._position : np.ndarray[int] = np.array([0, 0], dtype=np.int32)

        self._current_observation : LocalObservation = None
        self._current_action : ActionInformation = ActionInformation()
        self._current_state :  AgentSate = AgentSate()

        # Attributes of the agent. 
        self._visibility_range : int = 3
        self._energy_capacity : float = 100.0

        self.reset()
        
    def reset(self):
        """
        Reset the agent
        """
        self._current_state.current_energy = self._energy_capacity

    def move(self, dir : Direction | int):
        """
        Moves an agent along a specified direction. 
        The direction is either a Direction instance or an integer associated with a Direction value.
        """
        if not self._current_state.can_move():
            return 

        if type(dir) is Direction: 
            val = dir.value
        else: 
            val = dir 

        match(val):
            case Direction.NORTH.value: 
                self._current_action.movement = Direction.NORTH

            case Direction.SOUTH.value: 
                self._current_action.movement = Direction.SOUTH

            case Direction.EAST.value: 
                self._current_action.movement = Direction.EAST
            
            case Direction.WEST.value:
                self._current_action.movement = Direction.WEST

            case _: 
                raise Exception(f"Invalid direction specified {val}")

    def pick_up(self): 
        """
        Pick up a resource adjacent to this agent 
        """
        pass

    def reset_for_next_action(self):
        """
        Resets the agent for a new action
        """
        self._current_action.reset()

    def set_observation(self, observation : LocalObservation):
        self._current_observation = observation

    def get_observation(self) -> LocalObservation:
        return self._current_observation

    def bind_to_world(self, world_id : int):
        """
        Set the agent's ID to be that of the ID assigned to it by the world
        """
        self._id = world_id

    def get_id(self) -> int:
        """
        Return the ID of this agent in the world
        """
        return self._id 

    def set_position(self, position : np.array):
        """
        Set the agent's position to `position`
        """
        self._position = position

    def get_position(self) -> np.ndarray:
        """
        Get a copy of the agent's position
        """
        return self._position.copy()
    
    def get_action(self) -> ActionInformation:
        return self._current_action