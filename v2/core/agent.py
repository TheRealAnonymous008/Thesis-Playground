from __future__ import annotations

import numpy as np
from enum import Enum

from .observation import LocalObservation

from .action import *

class Agent:
    def __init__(self):
        """
        Initializes a simple agent 
        """
        self._id = id
        self._position : np.ndarray[int] = np.array([0, 0], dtype=np.int32)

        self._current_observation : LocalObservation = None
        self._current_action : ActionInformation = ActionInformation()
        
        # Attributes of the agent
        self._visibility_range : int = 3

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
        

    def move(self, dir : Direction | int):
        """
        Moves an agent along a specified direction. 
        The direction is either a Direction instance or an integer associated with a Direction value.
        """
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
            
    def reset_for_next_action(self):
        """
        Resets the agent for a new action
        """
        self._current_action.reset()

    def set_observation(self, observation : LocalObservation):
        self._current_observation = observation