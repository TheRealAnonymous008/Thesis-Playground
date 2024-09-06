from __future__ import annotations

import numpy as np
from enum import Enum

from .action import *

class Agent:
    def __init__(self):
        """
        Initializes a simple agent 
        """
        self._position : np.ndarray[int] = np.array([0, 0], dtype=np.int32)

        self._current_action : ActionInformation = ActionInformation()

    def set_position(self, position : np.array):
        """
        Set the agent's position to `position`
        """
        self._position = position

    def get_position(self) -> np.ndarray:
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
        self._current_action.reset()