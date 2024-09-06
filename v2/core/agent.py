from __future__ import annotations

import numpy as np
from enum import Enum

class Direction(Enum):
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4

class Agent:
    def __init__(self):
        """
        Initializes a simple agent 
        """
        self._position : tuple[int, int] = (0, 0)

        self._current_action = None

    def set_position(self, position : tuple[int, int]):
        """
        Set the agent's position to `position`
        """
        self._position = position

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
            case Direction.NORTH.value: pass 
            case Direction.SOUTH.value: pass 
            case Direction.EAST.value: pass 
            case Direction.WEST.value: pass
            case _: 
                raise Exception(f"Invalid direction specified {val}")