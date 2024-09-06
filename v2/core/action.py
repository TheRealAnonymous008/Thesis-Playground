from __future__ import annotations
from dataclasses import dataclass

from abc import ABC
from enum import Enum

class Direction(Enum):
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4

    @staticmethod
    def get_direction_of_movement(dir : Direction):
        match(dir.value):
            case Direction.NORTH.value: 
                return (0, -1)

            case Direction.SOUTH.value: 
                return (0, 1)

            case Direction.EAST.value: 
                return (1, 0)
            
            case Direction.WEST.value:
                return (-1, 0)

            case _: 
                raise Exception(f"Invalid direction specified {dir}")

@dataclass
class ActionInformation:
    """
    Contains attributes and misc. information about an agent's actions.

    If the value is None, then that action was not taken 

    `movement` - action correpsonding to motion on the world
    """
    movement : Direction | Direction = None

    def reset(self):
        self.movement = None