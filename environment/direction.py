from enum import Enum 
from .vector import Vector

class Direction(Enum):
    NORTH = 1
    SOUTH = 2,
    EAST = 3, 
    WEST = 4

class DirectionVectors:
    NORTH = Vector(0, -1)
    SOUTH = Vector(0, 1)
    EAST = Vector(1, 0)
    WEST = Vector(-1, 0)

def get_forward(dir : Direction):
    match(dir):
        case Direction.NORTH: return DirectionVectors.NORTH
        case Direction.SOUTH: return DirectionVectors.SOUTH
        case Direction.EAST: return DirectionVectors.EAST
        case Direction.WEST: return DirectionVectors.WEST