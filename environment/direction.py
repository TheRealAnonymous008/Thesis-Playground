from enum import Enum 
from .vector import Vector

class Direction(Enum):
    NONE = 0
    NORTH = 1
    EAST = 2 
    SOUTH = 3
    WEST = 4

class DirectionVectors:
    NORTH = Vector(0, -1)
    SOUTH = Vector(0, 1)
    EAST = Vector(1, 0)
    WEST = Vector(-1, 0)
    NONE = Vector(0, 0)

def get_forward(dir : Direction):
    match(dir):
        case Direction.NORTH: return DirectionVectors.NORTH
        case Direction.SOUTH: return DirectionVectors.SOUTH
        case Direction.EAST: return DirectionVectors.EAST
        case Direction.WEST: return DirectionVectors.WEST
        case Direction.NONE: return DirectionVectors.NONE

def get_rotation(dir : Direction):
    match(dir):
        case Direction.NORTH: return 90
        case Direction.SOUTH: return -90
        case Direction.EAST: return 0
        case Direction.WEST: return 180
        case Direction.NONE: return 0

def get_reverse(dir : Direction) -> Direction: 
    match(dir):
        case Direction.NORTH: return Direction.SOUTH
        case Direction.SOUTH: return Direction.NORTH
        case Direction.EAST: return Direction.WEST
        case Direction.WEST: return Direction.EAST
        case Direction.NONE: return Direction.NONE

def rotate_dir_cw(dir : Direction) -> Direction: 
    match(dir):
        case Direction.NORTH: return Direction.EAST
        case Direction.SOUTH: return Direction.WEST
        case Direction.EAST: return Direction.SOUTH
        case Direction.WEST: return Direction.NORTH
        case Direction.NONE: return Direction.NONE

def rotate_dir_ccw(dir : Direction) -> Direction: 
    match(dir):
        case Direction.NORTH: return Direction.WEST
        case Direction.SOUTH: return Direction.EAST
        case Direction.EAST: return Direction.NORTH 
        case Direction.WEST: return Direction.SOUTH
        case Direction.NONE: return Direction.NONE