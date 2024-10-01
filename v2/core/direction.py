from __future__ import annotations

import numpy as np
from enum import Enum


class Direction(Enum):
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4

    @staticmethod
    def get_direction_of_movement(dir : Direction):
        try:
            return np.array(DIRECTION_MAP[dir])
        except KeyError:
            raise ValueError(f"Invalid direction specified: {dir}")

DIRECTION_MAP = {
    Direction.NORTH: [0, -1],
    Direction.SOUTH: [0, 1],
    Direction.EAST: [1, 0],
    Direction.WEST: [-1, 0]
}