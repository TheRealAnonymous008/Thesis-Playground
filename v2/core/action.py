from __future__ import annotations
from dataclasses import dataclass

from enum import Enum
import numpy as np

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

from .resource import _ResourceType

@dataclass 
class ProductionJob:
    """
    Details on a production job.

    Note that `time` and `qty_produced` will be specified separately from `type`
    """
    prod_type : _ResourceType
    time : int = -1
    qty_produced : int = -1

@dataclass
class ActionInformation:
    """
    Contains attributes and misc. information about an agent's actions.

    If the value is None, then that action was not taken 

    :param movement: Action correpsonding to motion on the world
    :pick_up: Action corresponding to picking up an object in the world
    :put_down:  Action corresponding to putting an item in the inventory down 
    :production_job:  Product to make at the moment
    """
    movement : Direction | None = None
    moved_successfully : bool = False
    pick_up : Direction | None = None 
    put_down : Direction | None = None 
    production_job : _ResourceType | None = None 

    def reset(self):
        self.movement = None
        self.moved_successfully = False
        self.pick_up = None 
        self.put_down = None 
        self.production_job = None 