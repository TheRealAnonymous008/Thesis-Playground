from __future__ import annotations
import numpy as np 
from typing import Tuple

from typing import TYPE_CHECKING
if TYPE_CHECKING: 
    from .component import *

from .vector import * 

class WorldCell: 
    """
    The basic unit within the factory. The world cell may contain:
    - a factory component
    - a product
    - a robot 
    """

    def __init__(self, position : Vector):
        self._factory_component : FactoryComponent | None = None 
        self._position : Vector = position
        self.reset()

    def reset(self):
        self._factory_component = None 
        pass 

    def place_component(self, cmp : FactoryComponent):
        if self._factory_component == cmp: 
            return
        self._factory_component = cmp
        cmp.place(self)

class World: 
    """
    Contains information about the smart factory environment 
    """
    def __init__(self, shape : Tuple): 
        """
        shape - the dimensions of the environment in (width, height) format 
        """
        self.shape : Tuple = shape 
        self.map : list[list[WorldCell]] = [[WorldCell(position=make_vector(x, y)) for y in range(shape[1])] for x in range(shape[0])]

    def _width(self):
        return self.shape[0]
    
    def _height(self):
        return self.shape[1]
    
    def reset(self):
        for M in self.map: 
            for r in M: 
                r.reset()
    
    def get_cell(self, x : int, y : int) -> WorldCell | None: 
        if x < 0 or y < 0 or x >= self.shape[0] or y >= self.shape[1]:
            return None 
        
        return self.map[x][y]

    