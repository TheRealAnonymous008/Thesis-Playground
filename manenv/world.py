import numpy as np 
from typing import Tuple

class WorldCell: 
    """
    The basic unit within the factory. The world cell may contain:
    - a factory component
    - a product
    - a robot 
    """

    def __init__(self):
        self.reset()

    def reset(self):
        pass 

class World: 
    """
    Contains information about the smart factory environment 
    """
    def __init__(self, shape : Tuple): 
        """
        shape - the dimensions of the environment in (width, height) format 
        """
        self.shape : Tuple = shape 
        self.map : list[list[WorldCell]] = [[WorldCell() for _ in range(shape[1])] for _ in range(shape[0])]

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

    