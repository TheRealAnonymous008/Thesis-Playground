import numpy as np 
import pygame as pg 
from typing import Tuple, Iterable

class WorldCell: 
    """
    The basic unit within the factory. 
    """
    def __init__(self):
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
    
    def get_cell(self, x : int, y : int) -> WorldCell:
        return self.map[x][y]

