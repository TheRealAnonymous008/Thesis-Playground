import numpy as np 
import pygame as pg 
from typing import Tuple

class World: 
    """
    Contains information about the smart factory environment 
    """
    def __init__(self, shape : Tuple): 
        self.shape = shape 

    def _width(self):
        return self.shape[0]
    
    def _height(self):
        return self.shape[1]