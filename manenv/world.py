import numpy as np 
import pygame as pg 
from typing import Tuple

DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

class World: 
    """
    The World class contains information about the factory environment 
    """
    def __init__(self, shape : Tuple): 
        self.shape = shape 

    def _width(self):
        return self.shape[0]
    
    def _height(self):
        return self.shape[1]

    def render(self):
        pg.init()
        pg.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))

        self.running = True 
        while self.running :
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False 
                if event.type == pg.KEYDOWN: 
                    pass 

        
        pg.quit()