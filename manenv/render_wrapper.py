import numpy as np 
import pygame as pg 
from typing import Tuple

from manenv.world import World

DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

class RenderWrapper: 
    """
    Wraps arond the world for rendering and UI related things. 
    """
    def __init__(self, world : World): 
        self.world = world 

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