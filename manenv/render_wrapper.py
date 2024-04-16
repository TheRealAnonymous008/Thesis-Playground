import numpy as np 
import pygame as pg 
from typing import Tuple

from manenv.world import World

class RenderWrapper: 
    """
    Wraps arond the world for rendering and UI related things. 
    """
    def __init__(self, world : World, display_dims : Tuple = (1000, 600), cell_dims : Tuple = (100, 100)): 
        """
        world - the world object to be rendered

        display_dims - the display width and height of the screen

        visible cells - the number of visible cells on each axis. 
        """
        self.world : World = world 
        self.display_dims : Tuple = display_dims
        self.cell_dims : Tuple = cell_dims
        self._visible_cells : Tuple = (int(self.display_dims[0] / self.cell_dims[0]), int(self.display_dims[1] / self.cell_dims[1]))

    def render(self):
        pg.init()
        surface = pg.display.set_mode(self.display_dims)

        self.running = True 
        while self.running :
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False 
                if event.type == pg.KEYDOWN: 
                    pass 
            
            surface.fill(pg.Color(255, 255, 255))
            # Draw cell bgs
            for i in range(0, self._visible_cells[0]):
                for j in range(0, self._visible_cells[1]):
                    pg.draw.rect(
                        surface=surface, 
                        color= pg.Color(0, 0, 0), 
                        rect = pg.Rect(
                            i * self.cell_dims[0], 
                            j * self.cell_dims[1], 
                            self.cell_dims[0] * 0.95, 
                            self.cell_dims[1] * 0.95
                        )
                    )

            pg.display.flip()

        
        pg.quit()