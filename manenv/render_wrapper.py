import numpy as np 
import pygame as pg 
from pgu.gui import *
from typing import Tuple

from .world import World
from .vector import make_vector

class RenderWrapper: 
    """
    Wraps arond the world for rendering and UI related things. 
    """
    def __init__(self, world : World, display_dims : Tuple = (1000, 600), cell_dims : Tuple = (100, 100)): 
        """
        `world` - the world object to be rendered

        `display_dims` - the display width and height of the screen

        `visible cells` - the number of visible cells on each axis. 
        """
        self.world : World = world 
        self.display_dims : Tuple = display_dims
        self.cell_dims : Tuple = cell_dims
        self._visible_cells : Tuple = (int(self.display_dims[0] / self.cell_dims[0]), int(self.display_dims[1] / self.cell_dims[1]))

    def render(self):
        pg.init()
        screen = pg.display.set_mode(self.display_dims)

        self.running = True 
        while self.running :
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False 
                if event.type == pg.KEYDOWN: 
                    pass 
            
            # Draw cell bgs
            for i in range(0, self._visible_cells[0]):
                for j in range(0, self._visible_cells[1]):
                    cell = self.world.get_cell(make_vector(i, j))
                    r_left, r_top = i * self.cell_dims[0], j * self.cell_dims[1]

                    if cell != None:
                        # Cell border
                        pg.draw.rect(
                            surface=screen, 
                            color= pg.Color(255, 255, 255), 
                            rect = pg.Rect( r_left, r_top, self.cell_dims[0], self.cell_dims[1])
                        )
                        # Cell fill
                        pg.draw.rect(
                            surface=screen, 
                            color= pg.Color(0, 0, 0), 
                            rect = pg.Rect(r_left, r_top, self.cell_dims[0] * 0.95, self.cell_dims[1] * 0.95)
                        )

                        # Cell contents
                        # Factory Component
                        if cell._factory_component != None:       
                            img = pg.image.load(cell._factory_component._asset)
                            img = pg.transform.scale(img, (self.cell_dims[0], self.cell_dims[1]))
                            img.convert()
                            rect = img.get_rect()
                            rect.center = r_left + self.cell_dims[0] / 2, r_top + self.cell_dims[1] / 2
                            screen.blit(img, rect)


            pg.display.flip()

        
        pg.quit()