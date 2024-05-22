import numpy as np 
import pygame as pg 
import pygame_gui as gui

from typing import Tuple

from manenv.components import *

from .core.world import World, WorldCell
from .core.product import Product
from .core.effector import Effector
from .utils.vector import make_vector
from .asset_paths import AssetPath
from .core.component import * 

UI_WIDTH = 300 

class ProductDisplayWindow(gui.elements.UIPanel):
    """
    Popup window for rendering product
    """
    def __init__(self, display_dims : Tuple, manager : gui.UIManager):
        self._window_size : Tuple = (UI_WIDTH, UI_WIDTH)
        self._product_surface = pg.Surface(self._window_size)

        super().__init__(
            relative_rect = pg.Rect(display_dims[0] - self._window_size[0], 0, self._window_size[0], self._window_size[1]),
            manager = manager,
        )
        

    def render_cell(self, cell : WorldCell):
        self.clear_view()
        if (len(cell._products) == 0):
            return 
        
        self.set_curr_product(cell._products[0])
        
    def update(self, time_delta: float):
        super().update(time_delta)

    def set_curr_product(self, product : Product):
        px, py = product._structure.shape 
        s = max(px, py)
        cell_size = self._window_size[0] / s, self._window_size[1] / s

        for x in range(s):
            for y in range(s):
                if not (x < px and y < py):
                    continue 

                asset = AssetPath.get_product_asset(product._structure[x][y])
                if asset == "":
                    continue 
                
                img = pg.image.load(asset) 
                img = pg.transform.scale(img, cell_size)
                lx, ly = x * self._window_size[0] / s, y * self._window_size[1] / s
                self._product_surface.blit(img, pg.Rect(ly, lx, self._window_size[1], self._window_size[0]))

        self.set_image(self._product_surface)

    def clear_view(self):
        self._product_surface.fill(pg.Color('#c0c0c0'))
        self.set_image(self._product_surface)


class AssemblerDisplayWindow(gui.elements.UIPanel):
    """
    Popup window for rendering stuff in the assembler
    """
    def __init__(self, display_dims : Tuple, manager : gui.UIManager):
        self._window_size : Tuple = (UI_WIDTH, UI_WIDTH)
        self._assembler_surface = pg.Surface(self._window_size)

        super().__init__(
            relative_rect = pg.Rect(display_dims[0] - self._window_size[0], display_dims[1] - self._window_size[1], self._window_size[0], self._window_size[1]),
            manager = manager,
        )
        

    def render_cell(self, cell : WorldCell):
        self.clear_view()
        self._render_assembler(cell._factory_component)
        
    def update(self, time_delta: float):
        super().update(time_delta)

    def _render_assembler(self, assembler : Assembler):
        px, py = assembler._workspace_size[0] - 1, assembler._workspace_size[1] - 1
        cell_size = self._window_size[0] / px, self._window_size[1] / py
        
        for x in range(px):
            for y in range(py):
                asset = AssetPath.get_product_asset(assembler._workspace[x][y])
                if asset == "":
                    continue 
                
                img = pg.image.load(asset) 
                img = pg.transform.scale(img, cell_size)
                img.fill((255, 255, 255, 128), None, pg.BLEND_RGBA_MULT)
                lx, ly = x * self._window_size[0] / px, y * self._window_size[1] / py
                self._assembler_surface.blit(img, pg.Rect(ly, lx, self._window_size[0], self._window_size[1]))

        for effector in assembler._effectors:
            asset = effector._asset
            x, y = effector._position[0], effector._position[1]
            img = pg.image.load(asset) 
            img = pg.transform.scale(img, cell_size)
            lx, ly = x * self._window_size[0] / px, y * self._window_size[1] / py
            self._assembler_surface.blit(img, pg.Rect(ly, lx, self._window_size[0], self._window_size[1]))



        self.set_image(self._assembler_surface)

    def clear_view(self):
        self._assembler_surface.fill(pg.Color('#c0c0c0'))
        self.set_image(self._assembler_surface)



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


    def render(self, update_mode = False ):
        pg.init()
        screen = pg.display.set_mode(self.display_dims)
        background = pg.Surface(self.display_dims)
        background.fill(pg.Color('#000000'))

        ui_manager = gui.UIManager(self.display_dims)
        clock = pg.time.Clock()

        # UI elements
        product_window = ProductDisplayWindow(self.display_dims, ui_manager)
        assembler_window = AssemblerDisplayWindow(self.display_dims, ui_manager)

        self.is_running = True 
        while self.is_running :
            time_delta = clock.tick(60)/1000.0

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.is_running = False 
                elif event.type == pg.KEYDOWN: 
                    if event.key == pg.K_TAB:
                        self.world.log_world_status()
                elif event.type == pg.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        mouse_pos = pg.mouse.get_pos()
                        clicked_cell = self.get_clicked_cell(mouse_pos)
                        if clicked_cell:
                            if type(clicked_cell._factory_component) == Assembler: 
                                assembler_window.render_cell(clicked_cell)
                            else:
                                product_window.render_cell(clicked_cell)
            
                ui_manager.process_events(event)

            if update_mode:
                self.world.update()

            ui_manager.update(time_delta)

            
            screen.blit(background, (0, 0))
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
                            rect = pg.Rect( r_top, r_left, self.cell_dims[0], self.cell_dims[1])
                        )
                        # Cell fill
                        pg.draw.rect(
                            surface=screen, 
                            color= pg.Color(0, 0, 0), 
                            rect = pg.Rect(r_top, r_left, self.cell_dims[0] * 0.95, self.cell_dims[1] * 0.95)
                        )

                        # Cell contents
                        # Factory Component
                        if cell._factory_component != None: 
                            if type(cell._factory_component) != Conveyor:
                                img = pg.image.load(cell._factory_component._asset)
                                img = pg.transform.scale(img, (self.cell_dims[0], self.cell_dims[1]))
                                img.convert()
                                rect = img.get_rect()
                                rect.center = r_top + self.cell_dims[1] / 2, r_left + self.cell_dims[0] / 2
                                screen.blit(img, rect)
                                
            # Draw conveyors
                        # Draw cell bgs
            for i in range(0, self._visible_cells[0]):
                for j in range(0, self._visible_cells[1]):
                    cell = self.world.get_cell(make_vector(i, j))
                    r_left, r_top = i * self.cell_dims[0], j * self.cell_dims[1]
                    if cell != None:
                        if type(cell._factory_component) == Conveyor:
                            self._render_conveyor(cell, screen)
                                
            ui_manager.draw_ui(screen)
            pg.display.flip()
        
        pg.quit()



    def get_clicked_cell(self, mouse_pos) -> WorldCell:
        mouse_x, mouse_y = mouse_pos
        x, y = int(mouse_x / self.cell_dims[0]), int(mouse_y / self.cell_dims[1])
        return self.world.get_cell(make_vector(y, x))
    
    def _render_conveyor(self, cell : WorldCell, surface : pg.Surface):
        cv = self._get_cell_world_position(cell)
        conveyor : Conveyor = cell._factory_component

        for y in range(-1, 2):
            for x in range(-1, 2):
                neighbor_cell = self.world.get_cell(cell._position + make_vector(y, x))
                if neighbor_cell == None: 
                    continue 
                
                nv = self._get_cell_world_position(neighbor_cell)
                
                if conveyor._weights[x + 1][y + 1] > 0: 
                    pg.draw.line(surface, pg.Color(0, 0, 255), cv, nv, 5)
                elif conveyor._weights[x + 1][y + 1] < 0: 
                    pg.draw.line(surface, pg.Color(255, 0, 0), cv, nv, 5)

    def _get_cell_world_position(self, cell : WorldCell) -> Vector:
        i, j = cell._position[0], cell._position[1]
        r_left, r_top = i * self.cell_dims[0], j * self.cell_dims[1]
        c_x, c_y = r_left + self.cell_dims[0] / 2, r_top + self.cell_dims[1] / 2

        return make_vector(c_x, c_y)