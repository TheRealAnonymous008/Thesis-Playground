import numpy as np 
import pygame as pg 
import pygame_gui as gui

from typing import Tuple

from .world import World, WorldCell
from .product import Product
from .vector import make_vector
from .asset_paths import AssetPath

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
        cell_size = self._window_size[0] / px, self._window_size[1] / py

        for x in range(px):
            for y in range(py):
                asset = AssetPath.get_product_asset(product._structure[x][y])
                if asset == "":
                    continue 
                
                img = pg.image.load(asset) 
                img = pg.transform.scale(img, cell_size)
                lx, ly = x * self._window_size[0] / px, y * self._window_size[1] / py
                self._product_surface.blit(img, pg.Rect(lx, ly, self._window_size[0], self._window_size[1]))

        self.set_image(self._product_surface)

    def clear_view(self):
        self._product_surface.fill(pg.Color('#c0c0c0'))
        self.set_image(self._product_surface)


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
        background = pg.Surface(self.display_dims)
        background.fill(pg.Color('#000000'))

        ui_manager = gui.UIManager(self.display_dims, "theme.json")
        clock = pg.time.Clock()

        # UI elements
        product_window = ProductDisplayWindow(self.display_dims, ui_manager)
        time_step_label = gui.elements.UILabel(pg.Rect(self.display_dims[0] - UI_WIDTH, self.display_dims[1] - 200, UI_WIDTH, 200), "Timestep = 0")

        self.is_running = True 
        while self.is_running :
            time_delta = clock.tick(60)/1000.0

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.is_running = False 
                elif event.type == pg.KEYDOWN: 
                    pass 
                elif event.type == pg.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        mouse_pos = pg.mouse.get_pos()
                        clicked_cell = self.get_clicked_cell(mouse_pos)
                        if clicked_cell:
                            product_window.render_cell(clicked_cell)
            
                ui_manager.process_events(event)

            time_step_label.text = "Timestep :" + str(self.world._time_step)
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

            ui_manager.draw_ui(screen)
            pg.display.flip()

        
        pg.quit()

    def get_clicked_cell(self, mouse_pos) -> WorldCell:
        mouse_x, mouse_y = mouse_pos
        x, y = int(mouse_x / self.cell_dims[0]), int(mouse_y / self.cell_dims[1])
        return self.world.get_cell(make_vector(x, y))