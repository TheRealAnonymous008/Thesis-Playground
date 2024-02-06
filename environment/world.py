import pygame
from .components import FactoryComponent, Assembler
from .factory import Factory
from .vector import Vector
from .direction import Direction
from .constants import is_in_bounds
from .world_tile import WallTile, EmptyTile, WorldTile

class World:
    def __init__(self, width, height, block_size):
        self.width = width
        self.height = height
        self.block_size = block_size

        self.tiles : [[WorldTile]]= [[None for _ in range(height)] for _ in range(width)]


        self.init_tiles()
        self.init_factory()

        
    def init_tiles(self):
        for x in range(self.width):
            for y in range(self.height):
                self.tiles[x][y] = EmptyTile(Vector(x, y))
    
        self.tiles[3][7] =  WallTile(Vector(3, 7))

    def init_factory(self):
        self.factory = Factory(bounds= Vector(self.width, self.height))
        self.factory.add_assembler(Assembler(), Vector(3, 4), Direction.WEST)

    def draw(self, surface):
        # Draw the base 
        for x in range(self.width):
            for y in range(self.height):
                self.tiles[x][y].draw(surface)

        # Draw the factory components 
        self.factory.render(surface)
            

    def update(self):
        self.factory.update(self)

    def is_passable(self, position : Vector):
        # Is it in bounds
        if not is_in_bounds(position):
            return False
        
        # Is there a wall 
        if not self.tiles[position.x][position.y].is_passable: 
            return False
        
        # Is there a component from the factory that is passable
        return self.factory.is_passable(position)