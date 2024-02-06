import pygame
from .components import FactoryComponent, Assembler
from .factory import Factory
from .vector import Vector, ZERO_VECTOR, is_in_bounds
from .direction import Direction
from .world_tile import WallTile, EmptyTile, WorldTile

class World:
    def __init__(self, width, height, block_size):
        self.bounds : Vector = Vector(width, height)
        self.block_size = block_size

        self.tiles : [[WorldTile]]= [[None for _ in range(height)] for _ in range(width)]


        self.init_tiles()
        self.init_factory()

        
    def init_tiles(self):
        for x in range(self.bounds.x):
            for y in range(self.bounds.y):
                self.tiles[x][y] = EmptyTile(self, Vector(x, y))
    
        self.tiles[3][7] =  WallTile(self, Vector(3, 7))

    def init_factory(self):
        self.factory = Factory(bounds= Vector(self.bounds.x, self.bounds.y))
        self.factory.add_assembler(self, Vector(3, 4), Direction.WEST)

    def draw(self, surface):
        # Draw the base 
        for x in range(self.bounds.x):
            for y in range(self.bounds.y):
                self.tiles[x][y].draw(surface)

        # Draw the factory components 
        self.factory.render(surface)
            

    def update(self):
        self.factory.update(self)

    def is_passable(self, position : Vector):
        # Is it in bounds
        if not is_in_bounds(position, ZERO_VECTOR, self.bounds):
            return False
        
        # Is there a wall 
        if not self.tiles[position.x][position.y].is_passable: 
            return False
        
        # Is there a component from the factory that is passable
        return self.factory.is_passable(position)