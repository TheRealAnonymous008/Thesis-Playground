import pygame
from .tiles import Sprite, AssetProfiles
from .components import FactoryComponent, Assembler
from .factory import Factory
from .vector import Vector
from .direction import Direction
from .constants import is_in_bounds

class World:
    def __init__(self, width, height, block_size):
        self.width = width
        self.height = height
        self.block_size = block_size

        self.wall_mask = [[False for _ in range(height)] for _ in range(width)]
        self.sprites : [[Sprite]]= [[None for _ in range(height)] for _ in range(width)]

        self.factory = Factory()
        self.init_rects()
        self.init_factory()

        
    def init_rects(self):
        for x in range(self.width):
            for y in range(self.height):
                rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                self.sprites[x][y] = Sprite(AssetProfiles.EMPTY, rect)
    
    def init_factory(self):
        self.factory.add_assembler(Assembler(), Vector(3, 4), Direction.WEST)

    def draw(self, surface):
        # Draw the base 
        for x in range(self.width):
            for y in range(self.height):
                self.sprites[x][y].draw(surface)

        # Draw the factory components 
        self.factory.render(surface)
            

    def update(self):
        self.factory.update(self)

    def is_passable(self, position : Vector):
        # Is it in bounds
        if not is_in_bounds(position):
            return False
        
        # Is there a wall 
        if self.wall_mask[position.x][position.y]: 
            return False 
        
        # Is there a component from the factory that is passable
        return self.factory.is_passable(position)