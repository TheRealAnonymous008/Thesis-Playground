import pygame
from .tiles import Sprite, AssetProfiles

class World:
    def __init__(self, width, height, block_size):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.grid = [[None for _ in range(height)] for _ in range(width)]

        self.rects = [[pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size) 
                       for y in range(height)]
                       for x in range(width)]
        
        
    def draw(self, surface):
        for x in range(self.width):
            for y in range(self.height):
                sprite = Sprite(AssetProfiles.WALL, self.rects[x][y])
                sprite.draw(surface)
                
