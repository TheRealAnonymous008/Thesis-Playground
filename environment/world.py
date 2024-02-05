import pygame
from .tiles import Sprite, AssetProfiles

class World:
    def __init__(self, width, height, block_size):
        self.width = width
        self.height = height
        self.block_size = block_size

        self.grid = [[None for _ in range(height)] for _ in range(width)]
        self.sprites : [[Sprite]]= [[None for _ in range(height)] for _ in range(width)]

        self.init_rects()
        
    def init_rects(self):
        for x in range(self.width):
            for y in range(self.height):
                rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                self.sprites[x][y] = Sprite(AssetProfiles.EMPTY, rect)
    
    def draw(self, surface):
        for x in range(self.width):
            for y in range(self.height):
                self.sprites[x][y].draw(surface)
                
