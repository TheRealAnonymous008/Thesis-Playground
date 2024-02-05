import pygame

BLOCK_SIZE = 48
DEFAULT_RECT = pygame.rect.Rect(0, 0, BLOCK_SIZE, BLOCK_SIZE)

class Vector: 
    def __init__(self, x, y):
        self.x = x 
        self.y = y

