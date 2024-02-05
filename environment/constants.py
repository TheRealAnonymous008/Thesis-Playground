import pygame
from .vector import Vector


BOUNDS = Vector(10, 10)
BLOCK_SIZE = 48
DEFAULT_RECT = pygame.rect.Rect(0, 0, BLOCK_SIZE, BLOCK_SIZE)

def is_in_bounds(v : Vector):
    return v.x >= 0 and v.y >= 0 and v.x < BOUNDS.x and v.y < BOUNDS.y