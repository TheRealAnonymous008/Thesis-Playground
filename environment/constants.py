import pygame
from .vector import Vector


BOUNDS = Vector(10, 10)
BLOCK_SIZE = 64
DEFAULT_RECT = pygame.rect.Rect(0, 0, BLOCK_SIZE, BLOCK_SIZE)

BIG_NUMBER = 1000000000000