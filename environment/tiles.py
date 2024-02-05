import pygame 
import os
from enum import Enum

class Sprite(pygame.sprite.Sprite):
    def __init__(self, path, rect : pygame.Rect):
        super().__init__()
        self.image = pygame.transform.scale(pygame.image.load(path), (rect.w, rect.h))
        self.rect = rect

    def draw(self, surface):
        surface.blit(self.image, self.rect)

class AssetProfiles:
    ASSEMBLER = os.path.join('./assets', 'Assembler.png')
    CONVEYOR_BELT = os.path.join('./assets', 'Assembler.png')
    EMPTY = os.path.join('./assets', 'Assembler.png')
    MERGER = os.path.join('./assets', 'Assembler.png')
    OUTPORT = os.path.join('./assets', 'Assembler.png')
    SPLITTER = os.path.join('./assets', 'Assembler.png')
    WALL = os.path.join('./assets', 'Assembler.png')