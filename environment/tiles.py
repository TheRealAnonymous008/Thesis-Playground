import pygame 
import os
from enum import Enum

class Sprite(pygame.sprite.Sprite):
    def __init__(self, path, rect : pygame.Rect):
        super().__init__()
        self.image = pygame.transform.scale(pygame.image.load(path), (rect.w, rect.h))
        self.rect = rect

    def rotate(self, angle):
        self.image = pygame.transform.rotate(self.image, angle)

    def draw(self, surface):
        surface.blit(self.image, self.rect)

class AssetProfiles:
    ASSEMBLER = os.path.join('./assets', 'Assembler.png')
    CONVEYOR_BELT = os.path.join('./assets', 'ConveyorBelt.png')
    EMPTY = os.path.join('./assets', 'Empty.png')
    MERGER = os.path.join('./assets', 'Merger.png')
    OUTPORT = os.path.join('./assets', 'OutPort.png')
    SPLITTER = os.path.join('./assets', 'Splitter.png')
    WALL = os.path.join('./assets', 'Wall.png')