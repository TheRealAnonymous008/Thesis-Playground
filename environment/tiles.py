import pygame 
import os
from enum import Enum
from .constants import Vector

class Sprite(pygame.sprite.Sprite):
    def __init__(self, path, rect : pygame.Rect):
        super().__init__()
        self.image = pygame.transform.scale(pygame.image.load(path), (rect.w, rect.h))
        self.sprite = self.image
        self.rect : pygame.Rect = rect

    def set_rotation(self, angle):
        self.sprite = pygame.transform.rotate(self.image, angle)
    
    # Coord is unscaled. 
    def set_coordinate(self, coord : Vector):
        self.rect = self.rect.move(
            coord.x * self.sprite.get_width(),
            coord.y * self.sprite.get_height()
        ) 

    def reset(self):
        self.set_rotation(0)
        self.set_coordinate(0, 0)

    def draw(self, surface):
        surface.blit(self.sprite, self.rect)

class AssetProfiles:
    ASSEMBLER = os.path.join('./assets', 'Assembler.png')
    CONVEYOR_BELT = os.path.join('./assets', 'ConveyorBelt.png')
    EMPTY = os.path.join('./assets', 'Empty.png')
    MERGER = os.path.join('./assets', 'Merger.png')
    OUTPORT = os.path.join('./assets', 'OutPort.png')
    SPLITTER = os.path.join('./assets', 'Splitter.png')
    WALL = os.path.join('./assets', 'Wall.png')

    ARM = os.path.join('./assets', 'Arm.png')