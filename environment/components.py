import pygame 

from .direction import Direction
from .tiles import Sprite, AssetProfiles
from .constants import BLOCK_SIZE, Vector, DEFAULT_RECT

class FactoryComponent: 
    def __init__(self, position = Vector(0, 0), rotation = 0):
        self.position : Vector = position
        self.rotation : Direction = rotation
        self.tile : Sprite = None 

    def render(self, surface : pygame.surface.Surface):
        self.tile.draw(surface)

    def update_transform(self, position : Vector, rotation : int):
        self.position = position 
        self.rotation = rotation 
        self.tile.set_rotation(rotation)
        self.tile.set_coordinate(position)

class Assembler(FactoryComponent):
    def __init__(self, position = Vector(0, 0), rotation = 0):
        super().__init__(position, rotation)
        self.tile = Sprite(AssetProfiles.ASSEMBLER, DEFAULT_RECT)

        self.position = Vector(1, 2)
        self.rotation = 180
        self.update_transform(self.position, self.rotation)

