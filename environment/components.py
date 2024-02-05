import pygame 

from .direction import Direction
from .tiles import Sprite, AssetProfiles
from .constants import BLOCK_SIZE, Vector

class FactoryComponent: 
    def __init__(self):
        self.position : Vector = Vector(0, 0)
        self.rotation : Direction = Direction.EAST
        self.tile : Sprite = None 

    def render(self, surface : pygame.surface.Surface):
        self.tile.draw(surface)

    def update_transform(self, position : Vector, rotation : int):
        self.position = position 
        self.rotation = rotation 
        self.tile.set_rotation(rotation)
        self.tile.set_coordinate(position)

class Assembler(FactoryComponent):
    def __init__(self):
        super().__init__()
        self.tile = Sprite(AssetProfiles.ASSEMBLER, pygame.rect.Rect(0, 0, BLOCK_SIZE, BLOCK_SIZE))