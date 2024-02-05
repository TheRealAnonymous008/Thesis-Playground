import pygame 

from .direction import Direction, DirectionVectors
from .tiles import Sprite, AssetProfiles
from .constants import BLOCK_SIZE, DEFAULT_RECT, is_in_bounds
from .vector import Vector

class FactoryComponent: 
    def __init__(self, position = Vector(0, 0), rotation = 0, should_render = True ):
        self.position : Vector = position
        self.rotation : Direction = rotation
        self.tile : Sprite = None 
        self.should_render = should_render

    def render(self, surface : pygame.surface.Surface):
        if self.should_render:
            self.tile.draw(surface)

    def update_transform(self, position : Vector, rotation : int):
        self.place(position)
        self.rotate(rotation)
    
    def place(self, position : Vector):
        if not is_in_bounds(position):
            return 
        self.position = position 

        if self.should_render:
            self.tile.set_coordinate(position)

    def move(self, offset : Vector):
        self.place(self.position.add(offset))

    def rotate(self, rotation : int):
        self.rotation = rotation
        if self.should_render:
            self.tile.set_rotation(rotation)

    def rotate_cw(self):
        self.rotate(self.rotation + 90)
    
    def rotate_ccw(self):
        self.rotate(self.rotation -90)

    def move_direction(self, direction : Direction):
        match(direction):
            case Direction.NORTH:
                self.move(DirectionVectors.NORTH)
            case Direction.SOUTH:
                self.move(DirectionVectors.SOUTH)
            case Direction.EAST:
                self.move(DirectionVectors.EAST)
            case Direction.WEST:
                self.move(DirectionVectors.WEST)

class Assembler(FactoryComponent):
    def __init__(self, position = Vector(0, 0), rotation = 0):
        super().__init__(position, rotation)
        self.tile = Sprite(AssetProfiles.ASSEMBLER, DEFAULT_RECT)

        self.update_transform(self.position, self.rotation)

