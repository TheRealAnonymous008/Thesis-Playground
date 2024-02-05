import pygame 

from .direction import Direction, DirectionVectors
from .tiles import Sprite, AssetProfiles
from .constants import BLOCK_SIZE, DEFAULT_RECT, is_in_bounds
from .vector import Vector

class FactoryComponent: 
    def __init__(self, position = Vector(0, 0), rotation : int = 0 , should_render = True, sprite : Sprite = None ):
        self.position : Vector = position
        self.rotation : int = rotation

        self.tile : Sprite = sprite 
        if sprite is None: 
            self.should_render = False 
        else: 
            self.should_render = should_render
        
        self.place(self.position)
        self.rotate(self.rotation)


    def render(self, surface : pygame.surface.Surface):
        if self.should_render:
            self.tile.draw(surface)

    def update_transform(self, position : Vector, rotation : Direction):
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

    def rotate(self, direction : Direction):
        rotation = 0
        match(direction):
            case Direction.NORTH:
                rotation = 90
            case Direction.SOUTH:
                rotation = -90
            case Direction.EAST:
                rotation = 0 
            case Direction.WEST:
                rotation = -180

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

    def update(self):
        pass 

class Assembler(FactoryComponent):
    def __init__(self, position = Vector(0, 0), rotation = Direction.EAST, should_render = True ):
        super().__init__(position = position, 
                         rotation = rotation, 
                         should_render= should_render, 
                         sprite = Sprite(AssetProfiles.ASSEMBLER, DEFAULT_RECT))


class ConveyorBelt(FactoryComponent):
    def __init__(self, position = Vector(0, 0), rotation = Direction.EAST, should_render = True):
        super().__init__(position = position, 
                         rotation = rotation, 
                         should_render= should_render,
                         sprite = Sprite(AssetProfiles.CONVEYOR_BELT, DEFAULT_RECT))
        self.is_occupied = False


    def update(self):
        # Check if the current tile is occupied by anything

        # Conveyor belts pull resources from the opposite of where they are facing if they are not occupied
        pass 
    