from .direction import Direction, DirectionVectors
from .tiles import Sprite, AssetProfiles
from .constants import DEFAULT_RECT
from .vector import Vector
from .world_tile import WorldTile

class FactoryComponent(WorldTile):
    def __init__(self, position = Vector(0, 0), rotation : int = 0 , should_render = True, sprite : Sprite = None ):
        super().__init__(position=position,
                         should_render=should_render,
                         sprite=sprite
                         )
        self.rotation : int = rotation
        self.rotate(self.rotation)

    def update_transform(self, position : Vector, rotation : Direction):
        self.place(position)
        self.rotate(rotation)

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
            self.sprite.set_rotation(rotation)

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

    def update(self, world):
        pass 

class Assembler(FactoryComponent):
    def __init__(self, position = Vector(0, 0), rotation = Direction.EAST, should_render = True ):
        super().__init__(position = position, 
                         rotation = rotation, 
                         should_render= should_render, 
                         sprite = Sprite(AssetProfiles.ASSEMBLER, DEFAULT_RECT))

    def update(self, world):
        self.move_direction(Direction.SOUTH)
        self.rotate_cw()


class ConveyorBelt(FactoryComponent):
    def __init__(self, position = Vector(0, 0), rotation = Direction.EAST, should_render = True):
        super().__init__(position = position, 
                         rotation = rotation, 
                         should_render= should_render,
                         sprite = Sprite(AssetProfiles.CONVEYOR_BELT, DEFAULT_RECT))
        self.is_occupied = False


    def update(self, world):
        # Check if the current tile is occupied by anything

        # Conveyor belts pull resources from the opposite of where they are facing if they are not occupied
        pass 
    