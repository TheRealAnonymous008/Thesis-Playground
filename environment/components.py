from .direction import Direction, DirectionVectors
from .tiles import Sprite, AssetProfiles
from .constants import DEFAULT_RECT
from .vector import Vector
from .world_tile import WorldTile

class FactoryComponent(WorldTile):
    def __init__(self, world, position : Vector,  rotation : int = 0 , sprite : Sprite = None ):
        super().__init__(world=world,
                         position=position,
                         sprite=sprite
                         )
        self.rotation : int = rotation
        self.rotate(self.rotation)

    def update_transform(self, world, position : Vector, rotation : Direction):
        self.place(world, position)
        self.rotate(rotation)

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

    def rotate_cw(self):
        self.rotate(self.rotation + 90)
    
    def rotate_ccw(self):
        self.rotate(self.rotation -90)

    def draw(self, surface):
        self.sprite.set_rotation(self.rotation)
        super().draw(surface)

    def update(self, world):
        pass 

class Assembler(FactoryComponent):
    def __init__(self, position : Vector, world,  rotation = Direction.EAST ):
        super().__init__(position = position,
                         world = world, 
                         rotation = rotation, 
                         sprite = Sprite(AssetProfiles.ASSEMBLER, DEFAULT_RECT))

    def update(self, world):
        self.move_direction(world, Direction.SOUTH)
        self.rotate_cw()


class ConveyorBelt(FactoryComponent):
    def __init__(self, position : Vector, world,  rotation = Direction.EAST):
        super().__init__(position = position,
                         world = world, 
                         rotation = rotation, 
                         sprite = Sprite(AssetProfiles.CONVEYOR_BELT, DEFAULT_RECT))
        self.is_occupied = False


    def update(self, world):
        # Check if the current tile is occupied by anything

        # Conveyor belts pull resources from the opposite of where they are facing if they are not occupied
        pass 
    