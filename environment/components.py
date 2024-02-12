from .direction import *
from .tiles import Sprite, AssetProfiles
from .constants import DEFAULT_RECT
from .vector import Vector
from .world_tile import WorldTile
from enum import Enum

class FactoryComponent(WorldTile):
    def __init__(self, world, position : Vector,  rotation : Direction = 0 , sprite : Sprite = None ):
        super().__init__(world=world,
                         position=position,
                         sprite=sprite
                         )
        self.rotation : Direction = rotation
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

class ComponentTypes(Enum):
    ASSEMBLER = 1,
    CONVEYOR = 2,

class Assembler(FactoryComponent):
    def __init__(self, world, position : Vector,  rotation = Direction.EAST ):
        super().__init__(position = position,
                         world = world, 
                         rotation = rotation, 
                         sprite = Sprite(AssetProfiles.ASSEMBLER, DEFAULT_RECT, 1))

    def update(self, world):
        pass 

class ConveyorBelt(FactoryComponent):
    def __init__(self, world, position : Vector,  rotation = Direction.EAST):
        super().__init__(position = position,
                         world = world, 
                         rotation = rotation, 
                         sprite = Sprite(AssetProfiles.CONVEYOR_BELT, DEFAULT_RECT, 1))
        self.direction = rotation
        
    def update(self, world):
        # Check if the current tile is occupied by a resource. If it is move the resource in the direction 
        # of the flow
        if world.has_resource(self.position):
            rsrc = world.get_resource(self.position)
            rsrc.apply_velocity(self.direction)