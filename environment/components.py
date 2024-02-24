from environment.direction import Direction
from .direction import *
from .tiles import Sprite, AssetProfiles
from .constants import DEFAULT_RECT
from .vector import *
from .world_tile import WorldTile
from .resource import *
from enum import Enum

class FactoryComponent(WorldTile):
    def __init__(self, world, position : Vector,  direction : Direction = Direction.EAST , sprite : Sprite = None ):
        super().__init__(world=world,
                         position=position,
                         sprite=sprite
                         )
        self.direction = direction 
        self.rotation = get_rotation(direction)
        self.rotate(direction)

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
        self.direction = direction

    def rotate_cw(self):
        self.rotate(rotate_dir_cw(self.direction))
    
    def rotate_ccw(self):
        self.rotate(rotate_dir_ccw(self.direction))

    def draw(self, surface):
        self.sprite.set_rotation(self.rotation)
        super().draw(surface)

class ComponentTypes(Enum):
    ASSEMBLER = 1,
    CONVEYOR = 2,
    SPAWNER = 3, 
    SPLITTER = 4,
    MERGER = 5,
    OUTPORT = 6

class AssemblerMode(Enum):
    PUSH = 1,
    PULL = 2

class Assembler(FactoryComponent):
    def __init__(self, world, position : Vector,  direction = Direction.EAST ):
        super().__init__(position = position,
                         world = world, 
                         direction = direction, 
                         sprite = Sprite(AssetProfiles.ASSEMBLER, DEFAULT_RECT, 1))
        
        self.mode = AssemblerMode.PUSH
        self.is_passable = False 

    def move_direction(self, world, direction: Direction):
        if self.mode == AssemblerMode.PUSH:
            self.push(world, direction)
        else: 
            self.pull(world, direction)
    
    def push(self, world, direction : Direction):
        offset : Vector = get_forward(direction)

        rsrc : ResourceTile = world.get_resource(self.position + offset)
        if rsrc is not None:
            rsrc.push(direction)
        else:
            super().move_direction(world, direction)

    def pull(self, world, direction : Direction):
        reverse_offset = get_forward(get_reverse(self.direction))
        rsrc : ResourceTile = world.get_resource(self.position + reverse_offset)
        super().move_direction(world, direction)
        if rsrc is not None:
            rsrc.push(direction)

    def switch_mode(self):
        if self.mode == AssemblerMode.PULL:
            self.mode = AssemblerMode.PUSH
        else:
            self.mode = AssemblerMode.PULL
    def update(self, world):
        pass 

class Spawner(FactoryComponent):
    def __init__(self, world, position : Vector, resource : ResourceType):
        super().__init__(position = position, 
                         world = world,
                         sprite = Sprite(AssetProfiles.SPAWNER, DEFAULT_RECT, 1),
                         )
        self.resource_type = resource 

    def update(self, world):
        if world.has_resource(self.position):
            return 
        
        world.place_resource(self.resource_type, self.position)

class ConveyorBelt(FactoryComponent):
    def __init__(self, world, position : Vector,  direction = Direction.EAST):
        super().__init__(position = position,
                         world = world, 
                         direction = direction, 
                         sprite = Sprite(AssetProfiles.CONVEYOR_BELT, DEFAULT_RECT, 1))
        self.direction = direction
        
    def update(self, world):
        # Check if the current tile is occupied by a resource. If it is move the resource in the direction 
        # of the flow
        if world.has_resource(self.position):
            rsrc : ResourceTile = world.get_resource(self.position)
            rsrc.shift(self.direction)