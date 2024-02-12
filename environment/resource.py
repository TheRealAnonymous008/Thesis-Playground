from .world_tile import WorldTile
from .vector import *
from .tiles import Sprite, AssetProfiles
from .constants import DEFAULT_RECT
from enum import Enum
from .direction import * 

class ResourceTile(WorldTile):
    def __init__(self, world, position : Vector, sprite : Sprite = None ):
        super().__init__(world = world,
                         position=position,
                         sprite=sprite
                         )
        self.is_passable = False 
        self.velocity = Vector(0, 0)

    
    def move_direction(self, world, direction : Direction):
        # FIrst get all the resources in that 
        if direction == Direction.NONE:
            return 

        offset = get_forward(direction)
        self.move_offset(world, offset)

    def move_offset(self, world, offset: Vector):
        if offset.is_equal(ZERO_VECTOR):
            return 
        
        opposite = offset.mult(-1)
        current = self.position 
        resources_to_update = []
        while world.has_resource(current):
            resources_to_update.append(world.get_resource(current))
            current = current.add(offset)

            next = world.get_resource(current)
            if next is not None:
                if next.velocity.is_equal(opposite):
                    return 
                if not next.velocity.is_equal(offset) and not next.velocity.is_equal(ZERO_VECTOR):
                    break 
            
        for rsrc in resources_to_update:
            rsrc : WorldTile = rsrc 
            rsrc.move(world, offset)

    def apply_velocity(self, direction : Direction) :
        self.velocity = get_forward(direction)

    def update(self, world): 
        self.move_offset(world, self.velocity)

    def post_update(self, world):
        self.velocity = ZERO_VECTOR
        
class ResourceType(Enum):
    RED = 1,
    BLUE = 2,


class RedResource(ResourceTile):
    def __init__(self, world, position : Vector):
        super().__init__( world = world,
                         position=position,
                         sprite = Sprite(AssetProfiles.RED_RESOURCE, DEFAULT_RECT, 2)
                         )        
        
class BlueResource(ResourceTile):
    def __init__(self, world, position : Vector):
        super().__init__( world = world,
                         position=position,
                         sprite = Sprite(AssetProfiles.BLUE_RESOURCE, DEFAULT_RECT, 2)
                         )        