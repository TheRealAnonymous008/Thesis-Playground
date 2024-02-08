from .world_tile import WorldTile
from .vector import Vector 
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

    
    def move_direction(self, world, direction : Direction):
        # FIrst get all the resources in that 
        current = self.position 
        offset = None 
        match(direction):
            case Direction.NORTH:
                offset = DirectionVectors.NORTH
            case Direction.SOUTH:
                offset = DirectionVectors.SOUTH
            case Direction.EAST:
                offset= DirectionVectors.EAST 
            case Direction.WEST:
                offset = DirectionVectors.WEST

        resources_to_update = []
        while (world.has_resource(current)):
            resources_to_update.append(world.get_resource(current))
            current = current.add(offset)

        for rsrc in resources_to_update:
            rsrc : WorldTile = rsrc 
            rsrc.move(world, offset)
        
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