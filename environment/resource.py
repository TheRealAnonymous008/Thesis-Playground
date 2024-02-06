from .world_tile import WorldTile
from .vector import Vector 
from .tiles import Sprite, AssetProfiles
from .constants import DEFAULT_RECT
from enum import Enum

class ResourceTile(WorldTile):
    def __init__(self, position : Vector,  should_render = True, sprite : Sprite = None ):
        super().__init__(position=position,
                         should_render=should_render,
                         sprite=sprite
                         )
        self.is_passable = False 
        
class ResourceType(Enum):
    RED = 1,


class RedResource(ResourceTile):
    def __init__(self, position : Vector, should_render = True):
        super().__init__(position=position,
                         should_render=should_render,
                         sprite = Sprite(AssetProfiles.RED_RESOURCE, DEFAULT_RECT)
                         )        