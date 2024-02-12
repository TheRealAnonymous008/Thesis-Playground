from .world_tile import WorldTile
from .vector import *
from .tiles import Sprite, AssetProfiles
from .constants import DEFAULT_RECT
from enum import Enum
from .direction import * 
from pygame.surface import Surface
import pygame as pg

class ResourceTile(WorldTile):
    def __init__(self, world, position : Vector, sprite : Sprite = None ):
        super().__init__(world = world,
                         position=position,
                         sprite=sprite
                         )
        self.is_passable = True 
        self.velocity = Vector(0, 0)

        self.links = set()

        # This is used to iterate over neighbors
        self.updated_flag = False 

    def move(self, world, offset : Vector):
        if self.updated_flag:
            return 
        
        super().move(world, offset)
        self.updated_flag  = True 
        for neighbor in self.links:
            neighbor : ResourceTile = neighbor 
            neighbor.move(world, offset)

        self.updated_flag = False 

    def move_direction(self, world, direction : Direction) -> bool:
        # FIrst get all the resources in that 
        if direction == Direction.NONE:
            return False 

        offset = get_forward(direction)
        self.move_offset(world, offset)

    def move_offset(self, world, offset: Vector) -> bool:
        if offset.is_equal(ZERO_VECTOR):
            return False 
        
        opposite = offset.mult(-1)
        current = self.position 
        resources_to_update = []
        while world.has_resource(current):
            resources_to_update.append(world.get_resource(current))
            current = current.add(offset)

            next = world.get_resource(current)
            if next is not None:
                if next.velocity.is_equal(opposite):
                    return False 
                if not next.velocity.is_equal(offset) and not next.velocity.is_equal(ZERO_VECTOR):
                    break 
        
        # Get if it is possible to move
        can_move = world.is_passable(current)
        if not can_move:
            return False 
        
        resources_to_update.reverse()
        for rsrc in resources_to_update:
            rsrc : WorldTile = rsrc 
            rsrc.move(world, offset)

        return True 

    def apply_velocity(self, direction : Direction) :
        self.velocity = get_forward(direction)

    def update(self, world): 
        self.move_offset(world, self.velocity)

    def post_update(self, world):
        self.velocity = ZERO_VECTOR
    
    def merge(self, other):
        self.links.add(other)
        other.links.add(self)

    def draw(self, surface : Surface):
        if self.updated_flag: 
            return 
        
        super().draw(surface)
        for neighbor in self.links:
            neighbor : ResourceTile = neighbor 
            pg.draw.line(surface, (255, 255, 255), self.sprite.get_position(), neighbor.sprite.get_position(), 10)


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