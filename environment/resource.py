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
        self.is_passable = False
        self.velocity = Vector(0, 0)

        self.links = set()
        self.id = -1

        # This is used to iterate over neighbors
        self.updated_flag = False
        self.pushed_flag = False 

    def move(self, world, offset : Vector):
        if self.updated_flag:
            return 
        
        super().move(world, offset)

    def move_direction(self, world, direction : Direction):
        # FIrst get all the resources in that 
        if direction == Direction.NONE:
            return 
        
        offset = get_forward(direction)
        self.move_offset(world, offset)

    def move_offset(self, world, offset: Vector):
        if self.can_move(world, offset):
            self.move(world, offset)

    def can_move(self, world, offset : Vector):
        if not self.can_push(world, offset):
            return False 

        # Check if it can move. Ignore any resources that are part of this cluster 
        next_rsrc : ResourceTile = self.get_next_resource(world, offset)
        if next_rsrc is not None and next_rsrc.id != self.id :
            return False 

        # Check if all its neighbors can move 
        self.updated_flag = True 
        for neighbor in self.links:
            neighbor : ResourceTile = neighbor 
            if neighbor.updated_flag == True:
                continue 
            
            if not neighbor.can_move(world, offset):
                neighbor.updated_flag = True  
                return False 
        
        self.reset_updated_flag()
        return True 
    
    def can_push(self, world, offset):
         # Check if all its neighbors can move 
        if offset.is_equal(ZERO_VECTOR):
            return False 
        
        if not world.is_passable(self.position.add(offset)):
            return False 
        
        self.updated_flag = True 
        for neighbor in self.links:
            neighbor : ResourceTile = neighbor 
            if neighbor.updated_flag == True:
                continue 
            
            neighbor.updated_flag = True  
            if not neighbor.can_push(world, offset):
                return False 
            
        return True 

    def get_next_resource(self, world ,offset):
        return world.get_resource(self.position.add(offset))

    def apply_velocity(self, direction : Direction) :
        self.velocity = get_forward(direction)

    def update(self, world): 
        self.move_offset(world, self.velocity)

    def post_update(self, world):
        component = world.factory.get_component(self.position)
        if component is None: 
            self.velocity = ZERO_VECTOR

        self.pushed_flag = False 
        self.reset_updated_flag()
    
    def push(self, world, direction : Direction):
        offset : Vector = get_forward(direction)
        if self.can_push(world, offset) and not self.velocity.is_equal(ZERO_VECTOR):
            return False 
        
        self.pushed_flag = True
        next_rsrc = self.get_next_resource(world, offset)
        if next_rsrc != None: 
            self.merge(next_rsrc)
        self.apply_velocity(direction)

        # Also push the neighbors 
        for neighbor in self.links:
            if not neighbor.pushed_flag:
                neighbor.push(world, direction)
        return True 
                
    def merge(self, other):
        self.links.add(other)
        other.links.add(self)

        other.id = self.id

    def draw(self, surface : Surface):
        
        if self.updated_flag: 
            return 
        
        super().draw(surface)
        for neighbor in self.links:
            neighbor : ResourceTile = neighbor 
            pg.draw.line(surface, (255, 255, 255), self.sprite.get_position(), neighbor.sprite.get_position(), 10)

    def reset_updated_flag(self):
        self.updated_flag = False 
        for neighbor in self.links: 
            if neighbor.updated_flag:
                neighbor.reset_updated_flag()

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