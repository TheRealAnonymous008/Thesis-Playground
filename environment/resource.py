# TODO: Fix issue where functionality doesn't work for north and west face due to iteration. 
from .world_tile import WorldTile
from .vector import *
from .tiles import Sprite, AssetProfiles
from .constants import DEFAULT_RECT
from enum import Enum
from .direction import * 
from pygame.surface import Surface
import pygame as pg

class ResourceType(Enum):
    RED = 1
    BLUE = 2
    
TOTAL_RESOURCE_TYPES = len(ResourceType)

class MotionDetails:
    def __init__(self, velocity : Vector, ignore_same : bool, is_push : bool):
        self.velocity : Vector = velocity
        self.ignore_same : bool = ignore_same
        self.is_push : bool = is_push 

class ResourceTile(WorldTile):
    def __init__(self, world, position : Vector, type : ResourceType, sprite : Sprite = None ):
        super().__init__(world = world,
                         position=position,
                         sprite=sprite
                         )
        self.type : ResourceType = type
        self.is_passable = False 
        self.motion : MotionDetails = None
        self.is_dead = False 

        self.links = set()
        self.id = -1

    def _move_offset(self, world, offset):
        if offset == ZERO_VECTOR:
            return 
        
        if self.can_move(world, offset): 
            stack = [self]
            visited = set()

            while len(stack) > 0:
                current = stack.pop()
                if current in visited:
                    continue 
                visited.add(current)
                current.has_moved = True 

                current.place(world, current.position + offset)

                for neighbor in current.links:
                    stack.append(neighbor)

    def can_move(self, world, offset : Vector):
        if not self.can_push(world, offset):
            return False 

        stack = [self]
        visited = set()
        while len(stack) > 0: 
            current = stack.pop()
            if current in visited:
                continue 
            visited.add(current)

            # Check if it can move. Ignore any resources that are part of this cluster 
            next_rsrc : ResourceTile = current.get_next_resource(world, offset)
            if next_rsrc is not None and next_rsrc.id != current.id :
                return False 
            
            for neighbor in current.links: 
                stack.append(neighbor)

        return True 
    
    def can_push(self, world, offset):
         # Check if all its neighbors can move 
        if offset == ZERO_VECTOR:
            return False 
        
        stack = [self]
        visited = set()
        while len(stack) > 0: 
            current = stack.pop()
            if current in visited:
                continue 
            visited.add(current)

            if not world.is_passable(current.position + offset):
                return False 
            
            for neighbor in current.links: 
                stack.append(neighbor)

        return True 

    def get_next_resource(self, world ,offset):
        return world.get_resource(self.position + offset)
    
    def get_link_mask(self):
        # Mask is sorted as N, E, W, S
        mask = [0, 0, 0, 0]
        for neighbor in self.links:
            neighbor : ResourceTile = neighbor 
            offset = neighbor.position - self.position
            if offset == DirectionVectors.NORTH: 
                mask[0] = 1
            elif offset == DirectionVectors.EAST: 
                mask[1] = 1
            elif offset == DirectionVectors.WEST: 
                mask[2] = 1
            elif offset == DirectionVectors.SOUTH:
                mask[3] = 1 
            
        return mask

    def update(self, world): 
        if self.motion == None or self.is_dead:
            return
        if not self.motion.is_push:
            self._move_offset(world, self.motion.velocity)
        else:
            self._push(world, self.motion.velocity)

    def post_update(self, world):
        self.motion = None
    
    def push(self, direction : Direction):
        self.motion = MotionDetails(get_forward(direction), True, True)

    def shift(self, direction : Direction):
        self.motion = MotionDetails(get_forward(direction), False, False)
    
    def destroy(self):
        self.is_dead = True 

    def _push(self, world, offset):        
        if not self.can_push(world, offset):
            return False 

        visited = set()
        stack = []
        # Also push the neighbors 
        stack.append(self)

        has_merged = False 
        while len(stack) > 0:
            current = stack.pop()
            if current in visited:
                continue 
            visited.add(current)

            next_rsrc = current.get_next_resource(world, offset)
            if next_rsrc != None and not (next_rsrc in current.links or current in next_rsrc.links) :
                current.merge(next_rsrc)
                has_merged = True 
            for neighbor in current.links:
                stack.append(neighbor)
        
        # Apply velocity to all nodes if no merging happened
        if not has_merged:
            for x in visited:
                x : ResourceTile = x 
                x.move(world, offset)
        return True 


    def merge(self, other):
        self.links.add(other)
        other.links.add(self)

        other.id = self.id

    def draw(self, surface : Surface):
        super().draw(surface)
        for neighbor in self.links:
            neighbor : ResourceTile = neighbor 
            pg.draw.line(surface, (255, 255, 255), self.sprite.get_position(), neighbor.sprite.get_position(), 10)




class RedResource(ResourceTile):
    def __init__(self, world, position : Vector):
        super().__init__( world = world,
                         position=position,
                         type=ResourceType.RED,
                         sprite = Sprite(AssetProfiles.RED_RESOURCE, DEFAULT_RECT, 2)
                         )        
        
class BlueResource(ResourceTile):
    def __init__(self, world, position : Vector):
        super().__init__( world = world,
                         position=position,
                         type=ResourceType.BLUE,
                         sprite = Sprite(AssetProfiles.BLUE_RESOURCE, DEFAULT_RECT, 2)
                         )        