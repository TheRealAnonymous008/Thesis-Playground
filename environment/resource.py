# TODO: Fix issue where functionality doesn't work for north and west face due to iteration. 
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
        self.ignore_same = False 
        self.has_moved = False 

        self.links = set()
        self.id = -1

    def move_direction(self, world, direction : Direction):
        # FIrst get all the resources in that 
        if direction == Direction.NONE:
            return 
        
        offset = get_forward(direction)
        self.move_offset(world, offset)

    def move_offset(self, world, offset: Vector):
        # Move this entire cluster 
        if self.has_moved:
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

                current.place(world, current.position.add(offset))

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
            if next_rsrc is not None and next_rsrc.id != current.id and not current.ignore_same :
                return False 
            
            for neighbor in current.links: 
                stack.append(neighbor)

        return True 
    
    def can_push(self, world, offset):
         # Check if all its neighbors can move 
        if offset.is_equal(ZERO_VECTOR):
            return False 
        
        stack = [self]
        visited = set()
        while len(stack) > 0: 
            current = stack.pop()
            if current in visited:
                continue 
            visited.add(current)

            if not world.is_passable(current.position.add(offset)):
                return False 
            
            for neighbor in current.links: 
                stack.append(neighbor)

        return True 

    def get_next_resource(self, world ,offset):
        return world.get_resource(self.position.add(offset))

    def apply_velocity(self, direction : Direction, ignore_neighbors = False) :
        self.velocity = get_forward(direction)
        self.ignore_same = ignore_neighbors

    def update(self, world): 
        self.move_offset(world, self.velocity)

    def post_update(self, world):
        component = world.factory.get_component(self.position)
        if component is None: 
            self.velocity = ZERO_VECTOR
 
        self.has_moved = False 
    
    def push(self, world, direction : Direction):
        if direction == Direction.NONE:
            return False 
        
        offset : Vector = get_forward(direction)
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
                x.apply_velocity(direction, True)
        return True 
    
    def pull(self, world, assembler, direction):
        pass


    def merge(self, other):
        self.links.add(other)
        other.links.add(self)

        other.id = self.id

    def draw(self, surface : Surface):
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