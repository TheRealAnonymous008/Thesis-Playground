import pygame
from .components import * 
from .factory import Factory
from .vector import Vector, ZERO_VECTOR, is_in_bounds
from .direction import Direction
from .world_tile import *
from .resource_manager import ResourceMap
from .resource import ResourceType

class World:
    def __init__(self, width, height, block_size):
        self.bounds : Vector = Vector(width, height)
        self.block_size = block_size

        self.tiles : list(list(WorldTile))= [[None for _ in range(height)] for _ in range(width)]
        self.resource_map  : ResourceMap = ResourceMap(self.bounds)

        self.init_tiles()
        self.init_factory()
        self.init_resources()

        
    def init_tiles(self):
        for x in range(self.bounds.x):
            for y in range(self.bounds.y):
                self.tiles[x][y] = EmptyTile(self, Vector(x, y))
    
        self.tiles[3][7] =  WallTile(self, Vector(3, 7))

    def init_factory(self):
        self.factory = Factory(bounds= Vector(self.bounds.x, self.bounds.y))
        self.factory.add_component(self, ComponentTypes.ASSEMBLER, Vector(3, 4), Direction.WEST)
        self.factory.add_component(self, ComponentTypes.ASSEMBLER, Vector(3, 5), Direction.WEST)

        self.factory.add_component(self, ComponentTypes.CONVEYOR, Vector(5, 5), Direction.EAST)
        self.factory.add_component(self, ComponentTypes.CONVEYOR, Vector(6, 5), Direction.NORTH)

    def init_resources(self):
        r1 = self.resource_map.place_resource(self, ResourceType.BLUE, Vector(5, 5))
        r2 = self.resource_map.place_resource(self, ResourceType.RED, Vector(6, 5))
        r3 = self.resource_map.place_resource(self, ResourceType.RED, Vector(2, 2))

        # r1.merge(r2)

    def draw(self, surface):
        # Draw the base 
        for x in range(self.bounds.x):
            for y in range(self.bounds.y):
                self.tiles[x][y].draw(surface)

        # Draw the factory components 
        self.factory.draw(surface)

        # Draw the resources
        self.resource_map.draw(surface)
            

    def update(self):
        self.factory.update(self)
        self.resource_map.update(self)

    def is_passable(self, position : Vector):
        # Is it in bounds
        if not is_in_bounds(position, ZERO_VECTOR, self.bounds):
            return False
        
        # Is there a wall 
        if not self.tiles[position.x][position.y].is_passable: 
            return False

        # Is there a component from the factory that is passable
        return self.factory.is_passable(position)
    
    def has_resource(self, position : Vector):
        return self.resource_map.has_resource(position)
    
    def get_resource(self, position: Vector):
        return self.resource_map.get_resource(position)