from .components import * 
from .factory import Factory
from .vector import Vector, ZERO_VECTOR, is_in_bounds
from .direction import Direction
from .world_tile import *
from .resource_manager import ResourceMap
from .resource import ResourceType
from .demand import * 

import numpy as np 

class World:
    def __init__(self, width, height, block_size):
        self.bounds : Vector = Vector(width, height)
        self.block_size = block_size

        self.tiles = [[None for _ in range(height)] for _ in range(width)]
        self.resource_map  : ResourceMap = ResourceMap(self.bounds)

        self.demand_manager : DemandManager = DemandManager()

        self.init_tiles()
        self.init_resources()
        self.init_factory()
        self.init_demand()

        
    def init_tiles(self):
        for x in range(self.bounds.x):
            for y in range(self.bounds.y):
                self.tiles[x][y] = EmptyTile(self, Vector(x, y))
    
        self.tiles[3][7] =  WallTile(self, Vector(3, 7))

    def init_resources(self):
        r1 = self.place_resource(ResourceType.RED, Vector(0, 0))
        r2 = self.place_resource(ResourceType.RED, Vector(0, 1))
        r1.merge(r2)

    def init_factory(self):
        self.factory = Factory(bounds= Vector(self.bounds.x, self.bounds.y))
        self.factory.add_component(self, ComponentType.ASSEMBLER, Vector(3, 4), Direction.WEST)

        self.factory.add_component(self, ComponentType.SPAWNER, Vector(4, 5), ResourceType.RED)
        self.factory.add_component(self, ComponentType.OUTPORT, Vector(1, 1), ResourceType.RED)

        self.factory.add_component(self, ComponentType.CONVEYOR, Vector(5, 5), Direction.EAST)
        self.factory.add_component(self, ComponentType.CONVEYOR, Vector(6, 5), Direction.EAST)
        self.factory.add_component(self, ComponentType.CONVEYOR, Vector(7, 5), Direction.NORTH)

    def init_demand(self):
        self.demand_manager.generate_order()
        self.demand_manager.generate_order()
        self.demand_manager.generate_order()
        self.demand_manager.generate_order()
        self.demand_manager.generate_order()

    def place_resource(self, resource_type : ResourceType, position : Vector):
        return self.resource_map.place_resource(self, resource_type, position)

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
    
    def get_object_at(self, position : Vector):
        # Is it in bounds
        if not is_in_bounds(position, ZERO_VECTOR, self.bounds):
            return None
        
        # Is there a wall 
        if not self.tiles[position.x][position.y].is_passable: 
            return self.tiles[position.x][position.y]
        
        comp = self.factory.get_component(position)
        if comp != None:
            return comp
        
        rsrc = self.resource_map.get_resource(position)
        if rsrc != None:
            return rsrc 
        
        return None

    
    def has_resource(self, position : Vector):
        return self.resource_map.has_resource(position)
    
    def get_resource(self, position: Vector):
        return self.resource_map.get_resource(position)
    
    def move_resource(self, position: Vector):
        self.resource_map.request_move(self, position)
    
    def submit_resources(self, rsrcs : list[ResourceTile]):
        order  = Order()
        for rsrc in rsrcs:
            order.add_part(rsrc.type, rsrc.position)
        order.finalize()

        self.demand_manager.check_order(order)

    def get_state(self):
        state = {}
        state["world_mask"] = self.get_mask()
        state["resource_mask"] = self.resource_map.get_mask()
        state["factory_mask"] = self.factory.get_mask()

        return state 
    
    def get_mask(self):
        mask = np.ndarray((self.bounds.x, self.bounds.y))
        for x in range(self.bounds.x):
            for y in range(self.bounds.y): 
                mask[x][y] = 0 if type(self.tiles[x][y]) is EmptyTile else 1

        return mask