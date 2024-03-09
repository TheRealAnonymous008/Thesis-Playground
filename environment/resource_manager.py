from pygame.surface import Surface
from .resource import *
from .vector import Vector
import numpy as np 


class ResourceMap:
    current_id = 0

    def __init__(self, bounds : Vector): 
        self.bounds = bounds 
        self.resources = [[None for _ in range(self.bounds.y)] for _ in range(self.bounds.x)]

    def place_resource(self, world, type : ResourceType, position : Vector ) -> ResourceTile:
        resource: ResourceTile = None
        match(type):
            case ResourceType.RED: 
                resource = RedResource(world, position )
            case ResourceType.BLUE:
                resource = BlueResource(world, position)
        
        resource.id = self.current_id
        self.current_id += 1
        self.resources[position.x][position.y] = resource 
        return resource

    def draw(self, surface : Surface):
        for row in self.resources:
            for rsrc in row: 
                if rsrc != None:
                    rsrc.draw(surface)

    def has_resource(self, position : Vector):
        if is_in_bounds(position, ZERO_VECTOR, self.bounds):
            return self.resources[position.x][position.y] != None
        
        return False 
    
    def get_resource(self, position : Vector):
        if not self.has_resource(position):
            return None 
        return self.resources[position.x][position.y] 
    
    def destroy_resource(self, position: Vector):
        if self.has_resource(position): 
            del self.resources[position.x][position.y]
            self.resources[position.x][position.y] = None
    
    def update(self, world):
        buffer = [[None for _ in range(self.bounds.y)] for _ in range(self.bounds.x)]

        for row in self.resources:
            for rsrc in row:
                if rsrc != None: 
                    rsrc.update(world)

        # Perform updates that are after. Also delete  resources as needed
        for row in self.resources:
            for rsrc in row:
                if rsrc != None: 
                    rsrc.post_update(world)             

        # Update the array to take into account the objects being in new positions
        for row in self.resources:
            for rsrc in row: 
                if rsrc != None:
                    if not rsrc.is_dead:
                        buffer[rsrc.position.x][rsrc.position.y] = rsrc  

        self.resources = buffer

    def get_mask(self):
        # The mask should contain not 
        mask = np.ndarray((self.bounds.x, self.bounds.y))
        for x in range(self.bounds.x):
            for y in range(self.bounds.y): 
                pos = Vector(x, y)
                rsrc : ResourceTile = self.get_resource(pos)
                if rsrc is None:
                    mask[x][y] = 0
                else: 
                    mask[x][y] = rsrc.type.value

        return mask

    def reset():
        id = 1