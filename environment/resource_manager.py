from pygame.surface import Surface
from .resource import *
from .vector import Vector

class ResourceMap:
    def __init__(self, bounds : Vector): 
        self.bounds = bounds 
        self.resources = [[None for _ in range(self.bounds.y)] for _ in range(self.bounds.x)]

    def place_resource(self, world, type : ResourceType, position : Vector ):
        resource: ResourceTile = None
        match(type):
            case ResourceType.RED: 
                resource = RedResource(world, position )
            case ResourceType.BLUE:
                resource = BlueResource(world, position)

        self.resources[position.x][position.y] = resource 

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
    
    def update(self, world):
        buffer = [[None for _ in range(self.bounds.y)] for _ in range(self.bounds.x)]

        for row in self.resources:
            for rsrc in row:
                if rsrc != None: 
                    rsrc.update(world)

        # Perform updates that ar after
        for row in self.resources:
            for rsrc in row:
                if rsrc != None: 
                    rsrc.post_update(world)             

        # Update the array to take into account the objects being in new positions
        for row in self.resources:
            for rsrc in row: 
                if rsrc != None:
                    buffer[rsrc.position.x][rsrc.position.y] = rsrc  

        self.resources = buffer
