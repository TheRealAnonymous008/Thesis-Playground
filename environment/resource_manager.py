from pygame.surface import Surface
from .resource import *
from .vector import Vector

class ResourceMap:
    def __init__(self, bounds : Vector): 
        self.bounds = bounds 
        self.resources = [[None for _ in range(self.bounds.y)] for _ in range(self.bounds.x)]

    def place_resource(self, position : Vector, type : ResourceType):
        match(type):
            case ResourceType.RED: resource = RedResource(position )

        self.resources[position.x][position.y] = resource 

    def draw(self, surface : Surface):
        for row in self.resources:
            for rsrc in row: 
                if rsrc != None:
                    rsrc.draw(surface)