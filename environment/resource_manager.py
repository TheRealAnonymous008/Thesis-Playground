from .resource import ResourceTile
from .vector import Vector

class ResourceMap:
    def __init__(self, bounds : Vector): 
        self.bounds = bounds 
        self.resources = [[None for _ in range(self.bounds.y)] for _ in range(self.bounds.x)]

    def place_resource(self, position : Vector, resource : ResourceTile):
        self.resources[position.x][position.y] = resource 