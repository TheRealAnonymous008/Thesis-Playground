from . import components as cmp 
from .vector import Vector
from .direction import Direction
from .constants import BOUNDS

class Factory:
    def __init__(self):
        self.components : list(list(cmp.FactoryComponent)) = [[None for _ in range(BOUNDS.y)] for _ in range(BOUNDS.x)]
        self.assemblers = []

    def add_component(self, component : cmp.FactoryComponent, position : Vector, rotation : Direction):
        self.components[position.x][position.y] = component 
        component.update_transform(position, rotation)

    def add_assembler(self, assembler : cmp.Assembler, position : Vector, rotation : Direction):
        self.assemblers.append(assembler)
        self.add_component(assembler, position, rotation)

    def update(self, world):
        buffer = [[None for _ in range(BOUNDS.y)] for _ in range(BOUNDS.x)]

        for row in self.components:
            for comp in row: 
                if comp != None: 
                    comp.update(world)

        # Update the array to take into account the objects being in new positions
        for row in self.components:
            for comp in row: 
                if comp != None:
                    buffer[comp.position.x][comp.position.y] = comp 

        self.components = buffer

    def render(self, surface):
        for row in self.components:
            for comp in row:
                if comp != None:
                    comp.render(surface)

    def has_component(self, position : Vector):
        return self.components[position.x][position.y] != None
    
    def is_passable(self, position : Vector):
        cmp = self.components[position.x][position.y]

        if cmp is None:
            return True 
        
        return cmp.is_passable