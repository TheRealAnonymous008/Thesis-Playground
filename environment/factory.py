from . import components as cmp 
from .vector import Vector
from .direction import Direction

class Factory:
    def __init__(self, bounds):
        self.components : list(list(cmp.FactoryComponent)) = [[None for _ in range(bounds.y)] for _ in range(bounds.x)]
        self.assemblers = []
        self.bounds : Vector = bounds

    def add_component(self, component : cmp.FactoryComponent, world, position : Vector, rotation : Direction):
        self.components[position.x][position.y] = component 
        component.update_transform(world, position, rotation)

    def add_assembler(self, world, position : Vector, rotation : Direction):
        assembler = cmp.Assembler(position, world, rotation)
        self.assemblers.append(assembler)
        self.add_component(assembler, world, position, rotation)

    def update(self, world):
        buffer = [[None for _ in range(self.bounds.y)] for _ in range(self.bounds.x)]

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
        component : cmp.FactoryComponent= self.components[position.x][position.y]

        if component is None:
            return True 
        
        return component.is_passable