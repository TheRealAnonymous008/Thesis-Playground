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

    def update(self):
        for components in self.components:
            components.update()

    def render(self, surface):
        for row in self.components:
            for comp in row:
                if comp != None:
                    comp.render(surface)