from . import components as cmp 
from .vector import Vector
from .direction import Direction
import numpy as np 

class Factory:
    def __init__(self, bounds):
        self.components = [[None for _ in range(bounds.y)] for _ in range(bounds.x)]
        self.assemblers = [[None for _ in range(bounds.y)] for _ in range(bounds.x)]
        self.bounds : Vector = bounds

    def add_component(self, world, type : cmp.ComponentType, position : Vector, arg):
        component = None 
        match(type):
            case cmp.ComponentType.ASSEMBLER: 
                component = cmp.Assembler(world, position, arg)
                self.assemblers[position.x][position.y] = component
            case cmp.ComponentType.CONVEYOR:
                component = cmp.ConveyorBelt(world, position ,arg)
                self.components[position.x][position.y] = component 
            case cmp.ComponentType.SPAWNER: 
                component = cmp.Spawner(world, position, arg)
                self.components[position.x][position.y] = component
            case cmp.ComponentType.OUTPORT:
                component = cmp.OutPort(world, position)
                self.components[position.x][position.y] = component



        component.update_transform(world, position, arg)

    def update(self, world):
        self.update_components(world)
        self.update_assemblers(world)

    def update_components(self, world):
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

    def update_assemblers(self, world):
        buffer = [[None for _ in range(self.bounds.y)] for _ in range(self.bounds.x)]

        for row in self.assemblers:
            for comp in row: 
                if comp != None: 
                    comp.update(world)

        # Update the array to take into account the objects being in new positions
        for row in self.assemblers:
            for comp in row: 
                if comp != None:
                    buffer[comp.position.x][comp.position.y] = comp 

        self.assemblers = buffer

    def draw(self, surface):
        for row in self.components:
            for comp in row:
                if comp != None:
                    comp.draw(surface)

        for row in self.assemblers:
            for comp in row:
                if comp != None:
                    comp.draw(surface)

    def has_component(self, position : Vector):
        return self.components[position.x][position.y] != None or self.assemblers[position.x][position.y]
    
    def get_component(self, position):
        if self.has_component(position):
            return self.components[position.x][position.y]
        return None
    
    def is_passable(self, position : Vector):
        component : cmp.FactoryComponent= self.components[position.x][position.y]
        assembler : cmp.Assembler = self.assemblers[position.x][position.y]

        if component is None and assembler is None:
            return True 
        elif component is not None: 
            return component.is_passable 
        else:
            return assembler.is_passable
    
    def get_mask(self):
        mask = np.ndarray((self.bounds.x, self.bounds.y))
        for x in range(self.bounds.x):
            for y in range(self.bounds.y): 
                pos = Vector(x, y)
                comp : cmp.FactoryComponent = self.get_component(pos)
                if comp is None:
                    mask[x][y] = 0
                else: 
                    mask[x][y] = comp.type.value

        return mask