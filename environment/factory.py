from . import components as cmp 

class Factory:
    def __init__(self):
        self.components = []
        self.assemblers = []

    def add_component(self, component : cmp.FactoryComponent):
        self.components.append(component)

    def add_assembler(self, assembler : cmp.Assembler):
        self.assemblers.append(assembler)
        self.components.append(assembler)

    def update(self):
        for components in self.components:
            self.components.update()