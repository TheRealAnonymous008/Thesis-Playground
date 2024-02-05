from .direction import Direction

class FactoryComponent: 
    def __init__(self):
        self.position : (int, int) = (0, 0)
        self.rotation : Direction = Direction.EAST

class Assembler(FactoryComponent):
    pass 