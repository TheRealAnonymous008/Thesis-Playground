from manenv.components.assembler import Assembler
from manenv.core.demand import Order


class AssemblerSelectorModel:
    def __init__(self, assembler : Assembler):
        self._assembler = assembler
        
    def think(self, demand : list[Order]): 
        if not self._assembler.can_take_order():
            return 
        
        best_order : Order = None 
        for order in demand: 
            # TODO: Establish model to choose the best order to do. For now this works.
            best_order = order 
            break 

        self._assembler.add_order_to_queue(best_order)
        self._assembler.set_current_order(best_order)

