import numpy as np
from manenv.core.component import FactoryComponent
from manenv.utils.vector import *

from manenv.components import Assembler

class Conveyor(FactoryComponent): 
    """
    System for transporting products across the factory. 
    """
    def __init__(self, weights : np.ndarray):
        """
        `weights`: a 3x3 transition matrix that represent how a product will flow through this cell. 
        The middle entry is ignored 

        An integer n means that n of that product will be allowed first from that side (if applicable)
        Positive n means the product is moved away from the cell
        Negative n means the product is moved towards the cell. 

        """
        assert(weights.shape == (3, 3))
        self._weights = weights

        """
        curr in and curr out follow numpad order
        1 2 3
        4 5 6
        7 8 9
        """

        super().__init__("")

    def _on_place(self):
        self._initialize_in_out_ports()

    def _initialize_in_out_ports(self):
        self._curr_in = -1
        self._curr_out = -1
        self._inports = []
        self._outports = []

        for y in range(0, 3):
            for x in range(0, 3):
                idx = 3 * y + x
                offset = make_vector(x - 1, y - 1)
                if self._world.get_cell(offset + self._cell._position) == None:
                    continue 
                if self._weights[x][y] < 0:
                    self._inports.append(idx)
                elif self._weights[x][y] > 0:
                    self._outports.append(idx)

        np.random.shuffle(self._inports)
        np.random.shuffle(self._outports)
    
    def update(self): 
        super().update() 
        if len(self._inports) > 0: 
            self._curr_in = (self._curr_in + 1) % len(self._inports)
            idx = self._inports[self._curr_in]
            input_offset = make_vector(idx % 3 - 1, idx // 3 - 1)
            input_vec = self._cell._position + input_offset
            src_cell = self._world.get_cell(input_vec)

            # Move product into this cell
            if src_cell != None:
                for product in src_cell.get_product_list():
                    if self._cell.is_full():
                        break 

                    if isinstance(src_cell._factory_component, Assembler): 
                        # src_cell.remove_product(product)
                        self._cell.place_product(product)
                    elif self._cell.is_product_placed(product):
                        src_cell.remove_product(product)
                        self._cell.place_product(product, input_offset)

        if len(self._outports) > 0:
            self._curr_out = (self._curr_out + 1) % len(self._outports)
            idx = self._outports[self._curr_out]
            output_offset = make_vector(idx % 3 - 1, idx // 3 - 1)
            output_vec = self._cell._position + output_offset
            dest_cell = self._world.get_cell(output_vec)
            
            # Move current product out of this cell 
            if dest_cell != None: 
                P = self._cell.get_product_list()
                for product in P:
                    if self._cell.is_product_placed(product):
                        if isinstance(dest_cell._factory_component, Assembler):
                            dest_cell._factory_component.place_in_inventory(product)
                        else:
                            if dest_cell.is_full():
                                break
                            dest_cell.place_product(product, -output_offset)
                        
                        self._cell.remove_product(product)


    def reset(self):
        pass 