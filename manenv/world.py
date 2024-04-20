from __future__ import annotations
import numpy as np 
from typing import Tuple

from typing import TYPE_CHECKING
if TYPE_CHECKING: 
    from .component import *
    from .product import *

from .vector import * 

class WorldCell: 
    """
    The basic unit within the factory. The world cell may contain:
    - a factory component
    - a product
    - a robot 
    """

    def __init__(self, position : Vector):
        """
        `position` - the position associated with this cell
        """

        self._factory_component : FactoryComponent | None = None 
        self._products : list[Product] = [] 
        # is_placed stores a series of booleans corresponding to whether or not a product is at the edge
        # of this cell or not 

        self._dirty_set : set[int] = set()
        self._position : Vector = position
        self.reset()

    def reset(self):
        self._factory_component = None 
        pass 

    def place_component(self, cmp : FactoryComponent):
        if self._factory_component == cmp: 
            return
        self._factory_component = cmp
        cmp.place(self)

    def place_product(self, product: Product, position : Vector = ZERO_VECTOR):
        self._products.append(product)
        if position[0] != 0 or position[1] != 0:
            self._dirty_set.add(product._id)

    def is_product_placed(self, product : Product) -> bool:
        return not (product._id in self._dirty_set)
    
    def update_place_status(self, product : Product):
        if product._id in self._dirty_set: 
            self._dirty_set.remove(product._id) 

    def remove_product(self, product : Product):
        """
        Remove product with specified `id` from the products on this cell
        """
        idx = 0
        for _product in self._products:
            if _product._id == product._id and self.is_product_placed(product):
                self._products.remove(_product)
                break
            idx += 1

class World: 
    """
    Contains information about the smart factory environment 
    """
    def __init__(self, shape : Tuple): 
        """
        `shape` - the dimensions of the environment in (width, height) format 
        """
        self._shape : Tuple = shape 
        self._map : list[list[WorldCell]] = [[WorldCell(position=make_vector(x, y)) for y in range(shape[1])] for x in range(shape[0])]
        self._time_step : int  = 0 

    def _width(self):
        return self._shape[0]
    
    def _height(self):
        return self._shape[1]
    
    def reset(self):
        self._time_step = 0

        for M in self._map: 
            for r in M: 
                r.reset()

    def update(self): 
        self._time_step += 1
        # All dirty components are pushed to the cell's center
        for x in range(self._shape[0]):
            for y in range(self._shape[1]):
                cell = self._map[x][y]
                for prod in cell._products:
                    if not cell.is_product_placed(prod):
                        cell.update_place_status(prod)

        # Update all components
        for x in range(self._shape[0]):
            for y in range(self._shape[1]):
                if self._map[x][y]._factory_component != None: 
                    self._map[x][y]._factory_component.update()
    
    def get_cell(self, v : Vector) -> WorldCell | None: 
        if v[0] < 0 or v[1] < 0 or v[0] >= self._shape[0] or v[1] >= self._shape[1]:
            return None 
        
        return self._map[v[0]][v[1]]
    
    def place_component(self, pos: Vector, cmp : FactoryComponent):
        cmp.bind(self)
        self.get_cell(pos).place_component(cmp)
        
    def place_product(self, pos: Vector, product : Product):
        self.get_cell(pos).place_product(product)
    