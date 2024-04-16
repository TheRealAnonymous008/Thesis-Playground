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

    def place_product(self, product: FactoryComponent):
        self._products.append(product)
    
    def remove_product(self, id : int):
        """
        Remove product with specified `id` from the products on this cell
        """
        for product in self._products:
            if product._id == id:
                self._products.remove(product)
                break

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

    def _width(self):
        return self._shape[0]
    
    def _height(self):
        return self._shape[1]
    
    def reset(self):
        for M in self._map: 
            for r in M: 
                r.reset()

    def update(self): 
        # Update all components
        for x in range(self._shape[0]):
            for y in range(self._shape[1]):
                self._map[x][y]._factory_component.update(self)
    
    def get_cell(self, v : Vector) -> WorldCell | None: 
        if v[0] < 0 or v[1] < 0 or v[0] >= self._shape[0] or v[1] >= self._shape[1]:
            return None 
        
        return self._map[v[0]][v[1]]
    
    def place_component(self, pos: Vector, cmp : FactoryComponent):
        cmp.bind(self)
        self.get_cell(pos).place_component(cmp)
        
    def place_product(self, pos: Vector, product : Product):
        self.get_cell(pos).place_product(product)
    