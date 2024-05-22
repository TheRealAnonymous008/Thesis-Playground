from __future__ import annotations

from abc import abstractmethod, ABC
from ..utils.vector import *

from ..utils.product_utils import * 
from .product import Product


class Inventory(ABC):
    # Keeps track of current stock 
    def __init__(self):
        self._product_inventory : dict[int, Product] = {}
        self._cost = 0

    def add_product(self, product : Product):
        self._product_inventory[product._id] = product
        self._cost += self.compute_storage_cost(product)

    def remove_product(self, id : int) -> Product | None:
        if id in self._product_inventory:
            p = self._product_inventory.pop(id)
            self._cost -= self.compute_storage_cost(p)
            return p 
        
        return None             

    def reset(self):
        self._cost = 0
        self._product_inventory.clear()

    @abstractmethod
    def compute_storage_cost(self, product : Product) -> int:
        return 1

class DefaultInventorySystem(Inventory):
    def __init__(self):
        super().__init__()
    
    def compute_storage_cost(self, product: Product) -> int:
        x, y = product._structure.shape
        return x * y 