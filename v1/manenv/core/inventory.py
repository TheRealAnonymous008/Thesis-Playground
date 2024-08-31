from __future__ import annotations

from abc import abstractmethod, ABC
from ..utils.vector import *

from ..utils.product_utils import * 
from .product import Product


class Inventory(ABC):
    # Keeps track of current stock 
    def __init__(self, max_storage = -1):
        """
        `max_storage` keeps track of inventory limits
        """

        self._product_inventory : dict[int, Product] = {}
        self._max_storage = max_storage

    def add_product(self, product : Product):
        if len(self._product_inventory) >= self._max_storage and self._max_storage > 0:
            return 
        
        self._product_inventory[product._id] = product

    def remove_product(self, product : Product) -> Product | None:
        if product._id in self._product_inventory:
            p = self._product_inventory.pop(product._id)
            return p 
        
        return None             

    def reset(self):
        self._product_inventory.clear()

    @abstractmethod
    def compute_storage_cost(self) -> int:
        return len(self._product_inventory)

class DefaultInventorySystem(Inventory):
    def __init__(self):
        super().__init__()
    
    def compute_storage_cost(self) -> int:
        for product in self._product_inventory.values():
            # Rationale: The cost of storing this product is proportional to its size 
            x, y = product._structure.shape
            return x * y 