from __future__ import annotations

from abc import abstractmethod, ABC
from ..utils.vector import *

from typing import TYPE_CHECKING
from ..utils.product_utils import * 
from .product import Product
from .demand import Order

if TYPE_CHECKING: 
    from .world import World


class ServiceModule(ABC):
    # Matches inventory with demand
    def __init__(self, matching_tolerance = 0.9):
        """
        The service module satisfies orders with inventory items.

        `matching_tolerance` defines a tolerance to use.
        """
        self._world : World = None 
        self._matching_tolerance = matching_tolerance
        self.reset()
        
    def bind(self, world : World):
        self._world = world 

    def reset(self):
        pass

    def update(self):
        time = self._world._time_step

        for order in self._world._demand._orders:
            best_product = None
            best_rating = -1
            for product in self._world._inventory._product_inventory.values():
                rating = self.get_matching_rating(order, product)
                if rating > best_rating:
                    best_rating = rating 
                    best_product = product
            
            if best_rating > self._matching_tolerance and best_product != None:
                self._world._demand.resolve_order(best_product, order, time)
                self._world._inventory.remove_product(best_product)


    @abstractmethod
    def get_matching_rating(self, order : Order, product : Product) -> float:
        """
        Override this function. It should return only a positive float
        """
        return 0
    
class DefaultServiceModule(ServiceModule):
    def __init__(self, matching_tolerance : int = 0.9):
        super().__init__(matching_tolerance)

    def get_matching_rating(self, order: Order, product: Product) -> float:
        return 1