from __future__ import annotations

from abc import abstractmethod, ABC
from ..utils.vector import *

from ..utils.product_utils import * 
from .product import Product

class Order:
    def __init__(self, product):
        self._product : Product = product

    def __str__(self):
        return "Product: " + str(self._product)


class DemandSimulator(ABC):
    # Contains a list of product orders in queue
    def __init__(self, max_orders = -1):
        """
        max_orders defines the maximum number of orders in queue (-1 implies unbounded queue)
        """
        self._orders : list[Order] = []
        self._max_orders = max_orders

    def update(self):
        if self._max_orders > 0 and len(self._orders) >= self._max_orders:
            return 
        
        p = self.sample()
        if p == None:
            return 
        
        self._orders.append(p)
        
    def reset(self):
        self._orders.clear()
        
    @abstractmethod
    def sample(self) -> Order | None:
        pass 

class DefaultDemandSimulator(DemandSimulator):
    def __init__(self, max_orders = -1, p = 0.9):
        """
        p denotes the probability of not generating an order at the current tick
        """

        super().__init__(max_orders)

        self._products_list : list[Product] = []
        self._p = p

        # Product 1
        s = np.zeros((5, 5), dtype=np.int8)
        s[1][2] = 2
        s[1][1] = 2
        s[2][1] = 1
        s[3][1] = 1

        self._products_list.append(Product(s))
        
        # Product 2
        s = np.zeros((5, 5), dtype=np.int8)
        s[1][2] = 1
        s[1][1] = 1
        s[2][1] = 2
        s[3][1] = 2

        self._products_list.append(Product(s))

    def sample(self) -> Order:
        q = np.random.ranf()
        if q < self._p:
            return None 
        
        prod : Product = np.random.choice(self._products_list)
        return Order(prod.copy())
    
    