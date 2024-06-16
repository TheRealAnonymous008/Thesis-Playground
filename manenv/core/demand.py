from __future__ import annotations

from abc import abstractmethod, ABC

from manenv.core.idpool import IDPool
from ..utils.vector import *

from ..utils.product_utils import * 
from .product import Product
import random


class Order:
    def __init__(self, issued, product, due_date = -1):
        """
        An order consists of a `product` and an expected `due date`. If the due date is -1, we 
        ignore it when computing the lateness of the job
        """
        self._product : Product = product
        self._due_date = due_date
        self._issue_date = issued

        self._satisfied_time : int = -1

        self._id = IDPool.get()

    def satisfy(self, time):
        IDPool.pop(self._id)
        self._satisfied_time = time 

    def get_cycle_time(self):
        if self.is_satisfied():
            return self._satisfied_time - self._issue_date

        return 0 
    
    def is_satisfied(self):
        return self._satisfied_time > 0

    def __str__(self):
        return "Product: " + str(self._product)


class DemandSimulator(ABC):
    # Contains a list of product orders in queue
    def __init__(self, max_orders = -1):
        """
        max_orders defines the maximum number of orders in queue (-1 implies unbounded queue)
        """
        self._orders : dict[int, Order] = {}
        self._max_orders = max_orders

    def update(self):
        # Remove any resolved orders
        ids = []
        for order in self._orders.values():
            if order.is_satisfied():
                ids.append(order._id)

        for id in ids: 
            self._orders.pop(id)

        if self._max_orders > 0 and len(self._orders) >= self._max_orders:
            return 
        
        p = self.sample()
        if p == None:
            return 
        
        order = Order(self._current_time, p)
        self._orders[order._id] = order
        self._current_time += 1

    def resolve_order(self, product : Product, order : Order) -> float:
        """
        returns a float representing the level of customer satisfaction
        """
        if not order._id in self._orders:
            return 0 

        # lateness = np.max(0, time - order.due_date)
        order.satisfy(self._current_time)

        return 0
        
    def reset(self):
        self._orders.clear()
        self._current_time = 0
        
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
        return Order(self._current_time, prod.copy())
    
    