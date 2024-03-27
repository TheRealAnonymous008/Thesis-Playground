from .resource import *
from .constants import * 
from .order import *

def compare_orders(x: Order, y: Order) -> float:
    xset = set(x.parts.items())
    yset = set(y.parts.items())

    union = len(xset.union(xset, yset))
    intersection = len(xset.intersection(xset, yset))

    return intersection / union
            


class DemandManager: 
    def __init__(self):
        self.products : ProductListing = ProductListing()
        self.orders : list(Order) = [] 

    def reset(self):
        self.orders = []
        self.orders.append(self.products.get_product(0))

    def check_order(self, order: Order):
        dist = BIG_NUMBER
        order_to_remove = None 

        for ord in self.orders:
            distance = compare_orders(ord, order)
            if distance < dist:
                dist = distance 
                order_to_remove = ord 

        if order_to_remove != None: 
            return dist
        
        return 0