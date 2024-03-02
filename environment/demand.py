from .resource import *
from .constants import * 

class Order:
    def __init__(self):
        self.parts : dict[Vector, ResourceType] = {}

    def add_part(self, type : ResourceType, position : Vector):
        self.parts[position] = type 
        return self
    
    def finalize(self):
        finalized_parts : dict[Vector, ResourceType] = {}
        min_vec : Vector = Vector(BIG_NUMBER, BIG_NUMBER)

        for k in self.parts.keys():
            k :Vector = k 
            min_vec.x = min(min_vec.x, k.x)
            min_vec.y = min(min_vec.y, k.y)

        for k in self.parts.keys():
            finalized_parts[k - min_vec] = self.parts[k]

        self.parts = finalized_parts

def compare_orders(x: Order, y: Order) -> float:
    xset = set(x.parts.items())
    yset = set(y.parts.items())

    union = len(xset.union(xset, yset))
    intersection = len(xset.intersection(xset, yset))

    return intersection / union
            


class DemandManager: 
    def __init__(self):
        self.orders : list(Order) = []
    
    def generate_order(self):
        order : Order = Order() 
        order.add_part(ResourceType.RED, Vector(0, 0)).finalize()

        self.orders.append(order)

    def check_order(self, order: Order):
        dist = 100000000000000000
        order_to_remove = None 

        for ord in self.orders:
            distance = compare_orders(ord, order)
            if distance < dist:
                dist = distance 
                order_to_remove = ord 

        if order_to_remove != None: 
            print(dist)
            self.orders.remove(order_to_remove)