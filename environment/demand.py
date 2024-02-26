from .resource import *

class Order:
    def __init__(self):
        self.parts = dict(Vector, ResourceType)
    
    def add_part(self, type : ResourceType, position : Vector):
        self.parts[position] = type 
        return self
    
    def finalize(self):
        finalized_parts = dict(vars, ResourceType)
        min_vec : Vector = Vector(1000, 1000)

        for (k, v) in self.parts:
            k :Vector = k 
            min_vec.x = min(min_vec.x, k.x)
            min_vec.y = min(min_vec.y, k.y)

        for (k, v) in self.parts:
            finalized_parts[k - min_vec] = v 

        self.parts = finalized_parts


class DemandManager: 
    def __init__(self):
        self.orders : Order = []
        self.max_orders = 5
    
    def generate_order(self):
        order : Order = Order() 
        order.add_part(ResourceType.RED, Vector(0, 0)).finalize()

        self.orders.append(order)

    def check_order(self, order: Order):
        pass