
from environment.constants import BIG_NUMBER
from environment.resource import ResourceType
from environment.vector import Vector


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


class ProductListing: 
    products = []
    def __init__(self):
        self.products = []
        self._generate_products()

    def _generate_products(self):
        order : Order = Order() 
        order.add_part(ResourceType.RED, Vector(0, 0)).finalize()
        self.products.append(order)

    def get_product(self, idx):
        return self.products[idx]