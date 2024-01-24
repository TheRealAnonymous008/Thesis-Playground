import numpy as np 

import environment as env

class Industry: 
    def __init__(self, id):
        self.id = id 
        self.reset()

    def report(self):
        return {
            "demand": self.demand,
            "supply": self.supply,
        }
    
    def add_product(self, product):
        self.supply.append(product)
    
    def reset(self):
        self.demand = 0
        self.supply : list[env.product.Product]= []

    def update(self):
        pass