import numpy as np 

class Industry: 
    def __init__(self, id):
        self.id = id 
        self.reset()

    def report(self):
        return {
            "demand": self.demand,
            "supply": self.supply,
        }
    
    def reset(self):
        self.demand = 0
        self.supply = 0