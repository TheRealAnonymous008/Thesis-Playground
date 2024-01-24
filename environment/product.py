import numpy as np 

class Product:
    def __init__(self, quantity: int = 1, quality: np.array = [], price: float = 1):
        self.quantity = quantity 
        self.quality = quality 
        self.price = price

    def __str__(self):
        return f"qty={self.quantity}, quality={self.quality}, price={self.price}"