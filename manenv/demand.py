from __future__ import annotations

from abc import abstractmethod, ABC
from .vector import *

from .product_utils import * 
from .product import Product


class DemandSimulator(ABC):
    # Contains a list of product orders in queue
    def __init__(self):
        self._orders = []
    
    @abstractmethod
    def sample(self) -> Product:
        pass 

class DefaultDemandSimulator(DemandSimulator):
    def __init__(self):
        super.__init__()

        self._products_list : list[Product] = []

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

    def sample(self) -> Product:
        prod : Product = np.random.choice(self._products_list)
        return prod.copy()
    
    