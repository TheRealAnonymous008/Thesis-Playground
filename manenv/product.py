from __future__ import annotations

from abc import abstractmethod, ABC
import numpy as np
import random


from .vector import *
from .asset_paths import AssetPath
from .product_utils import trim_structure_array

from typing import TYPE_CHECKING
if TYPE_CHECKING: 
    from .world import World, WorldCell

class Product:
    """
    A product produced and consumed by the smart factory. To simplify things, a product is specified using a matrix
    """
    _IDs : set = set()

    def __init__(self, structure : np.ndarray):
        """
        `structure` - an array that represents the product's structure. The array must be of datatype int. 
        """
        self._structure : np.ndarray = trim_structure_array(structure)
        self._id = random.getrandbits(64)
        Product._IDs.add(self._id)

    def delete(self):
        Product._IDs.remove(self._id)

    def copy(self):
        return Product(structure=self._structure)
    
    def __str__(self):
        return "id: " + str(self._id) + "\n" +  str(self._structure)