from __future__ import annotations

from abc import abstractmethod, ABC
import numpy as np

from .vector import *
from .asset_paths import AssetPath
from .utils import trim_structure_array

from typing import TYPE_CHECKING
if TYPE_CHECKING: 
    from .world import World, WorldCell

class Product:
    """
    A product produced and consumed by the smart factory. To simplify things, a product is specified using a matrix
    """
    def __init__(self, structure : np.ndarray):
        """
        `structure` - an array that represents the product's structure. The array must be of datatype int. 
        """
        self._structure : np.ndarray = trim_structure_array(structure)
    
    def __str__(self):
        return str(self._structure)


