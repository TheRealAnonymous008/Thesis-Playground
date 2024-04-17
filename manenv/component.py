from __future__ import annotations

from abc import abstractmethod, ABC
from .vector import *
from .asset_paths import AssetPath
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING: 
    from .product import Product
    from .world import World, WorldCell

class FactoryComponent(ABC): 
    """
    Abstract class for factory components
    """
    def __init__(self, asset : str = ""):
        """
        `world` - the world instance that holds this factory component 

        `asset` - a path to the image associated with this asset. If empty, defaults to no asset. 
        """
        self._cell : WorldCell | None = None 
        self._world : World = None
        self._asset : str = asset

    def bind(self, world):
        self._world = world 

    def place(self, cell: WorldCell):
        self._check_is_bound()
        if self._cell == cell:
            return 
        
        if self._cell != None: 
            self._cell.place_component(None)

        self._cell = cell 
        self._cell.place_component(self)

    @abstractmethod
    def update(self):
        self._check_is_bound()
        pass 

    def _check_is_bound(self):
        if self._world == None: 
            raise Exception("World is not initialized for this component ")

class Spawner(FactoryComponent):
    """
    Spawns product objects based on a provided template product  
    """
    def __init__(self, product : Product):
        """
        product: The product that this spawner will spawn 
        """
        super().__init__(AssetPath.SPAWNER)
        self._product = product
    
    def update(self):
        if (len(self._cell._products)) == 0:
            self._cell.place_product(self._product.copy())