from __future__ import annotations
from typing import Tuple

from abc import abstractmethod, ABC
from ..utils.vector import *
from ..asset_paths import AssetPath
import numpy as np

from typing import TYPE_CHECKING
from ..utils.product_utils import * 
from .product import Product
if TYPE_CHECKING: 
    from .world import World, WorldCell
    from .effector import Effector

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

        self._on_place()

    def _on_place(self):
        pass 

    @abstractmethod
    def update(self):
        self._check_is_bound()
        pass 

    @abstractmethod
    def reset(self):
        pass 

    def _check_is_bound(self):
        if self._world == None: 
            raise Exception("World is not initialized for this component ")

