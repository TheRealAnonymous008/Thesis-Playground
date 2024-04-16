from __future__ import annotations

from abc import abstractmethod, ABC
from .vector import *
from .asset_paths import AssetPath
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING: 
    from .world import World, WorldCell

class FactoryComponent(ABC): 
    """
    Abstract class for factory components
    """
    def __init__(self, world : World, asset : str = ""):
        """
        `world` - the world instance that holds this factory component 

        `asset` - a path to the image associated with this asset. If empty, defaults to no asset. 
        """
        self._cell : WorldCell | None = None 
        self._world : World = world
        self._asset : str = asset

    def place(self, cell: WorldCell):
        if self._cell == cell:
            return 
        
        if self._cell != None: 
            self._cell.place_component(None)

        self._cell = cell 
        self._cell.place_component(self)

    @abstractmethod
    def update(self):
        pass 

class Spawner(FactoryComponent):
    def __init__(self, world : World):
        super().__init__(world, AssetPath.SPAWNER)
    
    def update(self):
        pass