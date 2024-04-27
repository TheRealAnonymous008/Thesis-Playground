from __future__ import annotations
from typing import Tuple
from enum import Enum

from abc import abstractmethod, ABC
from .vector import *
import numpy as np
from .asset_paths import AssetPath

from typing import TYPE_CHECKING
if TYPE_CHECKING: 
    from .component import Assembler
    from .product import Product

class Effector(ABC):
    """
    An effector models the actions possible by a robot assembly agent .
    """
    def __init__(self, action_space : Enum, asset = "", position = VectorBuiltin.ZERO_VECTOR):
        """
        `action_space` - defines the list of actions this Effector type can perform
        
        `asset` - the path to the asset to render this effector

        `position` - the location of the effector head
        
        """
        self._assembler = None 
        self._action_space = action_space

        self._position = position
        self._asset = asset
    
    def bind(self, assembler : Assembler):
        self._assembler = assembler
        self._workspace_size = self._assembler._workspace_size - make_vector(1, 1)

    def is_bound(self):
        return self._assembler != None
    
    @abstractmethod
    def execute_action(self, action_code : int):
        pass 


class GrabberActions(Enum):
    IDLE = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_FORWARD = 3
    MOVE_BACKWARD = 4
    GRAB = 5
    RELEASE = 6 
    GRAB_INVENTORY = 7
    DISCARD = 8
    ROTATE_CW = 9
    ROTATE_CCW = 10

class Grabber(Effector):
    def __init__(self):
        super().__init__(GrabberActions, AssetPath.GRABBER)
        self._grabbed_product = None

    def execute_action(self, action_code: int):
        match(action_code):
            case GrabberActions.IDLE:
                pass 

            case GrabberActions.MOVE_LEFT:
                self._position += VectorBuiltin.LEFT
                self._position = np.clip(self._position, (0, 0), self._workspace_size)
            
            case GrabberActions.MOVE_RIGHT:
                self._position += VectorBuiltin.RIGHT
                self._position = np.clip(self._position, (0, 0), self._workspace_size)

            case GrabberActions.MOVE_BACKWARD:
                self._position += VectorBuiltin.BACKWARD
                self._position = np.clip(self._position, (0, 0), self._workspace_size)

            case GrabberActions.MOVE_FORWARD:
                self._position += VectorBuiltin.FORWARD
                self._position = np.clip(self._position, (0, 0), self._workspace_size)

            case GrabberActions.GRAB:
                self._grabbed_product = self._assembler.get_product_in_workspace(self._position)

            case GrabberActions.RELEASE:
                if self._grabbed_product != None:
                    self._assembler.place_in_workspace(self._grabbed_product, self._position)
                    self._grabbed_product = None 
            
            case GrabberActions.GRAB_INVENTORY:
                inventory = self._assembler.get_product_inventory()
                if len(inventory) > 0:
                    self._grabbed_product = inventory.pop(0)

            case GrabberActions.DISCARD:
                self._grabbed_product = None 

            case GrabberActions.ROTATE_CW:
                if self._grabbed_product != None:
                    self._grabbed_product.rotate(1)
                
            case GrabberActions.ROTATE_CCW:
                if self._grabbed_product != None:
                    self._grabbed_product.rotate(-1)
                    
            case _: 
                pass 