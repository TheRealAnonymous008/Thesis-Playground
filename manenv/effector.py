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
                self._position = np.clip(self._position, VectorBuiltin.ZERO_VECTOR, self._assembler._workspace_size)

            case GrabberActions.MOVE_RIGHT:
                self._position += VectorBuiltin.RIGHT
                self._position = np.clip(self._position, VectorBuiltin.ZERO_VECTOR, self._assembler._workspace_size)

            case GrabberActions.MOVE_BACKWARD:
                self._position += VectorBuiltin.BACKWARD
                self._position = np.clip(self._position, VectorBuiltin.ZERO_VECTOR, self._assembler._workspace_size)

            case GrabberActions.MOVE_FORWARD:
                self._position += VectorBuiltin.RIGHT
                self._position = np.clip(self._position, VectorBuiltin.ZERO_VECTOR, self._assembler._workspace_size)

            case GrabberActions.GRAB:
                pass 

            case GrabberActions.RELEASE:
                pass 
            
            case GrabberActions.GRAB_INVENTORY:
                pass 

            case _: 
                pass 