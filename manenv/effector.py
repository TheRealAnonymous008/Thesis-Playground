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
    def __init__(self, action_space : Enum, asset = "", position = make_vector(0, 0)):
        """
        `action_space` - defines the list of actions this Effector type can perform
        
        `asset` - the path to the asset to render this effector

        `position` - the location of the effector head
        
        """
        self._assembler = None 
        self._action_space = action_space

        self._position = position
        self._asset = asset

        self._current_action = None 
    
    def bind(self, assembler : Assembler):
        self._assembler = assembler
        self._workspace_size = self._assembler._workspace_size - make_vector(1, 1)

    def is_bound(self):
        return self._assembler != None
    
    def set_action(self, action_code: int):
        self._current_action = action_code

    @abstractmethod
    def _preupdate(self):
        pass 

    @abstractmethod
    def _update(self):
        pass 

    @abstractmethod
    def _postupdate(self):
        self._current_action = None 


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
    def __init__(self, position : Vector = None):
        super().__init__(GrabberActions, AssetPath.GRABBER, position)
        self._grabbed_product = None

    def _preupdate(self):
        super()._preupdate()

        match(self._current_action):
            case GrabberActions.IDLE:
                pass 

            case GrabberActions.MOVE_LEFT:
                if self._grabbed_product != None:
                    self._grabbed_product.add_vel(VectorBuiltin.LEFT)
            
            case GrabberActions.MOVE_RIGHT:
                if self._grabbed_product != None:
                    self._grabbed_product.add_vel(VectorBuiltin.RIGHT)

            case GrabberActions.MOVE_BACKWARD:
                if self._grabbed_product != None:
                    self._grabbed_product.add_vel(VectorBuiltin.BACKWARD)

            case GrabberActions.MOVE_FORWARD:
                if self._grabbed_product != None:
                    self._grabbed_product.add_vel(VectorBuiltin.FORWARD)

            case GrabberActions.ROTATE_CW:
                if self._grabbed_product != None:
                    self._grabbed_product.add_ang_vel(1)
                
            case GrabberActions.ROTATE_CCW:
                if self._grabbed_product != None:
                    self._grabbed_product.add_ang_vel(-1)

            case _: 
                pass 
    
    def _update(self):
        match(self._current_action):
            case GrabberActions.IDLE:
                pass 

            case GrabberActions.MOVE_LEFT:
                if self._grabbed_product != None and not is_equal(self._grabbed_product._transform_vel, VectorBuiltin.LEFT):
                    return 
                self._position += VectorBuiltin.LEFT
                self._position = np.clip(self._position, (0, 0), self._workspace_size)
            
            case GrabberActions.MOVE_RIGHT:
                if self._grabbed_product != None and not is_equal(self._grabbed_product._transform_vel, VectorBuiltin.RIGHT):
                    return 
                self._position += VectorBuiltin.RIGHT
                self._position = np.clip(self._position, (0, 0), self._workspace_size)

            case GrabberActions.MOVE_BACKWARD:
                if self._grabbed_product != None and not is_equal(self._grabbed_product._transform_vel, VectorBuiltin.BACKWARD):
                    return 
                self._position += VectorBuiltin.BACKWARD
                self._position = np.clip(self._position, (0, 0), self._workspace_size)

            case GrabberActions.MOVE_FORWARD:
                if self._grabbed_product != None and not is_equal(self._grabbed_product._transform_vel, VectorBuiltin.FORWARD):
                    return 
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
                if self._grabbed_product != None and self._grabbed_product._transform_ang_vel > 0:
                    self._grabbed_product.rotate(1)
                
            case GrabberActions.ROTATE_CCW:
                if self._grabbed_product != None and self._grabbed_product._transform_ang_vel < 0:
                    self._grabbed_product.rotate(-1)
                    
            case _: 
                pass 

    def _postupdate(self):
        if self._grabbed_product != None:
            self._grabbed_product.reset_vel()
            self._grabbed_product.reset_ang_vel()

        super()._postupdate()
    