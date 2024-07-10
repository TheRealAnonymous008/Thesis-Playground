
from enum import Enum

from manenv.asset_paths import AssetPath
from manenv.core.effector import Effector
from manenv.core.product import Product
from manenv.utils.vector import *

class GrabberActions(Enum):
    IDLE = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_FORWARD = 3
    MOVE_BACKWARD = 4
    GRAB = 5
    RELEASE = 6 
    GRAB_INVENTORY = 7
    ROTATE_CW = 9
    ROTATE_CCW = 10

class Grabber(Effector):
    def __init__(self, position : Vector = None):
        super().__init__(GrabberActions, AssetPath.GRABBER, position)
        self._starting_position : Vector = self._position.copy()
        self._grabbed_product : Product = None

    def _preupdate(self):
        super()._preupdate()

        if self._grabbed_product != None and self._grabbed_product._is_dirty:
            self._grabbed_product = None
        
        match(self._current_action):
            case GrabberActions.IDLE.value:
                pass 

            case GrabberActions.MOVE_LEFT.value:
                if self._grabbed_product != None:
                    self._grabbed_product.add_vel(VectorBuiltin.LEFT)
            
            case GrabberActions.MOVE_RIGHT.value:
                if self._grabbed_product != None:
                    self._grabbed_product.add_vel(VectorBuiltin.RIGHT)

            case GrabberActions.MOVE_BACKWARD.value:
                if self._grabbed_product != None:
                    self._grabbed_product.add_vel(VectorBuiltin.BACKWARD)

            case GrabberActions.MOVE_FORWARD.value:
                if self._grabbed_product != None:
                    self._grabbed_product.add_vel(VectorBuiltin.FORWARD)

            case GrabberActions.ROTATE_CW.value:
                if self._grabbed_product != None:
                    self._grabbed_product.add_ang_vel(1)
                
            case GrabberActions.ROTATE_CCW.value:
                if self._grabbed_product != None:
                    self._grabbed_product.add_ang_vel(-1)

            case _: 
                pass 
    
    def _update(self):
        match(self._current_action):
            case GrabberActions.IDLE.value:
                pass 

            case GrabberActions.MOVE_LEFT.value:
                if self._grabbed_product != None and not is_equal(self._grabbed_product._transform_vel, VectorBuiltin.LEFT):
                    return 
                self._position += VectorBuiltin.LEFT
                self._position = np.clip(self._position, (0, 0), self._workspace_size)
            
            case GrabberActions.MOVE_RIGHT.value:
                if self._grabbed_product != None and not is_equal(self._grabbed_product._transform_vel, VectorBuiltin.RIGHT):
                    return 
                self._position += VectorBuiltin.RIGHT
                self._position = np.clip(self._position, (0, 0), self._workspace_size)

            case GrabberActions.MOVE_BACKWARD.value:
                if self._grabbed_product != None and not is_equal(self._grabbed_product._transform_vel, VectorBuiltin.BACKWARD):
                    return 
                self._position += VectorBuiltin.BACKWARD
                self._position = np.clip(self._position, (0, 0), self._workspace_size)

            case GrabberActions.MOVE_FORWARD.value:
                if self._grabbed_product != None and not is_equal(self._grabbed_product._transform_vel, VectorBuiltin.FORWARD):
                    return 
                self._position += VectorBuiltin.FORWARD
                self._position = np.clip(self._position, (0, 0), self._workspace_size)

            case GrabberActions.GRAB.value:
                self._grabbed_product = self._assembler.get_product_in_workspace(self._position)

            case GrabberActions.RELEASE.value:
                if self._grabbed_product != None:
                    self._assembler.place_in_workspace(self._grabbed_product, self._grabbed_product._transform_pos)
                    self._grabbed_product = None 
            
            case GrabberActions.GRAB_INVENTORY.value:
                inventory = self._assembler.get_product_inventory()
                if len(inventory) > 0:
                    self._grabbed_product = inventory.pop(0)
                    self._grabbed_product._transform_pos = self._position.copy()

            case GrabberActions.ROTATE_CW.value:
                if self._grabbed_product != None and self._grabbed_product._transform_ang_vel > 0:
                    self._grabbed_product.rotate(1)
                
            case GrabberActions.ROTATE_CCW.value:
                if self._grabbed_product != None and self._grabbed_product._transform_ang_vel < 0:
                    self._grabbed_product.rotate(-1)
                    
            case _: 
                pass 

    def _postupdate(self):
        if self._grabbed_product != None:
            if np.linalg.norm(self._grabbed_product._transform_vel) <= 1:
                self._grabbed_product.update()
            self._grabbed_product.reset_vel()
            self._grabbed_product.reset_ang_vel()

        super()._postupdate()
    
    def reset(self):
        self._position = self._starting_position.copy()
        if self._grabbed_product:
            self._grabbed_product.delete()
        self._grabbed_product = None