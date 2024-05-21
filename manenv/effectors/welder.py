from enum import Enum

from manenv.asset_paths import AssetPath
from manenv.effector import Effector
from manenv.product import Product
from manenv.vector import *
from manenv.product_utils import *

class WelderActions(Enum):
    IDLE = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_FORWARD = 3
    MOVE_BACKWARD = 4
    WELD_NORTH = 5
    WELD_SOUTH = 6
    WELD_EAST = 7
    WELD_WEST = 8
    

class Welder(Effector):
    def __init__(self, position : Vector = None):
        super().__init__(WelderActions, AssetPath.WELDER, position)

    def _preupdate(self):
        super()._preupdate()

        match(self._current_action):
            case _: 
                pass 
    
    def _update(self):
        match(self._current_action):
            case WelderActions.IDLE:
                pass 

            case WelderActions.MOVE_LEFT:
                self._position += VectorBuiltin.LEFT
                self._position = np.clip(self._position, (0, 0), self._workspace_size)
            
            case WelderActions.MOVE_RIGHT:
                self._position += VectorBuiltin.RIGHT
                self._position = np.clip(self._position, (0, 0), self._workspace_size)

            case WelderActions.MOVE_BACKWARD:
                self._position += VectorBuiltin.BACKWARD
                self._position = np.clip(self._position, (0, 0), self._workspace_size)
                
            case WelderActions.MOVE_FORWARD:
                self._position += VectorBuiltin.BACKWARD
                self._position = np.clip(self._position, (0, 0), self._workspace_size)

            case WelderActions.WELD_NORTH:
                pass 

            case WelderActions.WELD_SOUTH:
                pass 

            case WelderActions.WELD_EAST:
                pass 

            case WelderActions.WELD_WEST:
                p1 : Product = self._assembler.get_product_in_workspace(self._position)
                p2 : Product = self._assembler.get_product_in_workspace(self._position + VectorBuiltin.LEFT)

                if p1 == None or p2 == None:
                    return 

                self._assembler.delete_product_in_workspace(p1)
                self._assembler.delete_product_in_workspace(p2)

            case _: 
                pass 

    def _postupdate(self):
        super()._postupdate()

    