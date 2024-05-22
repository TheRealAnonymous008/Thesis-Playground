from enum import Enum

from manenv.asset_paths import AssetPath
from manenv.core.effector import Effector
from manenv.core.product import Product
from manenv.utils.vector import *
from manenv.utils.product_utils import *

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
        # The job details contain the new product welded and where to place it on the
        # workspace

        self._weld_job_details = None

    def _preupdate(self):
        super()._preupdate()
        if self._weld_job_details != None: 
            product, position = self._weld_job_details
            self._assembler.place_in_workspace(product, position)
            self._weld_job_details = None

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
                job_details = self._weld_at_offset(self._position + VectorBuiltin.FORWARD)
                self._weld_job_details = job_details

            case WelderActions.WELD_SOUTH:
                job_details = self._weld_at_offset(self._position + VectorBuiltin.BACKWARD)
                self._weld_job_details = job_details

            case WelderActions.WELD_EAST:
                job_details = self._weld_at_offset(self._position + VectorBuiltin.RIGHT)
                self._weld_job_details = job_details
                
            case WelderActions.WELD_WEST:
                job_details = self._weld_at_offset(self._position + VectorBuiltin.LEFT)
                self._weld_job_details = job_details

            case _: 
                pass 

    def _weld_at_offset(self, offset: Vector):
            p1 : Product = self._assembler.get_product_in_workspace(self._position)
            p2 : Product = self._assembler.get_product_in_workspace(offset)

            if p1 == None or p2 == None:
                return None

            self._assembler.delete_product_in_workspace(self._position)
            self._assembler.delete_product_in_workspace(offset)

            # Weld the two products together
            # Welding is defined by returning a new product that is the union of the two constituents 
            t1 = p1._transform_pos
            t2 = p2._transform_pos
            
            structure = np.zeros(self._assembler._workspace_size)
            structure = place_structure(p1._structure, structure, t1)
            structure = place_structure(p2._structure, structure, t2)
            
            return Product(structure), get_min(t1, t2)


    def _postupdate(self):
        super()._postupdate()

    