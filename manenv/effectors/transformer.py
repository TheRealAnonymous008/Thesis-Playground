from enum import Enum

from manenv.asset_paths import AssetPath
from manenv.core.effector import Effector
from manenv.core.product import Product
from manenv.utils.vector import *
from manenv.utils.product_utils import *

class TransformerActions(Enum):
    IDLE = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_FORWARD = 3
    MOVE_BACKWARD = 4
    TRANSFORM = 5
    

class Transformer(Effector):
    def __init__(self,  p_in : int, p_out : int, position : Vector = None,):
        """
        A transformer transforms a resource indexed t1 to a resource indexed t2
        """
        assert(p_in > 0 and p_out > 0 and p_in < len(AssetPath.PRODUCT_ASSETS) and p_out < len(AssetPath.PRODUCT_ASSETS))

        super().__init__(TransformerActions, AssetPath.TRANSFORMER, position)
        self._starting_position = position
        # The job details contain the new product welded and where to place it on the
        # workspace
        self._transform_job_details = None
        self.p_in = p_in 
        self.p_out = p_out

    def _preupdate(self):
        super()._preupdate()
        if self._transform_job_details != None: 
            product, position = self._transform_job_details
            self._assembler.place_in_workspace(product, position)
            self._transform_job_details = None

        match(self._current_action):
            case _: 
                pass 
    
    def _update(self):
        match(self._current_action):
            case TransformerActions.IDLE.value:
                pass 

            case TransformerActions.MOVE_LEFT.value:
                self._position += VectorBuiltin.LEFT
                self._position = np.clip(self._position, (0, 0), self._workspace_size - VectorBuiltin.ONE_VECTOR)
            
            case TransformerActions.MOVE_RIGHT.value:
                self._position += VectorBuiltin.RIGHT
                self._position = np.clip(self._position, (0, 0), self._workspace_size - VectorBuiltin.ONE_VECTOR)

            case TransformerActions.MOVE_BACKWARD.value:
                self._position += VectorBuiltin.BACKWARD
                self._position = np.clip(self._position, (0, 0), self._workspace_size - VectorBuiltin.ONE_VECTOR)
                
            case TransformerActions.MOVE_FORWARD.value:
                self._position += VectorBuiltin.BACKWARD
                self._position = np.clip(self._position, (0, 0), self._workspace_size - VectorBuiltin.ONE_VECTOR)

            case TransformerActions.TRANSFORM.value:
                job_details = self._transform_product()
                self._transform_job_details = job_details

            case _: 
                pass 

    def _transform_product(self):
            p1 = self._assembler.get_product_in_workspace(self._position)

            if p1 == None:
                return None

            self._assembler.delete_product_in_workspace(self._position)

            # Transform the product
            # Transformation takes one resource and then maps it to another
            t1 = p1._transform_pos
            structure = np.zeros(self._assembler._workspace_size)
            structure = place_structure(p1._structure, structure, t1)

            structure[self._position[0]][self._position[1]] = self.p_out
            
            self.do_work()
            return Product(structure), t1


    def _postupdate(self):
        super()._postupdate()

    def reset(self):
        # The job details contain the new product welded and where to place it on the
        # workspace
        self._position = self._starting_position
        self._transform_job_details = None

    