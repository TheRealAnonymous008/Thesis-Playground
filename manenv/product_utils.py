import numpy as np 
from .vector import *

def trim_structure_array(array: np.ndarray) -> np.ndarray:
    """
    Given the input `array` returns a new array where zeros on the edge are trimmed.

    For example: 
    
    ```
    0 0 0 
    0 1 2
    0 0 0
    ```

    becomes 

    ```
    1 2
    ```
    """

    non_zero_indices = np.nonzero(array)
    
    # Find the minimum and maximum row and column indices of non-zero elements
    min_row = np.min(non_zero_indices[0])
    max_row = np.max(non_zero_indices[0])
    min_col = np.min(non_zero_indices[1])
    max_col = np.max(non_zero_indices[1])
    
    # Return the bounding box coordinates
    return array[min_row:max_row+1, min_col:max_col+1]

def place_structure(obj: np.ndarray, target: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    Places the `obj` array at the specified `pos` within `target`. Returns the target ndarray
    """
    # Get the dimensions of the object array and the target array
    obj_height, obj_width = obj.shape
    target_height, target_width = target.shape
    
    # Get the position to place the object
    pos_row = pos[0]
    pos_col = pos[1]
    
    # Check if the object fits within the target array at the specified position
    if pos_row + obj_height > target_height or pos_col + obj_width > target_width:
        raise ValueError("Object does not fit within the target array at the specified position.")
    
    # Place the object within the target array
    target[pos_row:pos_row+obj_height, pos_col:pos_col+obj_width] = obj
    
    return target


def check_bounds(array1, array2) -> bool:
    """
    Returns true if array1 is in bound with array2
    """
    if len(array1) != len(array2):
        return False
    
    for i in range(len(array1)):
        if array1[i] < 0 or array1[i] >= array2[i]:
            return False
        
    return True