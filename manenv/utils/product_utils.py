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
        raise ValueError(f"Object does not fit within the target array at the specified position. {obj, target, pos}")
    
    # Place the object within the target array
    target[pos_row:pos_row+obj_height, pos_col:pos_col+obj_width] = obj
    
    return target

def is_region_zeros(mask: np.ndarray, obj: np.ndarray, pos: np.ndarray) -> bool:
    """
    Checks if the region in `obj` is all zeros based on a provided `mask` whose top left corner is placed in `pos`
    """

    # Determine the dimensions of the mask and object
    mask_height, mask_width = mask.shape
    obj_height, obj_width = obj.shape
    
    # Calculate the maximum valid region based on position
    start_row, end_row = max(0, pos[0]), min(obj_height, pos[0] + mask_height)
    start_col, end_col = max(0, pos[1]), min(obj_width, pos[1] + mask_width)
    
    obj_adjusted = obj[
        start_row:min(obj_height, end_row),
        start_col:min(obj_width, end_col)
    ]
    
    # Check if the region in the object covered by the mask is all zeros
    return np.all(obj_adjusted[mask != 0] == 0)

def rotate_structure(obj : np.ndarray, rots: int) -> np.ndarray: 
    """
    Rotate the input `obj` array by the specified `rots` amount (90 degree clockwise rotations)
    """
    return np.rot90(obj, -rots, axes=(0,1))
            

def check_bounds(array1 : np.ndarray, array2 : np.ndarray) -> bool:
    """
    Returns true if array1 is in bound with array2
    """
    if len(array1) != len(array2):
        return False
    
    for i in range(len(array1)):
        if array1[i] < 0 or array1[i] >= array2[i]:
            return False
        
    return True

def compare_structures(X : np.ndarray, Y : np.ndarray) -> float:
    """
    Returns the distance between two specified structures X and Y
    """
    if X.shape < Y.shape:
        X, Y = Y, X

    X_rows, X_cols = X.shape
    Y_rows, Y_cols = Y.shape
    
    # If the shapes are the same, compute similarity directly
    if X.shape == Y.shape:
        matches = np.sum(X == Y)
        total_entries = X.size
        similarity = matches / total_entries
        return similarity
    
    max_matches = 0
    
    # Slide Y over X and check for the best match
    for i in range(X_rows - Y_rows + 1):
        for j in range(X_cols - Y_cols + 1):
            sub_X = X[i:i+Y_rows, j:j+Y_cols]
            matches = np.sum(sub_X == Y)
            if matches > max_matches:
                max_matches = matches

    # Compute the similarity based on the best matching subarray
    total_entries = Y.size
    similarity = max_matches / total_entries
    
    return similarity