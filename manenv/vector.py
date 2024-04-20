import numpy as np 

type Vector = type[np.ndarray] 

def make_vector(x : int | float, y : int | float) -> Vector: 
    return np.array([x, y])

ZERO_VECTOR = make_vector(0, 0)