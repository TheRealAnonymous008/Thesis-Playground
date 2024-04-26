import numpy as np 

type Vector = type[np.ndarray] 

def make_vector(x : int | float, y : int | float) -> Vector: 
    return np.array([x, y])

class VectorBuiltin:
    ZERO_VECTOR = make_vector(0, 0)
    FORWARD = make_vector(0, 1)
    BACKWARD = make_vector(0, -1)
    LEFT = make_vector(-1, 0)
    RIGHT = make_vector(1, 0)
