import numpy as np 
from typing import TypeAlias

Vector : TypeAlias = type[np.ndarray] 

def make_vector(x : int | float, y : int | float, dtype = int) -> Vector: 
    return np.array([x, y], dtype=dtype)

class VectorBuiltin:
    """
    Warning. Do not use these to assign values. Use these only for checking.
    """
    ZERO_VECTOR = make_vector(0, 0)
    FORWARD = make_vector(1, 0)
    BACKWARD = make_vector(-1, 0)
    LEFT = make_vector(0, -1)
    RIGHT = make_vector(0, 1)
    ONE_VECTOR = make_vector(1, 1)

def is_equal(x : Vector, y : Vector) -> bool:
    return np.all(x == y)

def get_min(x : Vector, y : Vector) -> Vector:
    return make_vector(min(x[0], y[0]), min(x[1], y[1]))