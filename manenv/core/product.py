from __future__ import annotations

from abc import abstractmethod, ABC
import numpy as np
import random

from manenv.core.idpool import IDPool


from ..utils.vector import *
from ..utils.product_utils import *

class Product:
    """
    A product produced and consumed by the smart factory. To simplify things, a product is specified using a matrix
    """
    def __init__(self, structure : np.ndarray):
        """
        `structure` - an array that represents the product's structure. The array must be of datatype int. 
        """
        self._structure : np.ndarray = trim_structure_array(structure)
        self._id = IDPool.get()

        self._transform_pos = make_vector(0, 0)
        self._transform_vel = make_vector(0, 0)
        self._transform_ang_vel = 0

        self._is_dirty = False 
        
    def add_vel(self, vel : Vector):
        self._transform_vel += vel 

    def reset_vel(self):
        self._transform_vel *= 0

    def add_ang_vel(self, rot : int):
        self._transform_ang_vel += rot

    def reset_ang_vel(self):
        self._transform_ang_vel = 0

    def update(self):
        self._transform_pos += self._transform_vel

    def delete(self):
        IDPool.pop(self._id)
        self._is_dirty = True

    def copy(self):
        return Product(structure=self._structure)
    
    def rotate(self, k):
        self._structure =  rotate_structure(self._structure, k)
    
    def __str__(self):
        return "id: " + str(self._id) + "\n" +  str(self._structure)
    