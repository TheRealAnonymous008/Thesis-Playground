from __future__ import annotations
from typing import Literal, Tuple
from enum import Enum

from abc import abstractmethod, ABC

import gymnasium

from manenv.core.actor import Actor
from ..utils.vector import *
import numpy as np
from ..asset_paths import AssetPath

from typing import TYPE_CHECKING
if TYPE_CHECKING: 
    from ..components import Assembler
    from .product import Product

class Effector(Actor):
    """
    An effector models the actions possible by a robot assembly agent .
    """

    def __init__(self, action_space : Enum, asset = "", position = make_vector(0, 0)):
        """
        `action_space` - defines the list of actions this Effector type can perform
        
        `asset` - the path to the asset to render this effector

        `position` - the location of the effector head
        
        """
        super().__init__()
        self._assembler : Assembler = None 
        self._action_space = action_space 

        self._position : Vector = position
        self._asset = asset

        self._current_action = None 

    def bind(self, assembler : Assembler):
        self._assembler = assembler
        self._workspace_size = self._assembler._workspace_size.copy() - make_vector(1, 1)

    def is_bound(self):
        return self._assembler != None
    
    def set_action(self, action_code: int):
        self._current_action = action_code

    def get_observation(self):
        return {
            "position": self._position,
            "r_mask": self._assembler._workspace,
            "p_mask": self._assembler._product_mask,
        }
    
    def get_observation_space(self):
        return gymnasium.spaces.Dict({
            "position": gymnasium.spaces.Box(low=np.array(0, 0), high = np.array(self._assembler._workspace_size[0], self._assembler._workspace_size[1])),
            "r_mask": gymnasium.spaces.Box(),
            "p_mask": gymnasium.spaces.Box()
        })

    @abstractmethod
    def _preupdate(self):
        pass 

    @abstractmethod
    def _update(self):
        pass 

    @abstractmethod
    def _postupdate(self):
        self._current_action = None 

    def reset(self):
        pass