from __future__ import annotations 
from typing import TYPE_CHECKING


import numpy as np
from enum import Enum
from abc import ABC, abstractmethod

from .world import BaseWorld 

from .agent import Agent
from .utility import UtilityFunction
from .action import *

from .env_params import * 

class BaseDynamicsModel(ABC):
    def __init__(self):
        pass 
    
    @abstractmethod
    def forward(self, *args):
        pass 




TOTAL_FEATURES = PRODUCT_TYPES + RESOURCE_TYPES
class UtilitySampler:
    def __init__(self, dims = TOTAL_FEATURES ):
        self.dims = dims 
    
    def forward(self, agent : Agent): 
        params = np.random.normal(0, 1, self.dims)
        bias = np.random.normal(-2, 2)
        agent.set_utility(UtilityFunction(params, bias))