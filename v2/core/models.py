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


