from __future__ import annotations 
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from .action import *

from .env_params import * 

class BaseDynamicsModel(ABC):
    def __init__(self):
        pass 
    
    @abstractmethod
    def forward(self, *args):
        pass 


