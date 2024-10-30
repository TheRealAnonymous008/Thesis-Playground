from __future__ import annotations 
from abc import ABC, abstractmethod
from .action import *

class BaseDynamicsModel(ABC):
    def __init__(self):
        pass 
    
    @abstractmethod
    def forward(self, *args):
        pass 


