from __future__ import annotations
from dataclasses import dataclass

from enum import Enum
from typing import TYPE_CHECKING
import numpy as np


from abc import ABC, abstractmethod
if TYPE_CHECKING: 
    from core.agent import Agent 
    from core.world import BaseWorld

@dataclass
class ActionInformation(ABC):
    """
    Contains attributes and misc. information about an agent's actions.
    """
    @abstractmethod
    def reset(self):
        pass 


class BaseActionParser(ABC):
    """
    Handles how actions are interpreted.
    """
    @abstractmethod
    def take_action(self, code : int, agent : Agent): 
        pass 

    @abstractmethod 
    def get_action_space(self, agent : Agent):
        pass 

    @abstractmethod
    def get_observation_space(self, agent : Agent):
        pass

    @abstractmethod
    def get_observation(self, agent : Agent):
        pass 

    @abstractmethod
    def get_action_mask(self, agent : Agent, world : BaseWorld, device : str = "cpu"):
        pass 