from __future__ import annotations

import numpy as np

from .action import *
from .message import * 
from dataclasses import dataclass, field

from .resource import Resource, _QuantityType, _ResourceType

import torch

_UtilityType = float 

@dataclass 
class AgentTraits:
    """
    Data class containing fixed parameters / traits of the agent relevant to the simulation
    """
    _device = "cpu"

    @abstractmethod
    def to_tensor(self) -> torch.Tensor:
        pass 
    
    @abstractmethod 
    def to_device(self, device : str):
        self._device = device

@dataclass
class AgentState:
    """
    Contains attributes and misc. information about an agent's internal state.
    """
    relations : dict[int, int] = field(default_factory= lambda : {})
    msgs: list[Message] = field(default_factory=lambda : [])
    skills : np.ndarray | None = None 
    
    
    def reset(self, traits : AgentTraits):
        """
        Reset the state
        """
        self.relations.clear()
        self.msgs.clear()

    def add_message(self, message : Message) : 
        """
        Add a message to the message buffer
        """
        self.msgs.append(message)

    def clear_messages(self):
        """
        Clear all messages
        """
        self.msgs.clear()

    def set_relation(self, agent : int, weight : float): 
        """
        Set the relation between this agent and the specified `agent`  to `weight`
        """
        self.relations[agent] = weight

    def get_relation(self, agent : int) -> float:
        """
        Get the relation between this agent and the specified `agent`. Defaults to 0
        """
        if agent in self.relations:
            return self.relations[agent]
        return 0

    def remove_relatioon(self, agent : int) : 
        """
        Remove a relation 
        """
        self.relations.pop(agent)