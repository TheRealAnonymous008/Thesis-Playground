from __future__ import annotations

import numpy as np

from .action import *
from .message import * 
from dataclasses import dataclass, field

from .resource import Resource, _QuantityType, _ResourceType

_UtilityType = float 

@dataclass
class AgentState:
    """
    Contains attributes and misc. information about an agent's internal state.

    :param current_energy: Current energy of the agent
    :param current_utility:  Currently calculated utility. If None, utility was not calculated. 
    :param current_mass_carried: The total mass being carried by the agent at the moment .
    :param inventory:  Current agent inventory
    :param relations:  Dictionary mapping agent ids to agent relations
    :param msgs:  The current message buffer
    :param skill: The vector representing the skill of the agent. Must be initialized.
    """

    current_energy : float = 0
    current_utility : _UtilityType | None = None
    current_mass_carried : float = 0
    inventory : dict[_ResourceType, _QuantityType] = field(default_factory= lambda : {})
    relations : dict[int, int] = field(default_factory= lambda : {})
    msgs: list[Message] = field(default_factory=lambda : [])
    skills : np.ndarray | None = None 
    
    
    def reset(self):
        """
        Reset the state
        """
        self.current_energy = 0
        self.current_utility = None
        self.inventory.clear()
        self.relations.clear()
        self.msgs.clear()
        self.current_mass_carried = 0
        self.skills  = None 

    @property
    def can_move(self):
        return self.current_energy > 0
    
    def add_to_inventory(self, resource : Resource):
        """
        Add a resource to the inventory
        """
        self.current_mass_carried += resource.quantity
        if resource.type in self.inventory:
            self.inventory[resource.type] += resource.quantity
        else: 
            self.inventory[resource.type] = resource.quantity

    def get_from_inventory(self, resource_type : _ResourceType, qty : _QuantityType ) -> _QuantityType:
        """
        Gets resources from the inventory. Mutates the inventory
        """

        if resource_type not in self.inventory: 
            return 0
        if self.inventory[resource_type] < qty: 
            return 0 
            
        self.inventory[resource_type] -= qty
        return qty
    
    def has_in_inventory(self, resource_type : _ResourceType, qty : _QuantityType ) -> bool:
        """
        Gets resources from the inventory.
        """
        if resource_type not in self.inventory: 
            return False 
        if self.inventory[resource_type] < qty: 
            return False 
            
        return True 
    
    def get_qty_in_inventory(self, resource_type : _ResourceType) -> int:
        """
        Returns how many of the specified type are in inventory
        """
        if resource_type not in self.inventory: 
            return 0
        return self.inventory[resource_type]
 
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
