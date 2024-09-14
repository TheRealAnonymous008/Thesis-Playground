import numpy as np
from enum import Enum

from .agent import Agent
from .action import *

from .env_params import * 

class EnergyModel:
    def __init__(self):
        pass 

    def forward(self, agent : Agent) -> float: 
        """
        Compute the energy consumption of an egent and update its current state.

        Returns the total energy consumed
        """
        action = agent.action
        total_energy_consumption = 0
        
        if action.movement != None and agent.has_moved: 
            e = np.random.normal(0.5, 0.25)
            total_energy_consumption += max(0.1, e)

        agent._current_state.current_energy -= total_energy_consumption
        agent._current_state.current_energy = max(0, agent._current_state.current_energy)

        return total_energy_consumption
    
_ProductTypeId = int
class ChemistryModel: 
    def __init__(self, resource_types = RESOURCE_TYPES, product_types = PRODUCT_TYPES): 
        """
        Model for making products. 

        Note that we treat products as another form of resource
        """
        self.resource_types  = resource_types
        self.product_types = product_types
        self._initialize_recipes()

    def _initialize_recipes(self):
        pass 

    def get_product_id(self, raw_prod_type : _ProductTypeId) :
        """
        Returns the id of a product given its type. 
        """
        return raw_prod_type + self.resource_types

    def forward(self, agent : Agent, prod : _ProductTypeId):
        """
        Produce a product based on an initialized recipe and on the agent skill. 

        Updates the agent's inventory
        """ 
        inventory = agent.get_inventory()