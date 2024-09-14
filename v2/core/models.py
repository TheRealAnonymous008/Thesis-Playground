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
    
from dataclasses import dataclass, field
from .resource import Resource, _ResourceType, _QuantityType

_ProductType = _ResourceType

@dataclass 
class Recipe: 
    """
    Data class holding info about how products can be made

    `tgt_prod` - what is being made
    `tgt_qty` - how much will be made
    `time` - how long will it take
    `requirements` - what are needed to make the product
    """
    tgt_prod : _ResourceType = 0 
    tgt_qty : _QuantityType = 1
    time : int = 1
    requirements : dict[_ResourceType, _QuantityType] = field(default_factory= lambda : {})

    def add_requirement(self, rsrc_type : _ResourceType, qty : _QuantityType):
        self.requirements[rsrc_type] = qty 

class ChemistryModel: 
    def __init__(self, resource_types = RESOURCE_TYPES, product_types = PRODUCT_TYPES): 
        """
        Model for making products. 

        Note that we treat products as another form of resource
        """
        self.resource_types = resource_types
        self.product_types = product_types
        self.total_types = resource_types + product_types
        self._initialize_recipes()

    def _initialize_recipes(self):
        self.recipes : dict[_ResourceType, Recipe] = {}

        for prod_type in range(self.product_types):
            prod_rsrc_id : _ResourceType= self.get_resource_id(prod_type)
            recipe : Recipe = Recipe(tgt_prod= prod_rsrc_id, tgt_qty = 1)
            
            recipe.add_requirement(1, 2)

            self.recipes[prod_rsrc_id] = recipe
            break 


    def get_resource_id(self, raw_prod_type : _ProductType) -> _ResourceType :
        """
        Returns the id of a product given its type. 
        """
        return raw_prod_type + self.resource_types + 1

    def forward(self, agent : Agent, prod : _ResourceType):
        """
        Produce a product based on an initialized recipe and on the agent skill. 

        Updates the agent's inventory and returns a Resource object representing the product made. 
        """ 
        # Check if there is a way to make the product
        if not prod in self.recipes: 
            return None 
        # Check if the agent can make the product 
        recipe = self.recipes[prod]
        for t, q in recipe.requirements.items():
            if not agent.has_in_inventory(t, q):
                return None 
        
        for t, q in recipe.requirements.items():
            agent.get_from_inventory(t, q)
        agent.add_to_inventory(Resource(recipe.tgt_prod, recipe.tgt_qty))