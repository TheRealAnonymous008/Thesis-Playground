from __future__ import annotations 
from dataclasses import dataclass, field
from core.resource import Resource, _ResourceType, _QuantityType
from core.models import * 


_ProductType = _ResourceType

@dataclass 
class Recipe: 
    """
    Data class holding info about how products can be made

    :param tgt_prod: What is being made
    :param tgt_qty: How much will be made
    :param time: How long will it take
    :param requirements: What are needed to make the product
    """
    tgt_prod : _ResourceType = 0 
    tgt_qty : _QuantityType = 1
    time : int = 1
    requirements : dict[_ResourceType, _QuantityType] = field(default_factory= lambda : {})

    def add_requirement(self, rsrc_type : _ResourceType, qty : _QuantityType):
        self.requirements[rsrc_type] = qty 


class ChemistryModel(BaseDynamicsModel): 
    def __init__(self, resource_types = RESOURCE_TYPES, product_types = PRODUCT_TYPES): 
        """
        Model for making products. 

        Note that we treat products as another form of resource

        :param resource_types: Number of resource types possible in the environment
        :param product_types: Number of possible products in the environment 
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
