from core.agent import *
from core.resource import _QuantityType, _ResourceType

_UtilityType = float 

@dataclass 
class ProductionJob:
    """
    Details on a production job.

    Note that `time` and `qty_produced` will be specified separately from `type`
    """
    prod_type : _ResourceType
    time : int = -1
    qty_produced : int = -1

@dataclass
class TradeActionInformation(ActionInformation):
    """
    Contains attributes and misc. information about an agent's actions.

    If the value is None, then that action was not taken 

    :pick_up: Action corresponding to picking up an object in the world
    :put_down:  Action corresponding to putting an item in the inventory down 
    :production_job:  Product to make at the moment
    """
    pick_up : Direction | None = None 
    put_down : Direction | None = None 
    production_job : _ResourceType | None = None 

    def reset(self):
        self.pick_up = None 
        self.put_down = None 
        self.production_job = None 

@dataclass 
class TradeAgentTraits(AgentTraits):
    """
    Data class containing fixed parameters / traits of the agent relevant to the simulation
    """
    pass 

@dataclass
class TradeAgentState(AgentState):
    """
    Contains attributes and misc. information about an agent's internal state.

    :param current_utility:  Currently calculated utility. If None, utility was not calculated. 
    :param current_mass_carried: The total mass being carried by the agent at the moment .
    :param inventory:  Current agent inventory
    :param relations:  Dictionary mapping agent ids to agent relations
    :param msgs:  The current message buffer
    :param skill: The vector representing the skill of the agent. Must be initialized.
    """

    current_utility : _UtilityType | None = None
    current_mass_carried : float = 0
    inventory : dict[_ResourceType, _QuantityType] = field(default_factory= lambda : {})
    relations : dict[int, int] = field(default_factory= lambda : {})
    msgs: list[Message] = field(default_factory=lambda : [])
    skills : np.ndarray | None = None 
    
    
    def reset(self, traits : AgentTraits):
        """
        Reset the state
        """
        self.current_utility = None
        self.inventory.clear()
        self.relations.clear()
        self.msgs.clear()
        self.current_mass_carried = 0
        self.skills  = None 
    
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


class TradeAgent(Agent):
    """
    SAR Agent Class. Override or augment this class as needed.

    This is meant to be a wrapper that allows interfacing between actions and states.
    """

    def __init__(self):
        """
        Initializes a simple agent 
        """
        super().__init__()

    def _reset(self):
        self._current_observation : LocalObservation = None
        self._current_action : TradeActionInformation= TradeActionInformation()
        self._traits : AgentTraits = TradeAgentTraits()
        self._current_state :  AgentState = TradeAgentState()
        self._utility_function : UtilityFunction = None 

        
    def reset(self):
        """
        Reset the agent
        """
        self._previous_position = None 
        self._current_state.reset(self._traits)

    def update(self):
        """
        Update the agent's state
        """
        pass 

    def make(self, prod : _ResourceType): 
        """
        Choose to make a product 
        """
        self._current_action.production_job = prod 

    def add_to_inventory(self, res : Resource) -> float:
        """
        Add resource to inventory, assuming it can be carried. Return excess mass.
        """
        current_carried = self._current_state.current_mass_carried
        excess_mass = max(0, current_carried + res.quantity - self._carrying_capacity)
        res.quantity = res.quantity - excess_mass

        self._current_state.add_to_inventory(res)
        return excess_mass
    

    def get_from_inventory(self, type : _ResourceType, qty : _QuantityType) -> _QuantityType:
        """
        Get a product from the inventory. Update the inventory.
        """
        return self._current_state.get_from_inventory(type, qty)
    
    def has_in_inventory(self, type : _ResourceType, qty : _QuantityType) -> bool: 
        """
        Check if the agent has the specified product in the specified amount
        """
        return self._current_state.has_in_inventory(type, qty)

    @property
    def action(self) -> ActionInformation:
        """
        Return the current action of the agent 
        """
        return self._current_action