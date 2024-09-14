from __future__ import annotations

import numpy as np
from enum import Enum

from .observation import LocalObservation

from .action import *
from .map import Resource
from .message import * 
from dataclasses import dataclass, field

from .resource import Resource, _QuantityType, _ResourceType

@dataclass
class AgentSate:
    """
    Contains attributes and misc. information about an agent's internal state.

    `current_energy` - current energy of the agent
    `current_mass_carried` = the total mass being carried by the agent at the moment .
    `inventory` - current agent inventory
    `relations` - dictionary mapping agent ids to agent relations
    `msgs` - the current message buffer
    `skill` - the vector representing the skill of the agent. Must be initialized.
    """

    current_energy : float = 0
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


class Agent:
    def __init__(self):
        """
        Initializes a simple agent 
        """
        self._id = -1

        self._previous_position : np.ndarray[int] | None = None 
        self._current_position : np.ndarray[int] | None =  None 

        self._current_observation : LocalObservation = None
        self._current_action : ActionInformation = ActionInformation()
        self._current_state :  AgentSate = AgentSate()

        # Attributes of the agent. 
        self._visibility_range : int = 3
        self._energy_capacity : float = 100.0
        self._carrying_capacity : float = 100.0

        self.reset()
        
    def reset(self):
        """
        Reset the agent
        """
        self._previous_position = None 
        self._current_state.reset()
        self._current_state.current_energy = self._energy_capacity

    def move(self, dir : Direction | int):
        """
        Moves an agent along a specified direction. 
        The direction is either a Direction instance or an integer associated with a Direction value.
        """
        if not self._current_state.can_move:
            return 

        if type(dir) is Direction: 
            val = dir.value
        else: 
            val = dir 

        match(val):
            case Direction.NORTH.value: 
                self._current_action.movement = Direction.NORTH

            case Direction.SOUTH.value: 
                self._current_action.movement = Direction.SOUTH

            case Direction.EAST.value: 
                self._current_action.movement = Direction.EAST
            
            case Direction.WEST.value:
                self._current_action.movement = Direction.WEST

            case _: 
                raise Exception(f"Invalid direction specified {val}")

    def pick_up(self, dir : Direction | int ): 
        """
        Pick up a resource adjacent to this agent 
        """
        if not self._current_state.can_move:
            return 

        if type(dir) is Direction: 
            val = dir.value
        else: 
            val = dir 

        match(val):
            case Direction.NORTH.value: 
                self._current_action.pick_up = Direction.NORTH

            case Direction.SOUTH.value: 
                self._current_action.pick_up = Direction.SOUTH

            case Direction.EAST.value: 
                self._current_action.pick_up = Direction.EAST
            
            case Direction.WEST.value:
                self._current_action.pick_up = Direction.WEST

            case _: 
                raise Exception(f"Invalid direction specified {val}")

    def put_down(self, dir : Direction | int ): 
        """
        Put down the held resource to somewhere adjacent to the agent.
        """
        if not self._current_state.can_move:
            return 

        if type(dir) is Direction: 
            val = dir.value
        else: 
            val = dir 

        match(val):
            case Direction.NORTH.value: 
                self._current_action.pick_up = Direction.NORTH

            case Direction.SOUTH.value: 
                self._current_action.pick_up = Direction.SOUTH

            case Direction.EAST.value: 
                self._current_action.pick_up = Direction.EAST
            
            case Direction.WEST.value:
                self._current_action.pick_up = Direction.WEST

            case _: 
                raise Exception(f"Invalid direction specified {val}")

    def talk_to(self, agent : Agent):
        """
        Communicate with the agent. Construct a message to send to this agent 
        """
        self.send_message(agent, Message(self._id, 1))
    
    def send_message(self, agent : Agent, message : Message):
        """
        Send a message to the target agent
        """
        agent._current_state.msgs.append(message)

    def get_messages(self) -> list[Message]:
        """
        Return all messages 
        """
        return self._current_state.msgs

    def clear_messages(self):
        """
        Clear all messages 
        """
        self._current_state.clear_messages() 
    
    def add_relation(self, agent_id : int, weight : float):
        """
        Add a social relation between this agent and the one specified
        """
        self._current_state.set_relation(agent_id, weight  + self._current_state.get_relation(agent_id))

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


    def reset_for_next_action(self):
        """
        Resets the agent for a new action
        """
        self._current_action.reset()

    def set_observation(self, observation : LocalObservation):
        """
        Set local observation
        """
        self._current_observation = observation

    @property
    def agents_in_range(self) -> list[int]:
        """
        Returns a list of id's of all visible agents.
        """
        return self._current_observation.neighbors(self._id)

    @property
    def local_observation(self) -> LocalObservation:
        """
        Get local observation
        """
        return self._current_observation

    def bind_to_world(self, world_id : int):
        """
        Set the agent's ID to be that of the ID assigned to it by the world
        """
        self._id = world_id

    @property
    def id(self) -> int:
        """
        Return the ID of this agent in the world
        """
        return self._id 

    def __hash__(self) -> int:
        return self._id

    def set_position(self, position : np.array):
        """
        Set the agent's position to `position`
        """
        self._previous_position = self._current_position
        self._current_position = position
    
    @property
    def previous_position(self) -> np.ndarray | None :
        """
        Get a copy of the agent's previous position if it is defined
        """
        if self._previous_position == None:
            return None 
        return self._previous_position.copy()

    @property
    def current_position(self) -> np.ndarray:
        """
        Get the agent's current position
        """
        return self._current_position.copy()
    
    @property
    def current_position_const(self) -> np.ndarray:
        """
        Get the agent's current position. Does not return a copy! Never mutate the output of this function
        """
        return self._current_position

    @property
    def action(self) -> ActionInformation:
        """
        Return the current action of the agent 
        """
        return self._current_action
    
    @property
    def has_moved(self) -> bool:
        """
        Return if the agent has been displaced
        """
        if self._previous_position is None: 
            return False 
        return  np.all(self._previous_position == self._current_position)
        
        